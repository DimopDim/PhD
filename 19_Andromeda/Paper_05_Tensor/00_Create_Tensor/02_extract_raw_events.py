#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_extract_raw_events.py

Parallel, memory-bounded Stage 02 of the clean multi-resolution tensor pipeline.

Purpose
-------
Extract only numeric ICU measurements belonging to the stroke cohorts generated
by 01_build_stroke_cohorts.py. Events are anchored to ICU admission and limited
to the available observation history within the first 48 hours.

This stage intentionally DOES NOT:
- harmonize MIMIC-IV and eICU variables,
- aggregate into cumulative windows,
- impute missing values,
- create model splits,
- construct tensors.

Parallel execution
------------------
Large compressed CSV files are read once by the parent process. The parent
performs the cheapest cohort-ID prefilter, then sends only cohort-relevant
chunks to a bounded ProcessPoolExecutor. Workers perform timestamp conversion,
numeric cleaning, cohort merge, observation-horizon filtering, and long-format
conversion. The parent is the only Parquet writer.

This avoids:
- every worker independently decompressing the same source file,
- unbounded accumulation of chunks in RAM,
- concurrent writes to the same Parquet file.

If pigz is installed, .csv.gz decompression can additionally use multiple
threads through --pigz-workers.

Canonical project root
----------------------
/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

Default inputs
--------------
Stage-01 cohorts:
    <project-root>/data/01_cohorts/

MIMIC-IV v3.1:
    /home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1

eICU v2.0:
    /home/ddimopoulos/Datasets/00_Datasets/eicu-2_0

Default outputs
---------------
    <project-root>/data/02_raw_events/

Common event schema
-------------------
database
patient_id
stay_id
source_table
source_variable
source_itemid
event_offset_minutes
event_offset_hours
value
unit

Example
-------
cd /home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

python 02_extract_raw_events.py \
    --database mimic \
    --workers 10 \
    --chunksize 500000 \
    --max-in-flight 20 \
    --pigz-workers 4

Then:

python 02_extract_raw_events.py \
    --database eicu \
    --workers 10 \
    --chunksize 500000 \
    --max-in-flight 20 \
    --pigz-workers 4
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor")
MIMIC_ROOT_DEFAULT = Path("/home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1")
EICU_ROOT_DEFAULT = Path("/home/ddimopoulos/Datasets/00_Datasets/eicu-2_0")

COHORT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "01_cohorts"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "02_raw_events"

HORIZON_HOURS_DEFAULT = 48.0
WORKERS_DEFAULT = 10
CHUNKSIZE_DEFAULT = 500_000
PIGZ_WORKERS_DEFAULT = 4

EVENT_COLUMNS = [
    "database",
    "patient_id",
    "stay_id",
    "source_table",
    "source_variable",
    "source_itemid",
    "event_offset_minutes",
    "event_offset_hours",
    "value",
    "unit",
]

# Worker-local immutable state. Each executor initializes this once.
_WORKER_STATE: Dict[str, object] = {}


def _worker_init(state: Dict[str, object]) -> None:
    global _WORKER_STATE
    _WORKER_STATE = state

    # Prevent BLAS/OpenMP libraries from creating additional thread pools inside
    # every process. The desired parallelism is process-level.
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Stage 02 requires pyarrow for incremental Parquet output.\n"
            "Install it in the active environment with:\n"
            "  pip install pyarrow"
        ) from exc


def empty_event_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "database": pd.Series(dtype="object"),
            "patient_id": pd.Series(dtype="object"),
            "stay_id": pd.Series(dtype="object"),
            "source_table": pd.Series(dtype="object"),
            "source_variable": pd.Series(dtype="object"),
            "source_itemid": pd.Series(dtype="object"),
            "event_offset_minutes": pd.Series(dtype="float64"),
            "event_offset_hours": pd.Series(dtype="float64"),
            "value": pd.Series(dtype="float64"),
            "unit": pd.Series(dtype="object"),
        }
    )[EVENT_COLUMNS]


class IncrementalParquetWriter:
    """Single-writer, incremental Parquet output with a stable schema."""

    def __init__(self, path: Path, compression: str = "zstd") -> None:
        require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.path = path
        self.compression = compression
        self.writer = None
        self.rows = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        table = self.pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(
                self.path,
                table.schema,
                compression=self.compression,
                use_dictionary=True,
            )
        else:
            if table.schema != self.writer.schema:
                table = table.cast(self.writer.schema)

        self.writer.write_table(table)
        self.rows += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            return

        # Preserve the expected output file even if a source has zero retained
        # events, making downstream stages deterministic.
        empty_event_frame().to_parquet(
            self.path,
            index=False,
            compression=self.compression,
        )


class OrderedParallelRunner:
    """
    Bounded producer -> process pool -> ordered single-writer pipeline.

    Results are written in source-chunk order. This keeps output deterministic
    while still allowing multiple chunks to be processed concurrently.
    """

    def __init__(
        self,
        *,
        worker_fn,
        state: Dict[str, object],
        writer: IncrementalParquetWriter,
        workers: int,
        max_in_flight: int,
        logger: logging.Logger,
        source_name: str,
    ) -> None:
        self.worker_fn = worker_fn
        self.writer = writer
        self.workers = workers
        self.max_in_flight = max_in_flight
        self.logger = logger
        self.source_name = source_name
        self.pending: Deque[Tuple[int, Future]] = deque()
        self.completed_chunks = 0
        self.retained_rows = 0

        if workers <= 1:
            self.executor = None
            _worker_init(state)
        else:
            # Andromeda is Linux. Explicit fork avoids repeated large-state
            # serialization and gives copy-on-write cohort lookup tables.
            context = mp.get_context("fork")
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_worker_init,
                initargs=(state,),
            )

    def submit(self, chunk_id: int, chunk: pd.DataFrame) -> None:
        if self.executor is None:
            result = self.worker_fn(chunk)
            self.writer.write(result)
            self.completed_chunks += 1
            self.retained_rows += len(result)
            return

        future = self.executor.submit(self.worker_fn, chunk)
        self.pending.append((chunk_id, future))

        if len(self.pending) >= self.max_in_flight:
            self._consume_oldest()

    def _consume_oldest(self) -> None:
        chunk_id, future = self.pending.popleft()
        try:
            result = future.result()
        except Exception as exc:
            raise RuntimeError(
                f"{self.source_name}: worker failed on submitted chunk "
                f"{chunk_id}"
            ) from exc

        self.writer.write(result)
        self.completed_chunks += 1
        self.retained_rows += len(result)

    def drain(self) -> None:
        while self.pending:
            self._consume_oldest()

    def close(self) -> None:
        try:
            self.drain()
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_raw_events")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        output_dir / "02_extract_raw_events.log",
        mode="a",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def find_table(directory: Path, stem: str) -> Path:
    """
    Resolve a source table robustly on case-sensitive Linux filesystems.

    MIMIC filenames are generally lowercase, whereas the official eICU
    distribution contains several camelCase names (for example customLab,
    nurseCharting, vitalPeriodic, vitalAperiodic).  We first try exact names
    and then perform a case-insensitive filename lookup.
    """
    candidates = [
        directory / f"{stem}.parquet",
        directory / f"{stem}.csv.gz",
        directory / f"{stem}.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path

    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    wanted = {
        f"{stem}.parquet".lower(),
        f"{stem}.csv.gz".lower(),
        f"{stem}.csv".lower(),
    }

    matches = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.name.lower() in wanted
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        # Deterministic preference: parquet -> csv.gz -> csv.
        suffix_rank = {
            ".parquet": 0,
            ".gz": 1,
            ".csv": 2,
        }
        matches = sorted(
            matches,
            key=lambda p: (suffix_rank.get(p.suffix.lower(), 99), p.name.lower()),
        )
        return matches[0]

    raise FileNotFoundError(
        f"Could not locate table '{stem}' in {directory}. "
        f"Lookup was case-insensitive and tried parquet/csv.gz/csv variants."
    )


def read_table(
    path: Path,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(
            path,
            columns=list(usecols) if usecols is not None else None,
        )

    return pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        low_memory=False,
    )


def read_header(path: Path) -> List[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).schema.names
    return pd.read_csv(path, nrows=0).columns.tolist()


def _iter_csv_with_pigz(
    path: Path,
    *,
    usecols: Optional[Sequence[str]],
    chunksize: int,
    pigz_workers: int,
    logger: logging.Logger,
) -> Iterable[pd.DataFrame]:
    pigz = shutil.which("pigz")

    if (
        pigz_workers <= 0
        or pigz is None
        or not str(path).endswith(".csv.gz")
    ):
        if str(path).endswith(".csv.gz") and pigz_workers > 0 and pigz is None:
            logger.info(
                "pigz not found; falling back to pandas/gzip for %s", path
            )

        yield from pd.read_csv(
            path,
            usecols=list(usecols) if usecols is not None else None,
            chunksize=chunksize,
            low_memory=False,
        )
        return

    logger.info(
        "Using pigz parallel decompression: workers=%d file=%s",
        pigz_workers,
        path,
    )

    proc = subprocess.Popen(
        [pigz, "-dc", "-p", str(pigz_workers), str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )

    if proc.stdout is None:
        raise RuntimeError(f"Could not open pigz stdout for {path}")

    try:
        reader = pd.read_csv(
            proc.stdout,
            usecols=list(usecols) if usecols is not None else None,
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            yield chunk

        proc.stdout.close()
        stderr = b""
        if proc.stderr is not None:
            stderr = proc.stderr.read()
        return_code = proc.wait()

        if return_code != 0:
            raise RuntimeError(
                f"pigz failed for {path} with exit code {return_code}: "
                f"{stderr.decode(errors='replace')[:1000]}"
            )
    except BaseException:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        raise
    finally:
        if proc.stdout is not None and not proc.stdout.closed:
            proc.stdout.close()
        if proc.stderr is not None:
            proc.stderr.close()


def iter_table(
    path: Path,
    *,
    usecols: Optional[Sequence[str]],
    chunksize: int,
    pigz_workers: int,
    logger: logging.Logger,
) -> Iterable[pd.DataFrame]:
    if path.suffix == ".parquet":
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        scanner = dataset.scanner(
            columns=list(usecols) if usecols is not None else None,
            batch_size=chunksize,
        )
        for batch in scanner.to_batches():
            yield batch.to_pandas()
        return

    yield from _iter_csv_with_pigz(
        path,
        usecols=usecols,
        chunksize=chunksize,
        pigz_workers=pigz_workers,
        logger=logger,
    )


def save_small_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")


def finite_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.where(np.isfinite(values))


def standardize_event_frame(
    *,
    database: str,
    patient_id: pd.Series,
    stay_id: pd.Series,
    source_table: str,
    source_variable: pd.Series,
    source_itemid: pd.Series,
    event_offset_minutes: pd.Series,
    value: pd.Series,
    unit: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "database": database,
            "patient_id": patient_id.astype("string"),
            "stay_id": stay_id.astype("string"),
            "source_table": source_table,
            "source_variable": source_variable.astype("string"),
            "source_itemid": source_itemid.astype("string"),
            "event_offset_minutes": pd.to_numeric(
                event_offset_minutes, errors="coerce"
            ),
            "value": finite_numeric(value),
            "unit": unit.astype("string"),
        }
    )

    out = out.dropna(
        subset=[
            "patient_id",
            "stay_id",
            "source_variable",
            "event_offset_minutes",
            "value",
        ]
    )

    out = out.loc[np.isfinite(out["event_offset_minutes"])].copy()
    out["event_offset_minutes"] = out["event_offset_minutes"].astype("float64")
    out["event_offset_hours"] = (
        out["event_offset_minutes"] / 60.0
    ).astype("float64")
    out["value"] = out["value"].astype("float64")

    for col in [
        "database",
        "patient_id",
        "stay_id",
        "source_table",
        "source_variable",
        "source_itemid",
        "unit",
    ]:
        out[col] = out[col].fillna("").astype(str)

    return out[EVENT_COLUMNS]


def load_mimic_cohort(cohort_dir: Path) -> pd.DataFrame:
    path = find_table(cohort_dir, "mimic_stroke_first_icu_stay")
    df = read_table(path)

    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "los",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"MIMIC cohort file {path} is missing columns: {missing}"
        )

    df["intime"] = pd.to_datetime(df["intime"], errors="coerce")
    df["outtime"] = pd.to_datetime(df["outtime"], errors="coerce")
    df["los"] = pd.to_numeric(df["los"], errors="coerce")
    df = df.dropna(
        subset=["subject_id", "hadm_id", "stay_id", "intime", "outtime"]
    )

    if df["subject_id"].duplicated().any():
        raise ValueError(
            "MIMIC Stage-01 cohort is not one row per subject_id."
        )
    return df.reset_index(drop=True)


def load_eicu_cohort(cohort_dir: Path) -> pd.DataFrame:
    path = find_table(cohort_dir, "eicu_stroke_first_icu_stay")
    df = read_table(path)

    required = {
        "uniquepid",
        "patientunitstayid",
        "unitdischargeoffset",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"eICU cohort file {path} is missing columns: {missing}"
        )

    df["patientunitstayid"] = pd.to_numeric(
        df["patientunitstayid"], errors="coerce"
    )
    df["unitdischargeoffset"] = pd.to_numeric(
        df["unitdischargeoffset"], errors="coerce"
    )
    df = df.dropna(
        subset=["uniquepid", "patientunitstayid", "unitdischargeoffset"]
    ).copy()
    df["patientunitstayid"] = df["patientunitstayid"].astype(np.int64)

    if df["uniquepid"].duplicated().any():
        raise ValueError(
            "eICU Stage-01 cohort is not one row per uniquepid."
        )
    return df.reset_index(drop=True)


def build_mimic_reference_tables(
    mimic_root: Path,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    d_items_path = find_table(mimic_root / "icu", "d_items")
    d_labitems_path = find_table(mimic_root / "hosp", "d_labitems")

    d_items = read_table(d_items_path)
    d_labitems = read_table(d_labitems_path)

    chart_label: Dict[int, str] = {}
    chart_unit: Dict[int, str] = {}
    lab_label: Dict[int, str] = {}

    if "itemid" in d_items.columns:
        for _, row in d_items.iterrows():
            try:
                itemid = int(row["itemid"])
            except Exception:
                continue
            label = row.get("label", "")
            unit = row.get("unitname", "")
            chart_label[itemid] = "" if pd.isna(label) else str(label)
            chart_unit[itemid] = "" if pd.isna(unit) else str(unit)

    if "itemid" in d_labitems.columns:
        for _, row in d_labitems.iterrows():
            try:
                itemid = int(row["itemid"])
            except Exception:
                continue
            label = row.get("label", "")
            lab_label[itemid] = "" if pd.isna(label) else str(label)

    return chart_label, chart_unit, lab_label


def mimic_horizon_table(
    cohort: pd.DataFrame,
    horizon_hours: float,
) -> pd.DataFrame:
    x = cohort[
        ["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"]
    ].copy()

    x["horizon_end"] = x["intime"] + pd.to_timedelta(
        horizon_hours, unit="h"
    )
    x["observation_end"] = x[["outtime", "horizon_end"]].min(axis=1)
    return x


def _worker_mimic_chartevents(chunk: pd.DataFrame) -> pd.DataFrame:
    state = _WORKER_STATE
    cohort_merge: pd.DataFrame = state["cohort_merge"]  # type: ignore[assignment]
    labels: Mapping[int, str] = state["labels"]  # type: ignore[assignment]
    ref_units: Mapping[int, str] = state["units"]  # type: ignore[assignment]
    horizon_minutes = float(state["horizon_minutes"])

    for col in ["subject_id", "hadm_id", "stay_id"]:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        ).astype("Int64")

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"], errors="coerce"
    )
    chunk["valuenum"] = finite_numeric(chunk["valuenum"])
    chunk = chunk.dropna(
        subset=[
            "subject_id",
            "hadm_id",
            "stay_id",
            "charttime",
            "valuenum",
            "itemid",
        ]
    )
    if chunk.empty:
        return empty_event_frame()

    merged = chunk.merge(
        cohort_merge,
        on=["subject_id", "hadm_id", "stay_id"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return empty_event_frame()

    merged = merged[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= merged["observation_end"])
    ].copy()
    if merged.empty:
        return empty_event_frame()

    offset_minutes = (
        merged["charttime"] - merged["intime"]
    ).dt.total_seconds() / 60.0

    itemid_num = pd.to_numeric(
        merged["itemid"], errors="coerce"
    ).astype("Int64")
    variable = itemid_num.map(labels).astype("string")
    fallback = "itemid_" + itemid_num.astype("string")
    variable = variable.fillna(fallback)

    source_unit = merged["valueuom"].astype("string")
    reference_unit = itemid_num.map(ref_units).astype("string")
    source_unit = source_unit.fillna(reference_unit).fillna("")

    events = standardize_event_frame(
        database="MIMIC-IV",
        patient_id=merged["subject_id"],
        stay_id=merged["stay_id"],
        source_table="chartevents",
        source_variable=variable,
        source_itemid=itemid_num.astype("string"),
        event_offset_minutes=offset_minutes,
        value=merged["valuenum"],
        unit=source_unit,
    )

    return events[
        (events["event_offset_minutes"] >= 0)
        & (events["event_offset_minutes"] <= horizon_minutes)
    ].reset_index(drop=True)


def _worker_mimic_labevents(chunk: pd.DataFrame) -> pd.DataFrame:
    state = _WORKER_STATE
    cohort_merge: pd.DataFrame = state["cohort_merge"]  # type: ignore[assignment]
    labels: Mapping[int, str] = state["labels"]  # type: ignore[assignment]
    horizon_minutes = float(state["horizon_minutes"])

    for col in ["subject_id", "hadm_id"]:
        chunk[col] = pd.to_numeric(
            chunk[col], errors="coerce"
        ).astype("Int64")

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"], errors="coerce"
    )
    chunk["valuenum"] = finite_numeric(chunk["valuenum"])
    chunk = chunk.dropna(
        subset=[
            "subject_id",
            "hadm_id",
            "charttime",
            "valuenum",
            "itemid",
        ]
    )
    if chunk.empty:
        return empty_event_frame()

    merged = chunk.merge(
        cohort_merge,
        on=["subject_id", "hadm_id"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return empty_event_frame()

    merged = merged[
        (merged["charttime"] >= merged["intime"])
        & (merged["charttime"] <= merged["observation_end"])
    ].copy()
    if merged.empty:
        return empty_event_frame()

    offset_minutes = (
        merged["charttime"] - merged["intime"]
    ).dt.total_seconds() / 60.0

    itemid_num = pd.to_numeric(
        merged["itemid"], errors="coerce"
    ).astype("Int64")
    variable = itemid_num.map(labels).astype("string")
    fallback = "itemid_" + itemid_num.astype("string")
    variable = variable.fillna(fallback)

    source_unit = merged["valueuom"].astype("string").fillna("")

    events = standardize_event_frame(
        database="MIMIC-IV",
        patient_id=merged["subject_id"],
        stay_id=merged["stay_id"],
        source_table="labevents",
        source_variable=variable,
        source_itemid=itemid_num.astype("string"),
        event_offset_minutes=offset_minutes,
        value=merged["valuenum"],
        unit=source_unit,
    )

    return events[
        (events["event_offset_minutes"] >= 0)
        & (events["event_offset_minutes"] <= horizon_minutes)
    ].reset_index(drop=True)


def _log_source_progress(
    *,
    logger: logging.Logger,
    source: str,
    start_time: float,
    chunks_read: int,
    chunks_submitted: int,
    completed_chunks: int,
    scanned_rows: int,
    cohort_rows: int,
    retained_rows: int,
    workers: int,
) -> None:
    elapsed = max(time.monotonic() - start_time, 1e-9)
    rate = scanned_rows / elapsed

    logger.info(
        "%s progress | chunks read=%d submitted=%d completed=%d | "
        "scanned=%d cohort=%d retained=%d | workers=%d | "
        "elapsed=%.1fs rate=%.0f rows/s",
        source,
        chunks_read,
        chunks_submitted,
        completed_chunks,
        scanned_rows,
        cohort_rows,
        retained_rows,
        workers,
        elapsed,
        rate,
    )


def extract_mimic_chartevents(
    *,
    mimic_root: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    horizon_hours: float,
    chunksize: int,
    workers: int,
    max_in_flight: int,
    pigz_workers: int,
    labels: Mapping[int, str],
    units: Mapping[int, str],
    logger: logging.Logger,
) -> dict:
    path = find_table(mimic_root / "icu", "chartevents")
    out_path = output_dir / "mimic" / "chartevents.parquet"
    writer = IncrementalParquetWriter(out_path)

    cohort_h = mimic_horizon_table(cohort, horizon_hours)
    stay_ids = set(
        pd.to_numeric(cohort_h["stay_id"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .tolist()
    )

    cohort_merge = cohort_h[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "observation_end",
        ]
    ].copy()
    for col in ["subject_id", "hadm_id", "stay_id"]:
        cohort_merge[col] = pd.to_numeric(
            cohort_merge[col], errors="coerce"
        ).astype("Int64")

    state = {
        "cohort_merge": cohort_merge,
        "labels": dict(labels),
        "units": dict(units),
        "horizon_minutes": horizon_hours * 60.0,
    }

    runner = OrderedParallelRunner(
        worker_fn=_worker_mimic_chartevents,
        state=state,
        writer=writer,
        workers=workers,
        max_in_flight=max_in_flight,
        logger=logger,
        source_name="MIMIC chartevents",
    )

    source_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ]

    scanned = 0
    cohort_rows = 0
    chunks_read = 0
    chunks_submitted = 0
    start = time.monotonic()

    logger.info("Scanning MIMIC chartevents: %s", path)

    try:
        for chunk in iter_table(
            path,
            usecols=source_cols,
            chunksize=chunksize,
            pigz_workers=pigz_workers,
            logger=logger,
        ):
            chunks_read += 1
            scanned += len(chunk)

            stay_numeric = pd.to_numeric(
                chunk["stay_id"], errors="coerce"
            )
            chunk = chunk[stay_numeric.isin(stay_ids)].copy()
            if not chunk.empty:
                cohort_rows += len(chunk)
                chunks_submitted += 1
                runner.submit(chunks_read, chunk)

            if chunks_read % 25 == 0:
                _log_source_progress(
                    logger=logger,
                    source="chartevents",
                    start_time=start,
                    chunks_read=chunks_read,
                    chunks_submitted=chunks_submitted,
                    completed_chunks=runner.completed_chunks,
                    scanned_rows=scanned,
                    cohort_rows=cohort_rows,
                    retained_rows=runner.retained_rows,
                    workers=workers,
                )

        runner.drain()
    finally:
        runner.close()
        writer.close()

    elapsed = time.monotonic() - start
    logger.info(
        "MIMIC chartevents complete | scanned=%d cohort=%d retained=%d "
        "elapsed=%.1fs rate=%.0f rows/s",
        scanned,
        cohort_rows,
        runner.retained_rows,
        elapsed,
        scanned / max(elapsed, 1e-9),
    )

    return {
        "path": str(out_path),
        "source_rows_scanned": int(scanned),
        "cohort_rows_before_worker_time_filter": int(cohort_rows),
        "events_retained": int(runner.retained_rows),
        "chunks_read": int(chunks_read),
        "chunks_submitted": int(chunks_submitted),
        "workers": int(workers),
        "pigz_workers": int(pigz_workers),
        "elapsed_seconds": float(elapsed),
        "source_rows_per_second": float(scanned / max(elapsed, 1e-9)),
    }


def extract_mimic_labevents(
    *,
    mimic_root: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    horizon_hours: float,
    chunksize: int,
    workers: int,
    max_in_flight: int,
    pigz_workers: int,
    labels: Mapping[int, str],
    logger: logging.Logger,
) -> dict:
    path = find_table(mimic_root / "hosp", "labevents")
    out_path = output_dir / "mimic" / "labevents.parquet"
    writer = IncrementalParquetWriter(out_path)

    cohort_h = mimic_horizon_table(cohort, horizon_hours)
    hadm_ids = set(
        pd.to_numeric(cohort_h["hadm_id"], errors="coerce")
        .dropna()
        .astype(np.int64)
        .tolist()
    )

    cohort_merge = cohort_h[
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "observation_end",
        ]
    ].copy()
    for col in ["subject_id", "hadm_id", "stay_id"]:
        cohort_merge[col] = pd.to_numeric(
            cohort_merge[col], errors="coerce"
        ).astype("Int64")

    state = {
        "cohort_merge": cohort_merge,
        "labels": dict(labels),
        "horizon_minutes": horizon_hours * 60.0,
    }

    runner = OrderedParallelRunner(
        worker_fn=_worker_mimic_labevents,
        state=state,
        writer=writer,
        workers=workers,
        max_in_flight=max_in_flight,
        logger=logger,
        source_name="MIMIC labevents",
    )

    source_cols = [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ]

    scanned = 0
    cohort_rows = 0
    chunks_read = 0
    chunks_submitted = 0
    start = time.monotonic()

    logger.info("Scanning MIMIC labevents: %s", path)

    try:
        for chunk in iter_table(
            path,
            usecols=source_cols,
            chunksize=chunksize,
            pigz_workers=pigz_workers,
            logger=logger,
        ):
            chunks_read += 1
            scanned += len(chunk)

            hadm_numeric = pd.to_numeric(
                chunk["hadm_id"], errors="coerce"
            )
            chunk = chunk[hadm_numeric.isin(hadm_ids)].copy()
            if not chunk.empty:
                cohort_rows += len(chunk)
                chunks_submitted += 1
                runner.submit(chunks_read, chunk)

            if chunks_read % 25 == 0:
                _log_source_progress(
                    logger=logger,
                    source="labevents",
                    start_time=start,
                    chunks_read=chunks_read,
                    chunks_submitted=chunks_submitted,
                    completed_chunks=runner.completed_chunks,
                    scanned_rows=scanned,
                    cohort_rows=cohort_rows,
                    retained_rows=runner.retained_rows,
                    workers=workers,
                )

        runner.drain()
    finally:
        runner.close()
        writer.close()

    elapsed = time.monotonic() - start
    logger.info(
        "MIMIC labevents complete | scanned=%d cohort=%d retained=%d "
        "elapsed=%.1fs rate=%.0f rows/s",
        scanned,
        cohort_rows,
        runner.retained_rows,
        elapsed,
        scanned / max(elapsed, 1e-9),
    )

    return {
        "path": str(out_path),
        "source_rows_scanned": int(scanned),
        "cohort_rows_before_worker_time_filter": int(cohort_rows),
        "events_retained": int(runner.retained_rows),
        "chunks_read": int(chunks_read),
        "chunks_submitted": int(chunks_submitted),
        "workers": int(workers),
        "pigz_workers": int(pigz_workers),
        "elapsed_seconds": float(elapsed),
        "source_rows_per_second": float(scanned / max(elapsed, 1e-9)),
    }


def build_mimic_demographics(
    *,
    mimic_root: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    patients_path = find_table(mimic_root / "hosp", "patients")
    admissions_path = find_table(mimic_root / "hosp", "admissions")

    patient_header = read_header(patients_path)
    admission_header = read_header(admissions_path)

    patient_cols = [
        c
        for c in ["subject_id", "gender", "anchor_age", "anchor_year"]
        if c in patient_header
    ]
    admission_cols = [
        c
        for c in [
            "subject_id",
            "hadm_id",
            "race",
            "language",
            "marital_status",
            "hospital_expire_flag",
            "admittime",
            "dischtime",
            "deathtime",
        ]
        if c in admission_header
    ]

    patients = read_table(patients_path, patient_cols)
    admissions = read_table(admissions_path, admission_cols)

    demo = cohort.copy()
    demo = demo.merge(
        patients,
        on="subject_id",
        how="left",
        validate="one_to_one",
    )
    demo = demo.merge(
        admissions,
        on=["subject_id", "hadm_id"],
        how="left",
        validate="one_to_one",
    )

    if {"anchor_age", "anchor_year"}.issubset(demo.columns):
        admission_year = demo["intime"].dt.year
        demo["age"] = (
            pd.to_numeric(demo["anchor_age"], errors="coerce")
            + (
                admission_year
                - pd.to_numeric(demo["anchor_year"], errors="coerce")
            )
        ).clip(upper=91)

    demo["icu_los_days"] = pd.to_numeric(demo["los"], errors="coerce")
    demo["patient_id"] = demo["subject_id"].astype(str)
    demo["stay_id_canonical"] = demo["stay_id"].astype(str)
    demo["database"] = "MIMIC-IV"

    preferred = [
        "database",
        "patient_id",
        "stay_id_canonical",
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_los_days",
        "gender",
        "age",
        "race",
        "language",
        "marital_status",
        "hospital_expire_flag",
        "admittime",
        "dischtime",
        "deathtime",
    ]
    demo = demo[[c for c in preferred if c in demo.columns]].copy()

    path = output_dir / "mimic" / "demographics_outcomes.parquet"
    save_small_parquet(demo, path)

    logger.info(
        "Saved MIMIC demographics/outcomes: %s rows=%d",
        path,
        len(demo),
    )
    return {"path": str(path), "patients": int(len(demo))}


def eicu_horizon_lookup(
    cohort: pd.DataFrame,
    horizon_hours: float,
) -> pd.DataFrame:
    x = cohort[
        ["uniquepid", "patientunitstayid", "unitdischargeoffset"]
    ].copy()

    x["observation_end_minutes"] = np.minimum(
        pd.to_numeric(x["unitdischargeoffset"], errors="coerce"),
        horizon_hours * 60.0,
    )
    x["patientunitstayid"] = pd.to_numeric(
        x["patientunitstayid"], errors="coerce"
    ).astype("Int64")
    return x


def _worker_eicu_long(chunk: pd.DataFrame) -> pd.DataFrame:
    state = _WORKER_STATE

    lookup: pd.DataFrame = state["lookup"]  # type: ignore[assignment]
    stay_col = str(state["stay_col"])
    offset_col = str(state["offset_col"])
    variable_col = str(state["variable_col"])
    value_col = str(state["value_col"])
    unit_col = state["unit_col"]
    itemid_col = state["itemid_col"]
    source_table = str(state["source_table"])
    horizon_minutes = float(state["horizon_minutes"])

    chunk[stay_col] = pd.to_numeric(
        chunk[stay_col], errors="coerce"
    ).astype("Int64")
    chunk[offset_col] = pd.to_numeric(
        chunk[offset_col], errors="coerce"
    )
    chunk = chunk.dropna(
        subset=[stay_col, offset_col, variable_col, value_col]
    )
    if chunk.empty:
        return empty_event_frame()

    merged = chunk.merge(
        lookup,
        left_on=stay_col,
        right_on="patientunitstayid",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_cohort"),
    )
    if merged.empty:
        return empty_event_frame()

    merged = merged[
        (merged[offset_col] >= 0)
        & (merged[offset_col] <= merged["observation_end_minutes"])
        & (merged[offset_col] <= horizon_minutes)
    ].copy()
    if merged.empty:
        return empty_event_frame()

    if unit_col is not None and str(unit_col) in merged.columns:
        unit = merged[str(unit_col)].astype("string")
    else:
        unit = pd.Series("", index=merged.index, dtype="string")

    if itemid_col is not None and str(itemid_col) in merged.columns:
        itemid = merged[str(itemid_col)].astype("string")
    else:
        itemid = pd.Series("", index=merged.index, dtype="string")

    return standardize_event_frame(
        database="eICU",
        patient_id=merged["uniquepid"],
        stay_id=merged["patientunitstayid"],
        source_table=source_table,
        source_variable=merged[variable_col],
        source_itemid=itemid,
        event_offset_minutes=merged[offset_col],
        value=merged[value_col],
        unit=unit,
    ).reset_index(drop=True)


def extract_eicu_long_source(
    *,
    source_path: Path,
    source_table: str,
    cohort: pd.DataFrame,
    output_path: Path,
    stay_col: str,
    offset_col: str,
    variable_col: str,
    value_col: str,
    unit_col: Optional[str],
    itemid_col: Optional[str],
    horizon_hours: float,
    chunksize: int,
    workers: int,
    max_in_flight: int,
    pigz_workers: int,
    logger: logging.Logger,
) -> dict:
    header = read_header(source_path)
    required = [stay_col, offset_col, variable_col, value_col]
    missing = [c for c in required if c not in header]
    if missing:
        raise ValueError(
            f"{source_table}: missing required columns {missing} "
            f"in {source_path}"
        )

    usecols = required.copy()
    if unit_col and unit_col in header:
        usecols.append(unit_col)
    if itemid_col and itemid_col in header:
        usecols.append(itemid_col)
    usecols = list(dict.fromkeys(usecols))

    lookup = eicu_horizon_lookup(cohort, horizon_hours)
    stays = set(
        lookup["patientunitstayid"]
        .dropna()
        .astype(np.int64)
        .tolist()
    )
    lookup = lookup[
        ["uniquepid", "patientunitstayid", "observation_end_minutes"]
    ].copy()

    writer = IncrementalParquetWriter(output_path)

    state = {
        "lookup": lookup,
        "stay_col": stay_col,
        "offset_col": offset_col,
        "variable_col": variable_col,
        "value_col": value_col,
        "unit_col": unit_col if unit_col in header else None,
        "itemid_col": itemid_col if itemid_col in header else None,
        "source_table": source_table,
        "horizon_minutes": horizon_hours * 60.0,
    }

    runner = OrderedParallelRunner(
        worker_fn=_worker_eicu_long,
        state=state,
        writer=writer,
        workers=workers,
        max_in_flight=max_in_flight,
        logger=logger,
        source_name=f"eICU {source_table}",
    )

    scanned = 0
    cohort_rows = 0
    chunks_read = 0
    chunks_submitted = 0
    start = time.monotonic()

    logger.info("Scanning eICU %s: %s", source_table, source_path)

    try:
        for chunk in iter_table(
            source_path,
            usecols=usecols,
            chunksize=chunksize,
            pigz_workers=pigz_workers,
            logger=logger,
        ):
            chunks_read += 1
            scanned += len(chunk)

            stay_numeric = pd.to_numeric(
                chunk[stay_col], errors="coerce"
            )
            chunk = chunk[stay_numeric.isin(stays)].copy()
            if not chunk.empty:
                cohort_rows += len(chunk)
                chunks_submitted += 1
                runner.submit(chunks_read, chunk)

            if chunks_read % 25 == 0:
                _log_source_progress(
                    logger=logger,
                    source=source_table,
                    start_time=start,
                    chunks_read=chunks_read,
                    chunks_submitted=chunks_submitted,
                    completed_chunks=runner.completed_chunks,
                    scanned_rows=scanned,
                    cohort_rows=cohort_rows,
                    retained_rows=runner.retained_rows,
                    workers=workers,
                )

        runner.drain()
    finally:
        runner.close()
        writer.close()

    elapsed = time.monotonic() - start
    logger.info(
        "eICU %s complete | scanned=%d cohort=%d retained=%d "
        "elapsed=%.1fs rate=%.0f rows/s",
        source_table,
        scanned,
        cohort_rows,
        runner.retained_rows,
        elapsed,
        scanned / max(elapsed, 1e-9),
    )

    return {
        "path": str(output_path),
        "source_rows_scanned": int(scanned),
        "cohort_rows_before_worker_time_filter": int(cohort_rows),
        "events_retained": int(runner.retained_rows),
        "chunks_read": int(chunks_read),
        "chunks_submitted": int(chunks_submitted),
        "workers": int(workers),
        "pigz_workers": int(pigz_workers),
        "elapsed_seconds": float(elapsed),
        "source_rows_per_second": float(scanned / max(elapsed, 1e-9)),
    }


def detect_nursechart_columns(path: Path) -> Dict[str, Optional[str]]:
    header = read_header(path)

    def first(
        options: Sequence[str],
        required: bool = True,
    ) -> Optional[str]:
        for name in options:
            if name in header:
                return name
        if required:
            raise ValueError(
                f"None of the expected nursecharting columns "
                f"{list(options)} were found in {path}"
            )
        return None

    return {
        "stay": first(["patientunitstayid"]),
        "offset": first(
            ["nursingchartoffset", "nursingchartentryoffset"]
        ),
        "variable": first(
            [
                "nursingchartcelltypevalname",
                "nursingchartcelltypevallabel",
            ]
        ),
        "value": first(["nursingchartvalue"]),
        "unit": first(
            ["nursingchartcelltypecat"],
            required=False,
        ),
    }


def _worker_eicu_wide_vitals(chunk: pd.DataFrame) -> pd.DataFrame:
    state = _WORKER_STATE

    lookup: pd.DataFrame = state["lookup"]  # type: ignore[assignment]
    source_table = str(state["source_table"])
    value_cols: List[str] = state["value_cols"]  # type: ignore[assignment]
    horizon_minutes = float(state["horizon_minutes"])

    stay_col = "patientunitstayid"
    offset_col = "observationoffset"

    chunk[stay_col] = pd.to_numeric(
        chunk[stay_col], errors="coerce"
    ).astype("Int64")
    chunk[offset_col] = pd.to_numeric(
        chunk[offset_col], errors="coerce"
    )
    chunk = chunk.dropna(subset=[stay_col, offset_col])
    if chunk.empty:
        return empty_event_frame()

    merged = chunk.merge(
        lookup,
        on=stay_col,
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        return empty_event_frame()

    merged = merged[
        (merged[offset_col] >= 0)
        & (merged[offset_col] <= merged["observation_end_minutes"])
        & (merged[offset_col] <= horizon_minutes)
    ].copy()
    if merged.empty:
        return empty_event_frame()

    melted = merged.melt(
        id_vars=["uniquepid", stay_col, offset_col],
        value_vars=value_cols,
        var_name="source_variable",
        value_name="value",
    )
    melted["value"] = finite_numeric(melted["value"])
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        return empty_event_frame()

    blanks = pd.Series("", index=melted.index, dtype="string")

    return standardize_event_frame(
        database="eICU",
        patient_id=melted["uniquepid"],
        stay_id=melted[stay_col],
        source_table=source_table,
        source_variable=melted["source_variable"],
        source_itemid=blanks,
        event_offset_minutes=melted[offset_col],
        value=melted["value"],
        unit=blanks,
    ).reset_index(drop=True)


def extract_eicu_wide_vitals(
    *,
    source_path: Path,
    source_table: str,
    cohort: pd.DataFrame,
    output_path: Path,
    horizon_hours: float,
    chunksize: int,
    workers: int,
    max_in_flight: int,
    pigz_workers: int,
    logger: logging.Logger,
) -> dict:
    header = read_header(source_path)
    stay_col = "patientunitstayid"
    offset_col = "observationoffset"

    for required in [stay_col, offset_col]:
        if required not in header:
            raise ValueError(
                f"{source_table}: required column '{required}' not found"
            )

    id_like = {
        stay_col,
        offset_col,
        "vitalperiodicid",
        "vitalaperiodicid",
    }
    value_cols = [c for c in header if c not in id_like]
    if not value_cols:
        raise ValueError(f"{source_table}: no measurement columns found")

    lookup = eicu_horizon_lookup(cohort, horizon_hours)
    stays = set(
        lookup["patientunitstayid"]
        .dropna()
        .astype(np.int64)
        .tolist()
    )
    lookup = lookup[
        ["uniquepid", "patientunitstayid", "observation_end_minutes"]
    ].copy()

    writer = IncrementalParquetWriter(output_path)
    state = {
        "lookup": lookup,
        "source_table": source_table,
        "value_cols": value_cols,
        "horizon_minutes": horizon_hours * 60.0,
    }

    runner = OrderedParallelRunner(
        worker_fn=_worker_eicu_wide_vitals,
        state=state,
        writer=writer,
        workers=workers,
        max_in_flight=max_in_flight,
        logger=logger,
        source_name=f"eICU {source_table}",
    )

    usecols = [stay_col, offset_col] + value_cols

    scanned = 0
    cohort_rows = 0
    chunks_read = 0
    chunks_submitted = 0
    start = time.monotonic()

    logger.info(
        "Scanning eICU %s: %s (%d measurement columns)",
        source_table,
        source_path,
        len(value_cols),
    )

    try:
        for chunk in iter_table(
            source_path,
            usecols=usecols,
            chunksize=chunksize,
            pigz_workers=pigz_workers,
            logger=logger,
        ):
            chunks_read += 1
            scanned += len(chunk)

            stay_numeric = pd.to_numeric(
                chunk[stay_col], errors="coerce"
            )
            chunk = chunk[stay_numeric.isin(stays)].copy()
            if not chunk.empty:
                cohort_rows += len(chunk)
                chunks_submitted += 1
                runner.submit(chunks_read, chunk)

            if chunks_read % 25 == 0:
                _log_source_progress(
                    logger=logger,
                    source=source_table,
                    start_time=start,
                    chunks_read=chunks_read,
                    chunks_submitted=chunks_submitted,
                    completed_chunks=runner.completed_chunks,
                    scanned_rows=scanned,
                    cohort_rows=cohort_rows,
                    retained_rows=runner.retained_rows,
                    workers=workers,
                )

        runner.drain()
    finally:
        runner.close()
        writer.close()

    elapsed = time.monotonic() - start
    logger.info(
        "eICU %s complete | scanned=%d cohort=%d retained=%d "
        "elapsed=%.1fs rate=%.0f rows/s",
        source_table,
        scanned,
        cohort_rows,
        runner.retained_rows,
        elapsed,
        scanned / max(elapsed, 1e-9),
    )

    return {
        "path": str(output_path),
        "source_rows_scanned": int(scanned),
        "cohort_rows_before_worker_time_filter": int(cohort_rows),
        "events_retained": int(runner.retained_rows),
        "measurement_columns": value_cols,
        "chunks_read": int(chunks_read),
        "chunks_submitted": int(chunks_submitted),
        "workers": int(workers),
        "pigz_workers": int(pigz_workers),
        "elapsed_seconds": float(elapsed),
        "source_rows_per_second": float(scanned / max(elapsed, 1e-9)),
    }


def build_eicu_demographics(
    *,
    cohort: pd.DataFrame,
    output_dir: Path,
    logger: logging.Logger,
) -> dict:
    demo = cohort.copy()
    demo["database"] = "eICU"
    demo["patient_id"] = demo["uniquepid"].astype(str)
    demo["stay_id_canonical"] = (
        pd.to_numeric(demo["patientunitstayid"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    demo["icu_los_days"] = (
        pd.to_numeric(demo["unitdischargeoffset"], errors="coerce")
        / 1440.0
    )

    preferred = [
        "database",
        "patient_id",
        "stay_id_canonical",
        "uniquepid",
        "patienthealthsystemstayid",
        "patientunitstayid",
        "gender",
        "age",
        "ethnicity",
        "unitdischargestatus",
        "hospitaldischargestatus",
        "unitdischargeoffset",
        "icu_los_days",
        "unitvisitnumber",
        "unitstaytype",
        "unittype",
        "apacheadmissiondx",
    ]
    demo = demo[[c for c in preferred if c in demo.columns]].copy()

    path = output_dir / "eicu" / "demographics_outcomes.parquet"
    save_small_parquet(demo, path)

    logger.info(
        "Saved eICU demographics/outcomes: %s rows=%d",
        path,
        len(demo),
    )
    return {"path": str(path), "patients": int(len(demo))}


def extract_eicu_sources(
    *,
    eicu_root: Path,
    cohort: pd.DataFrame,
    output_dir: Path,
    horizon_hours: float,
    chunksize: int,
    workers: int,
    max_in_flight: int,
    pigz_workers: int,
    logger: logging.Logger,
) -> dict:
    results: Dict[str, object] = {}

    lab_path = find_table(eicu_root, "lab")
    lab_header = read_header(lab_path)
    lab_unit_col = (
        "labmeasurenamesystem"
        if "labmeasurenamesystem" in lab_header
        else None
    )
    lab_itemid_col = (
        "labtypeid" if "labtypeid" in lab_header else None
    )
    results["lab"] = extract_eicu_long_source(
        source_path=lab_path,
        source_table="lab",
        cohort=cohort,
        output_path=output_dir / "eicu" / "lab.parquet",
        stay_col="patientunitstayid",
        offset_col="labresultoffset",
        variable_col="labname",
        value_col="labresult",
        unit_col=lab_unit_col,
        itemid_col=lab_itemid_col,
        horizon_hours=horizon_hours,
        chunksize=chunksize,
        workers=workers,
        max_in_flight=max_in_flight,
        pigz_workers=pigz_workers,
        logger=logger,
    )

    customlab_path = find_table(eicu_root, "customlab")
    custom_header = read_header(customlab_path)
    custom_id = (
        "labothertypeid"
        if "labothertypeid" in custom_header
        else None
    )
    results["customlab"] = extract_eicu_long_source(
        source_path=customlab_path,
        source_table="customlab",
        cohort=cohort,
        output_path=output_dir / "eicu" / "customlab.parquet",
        stay_col="patientunitstayid",
        offset_col="labotheroffset",
        variable_col="labothername",
        value_col="labotherresult",
        unit_col=None,
        itemid_col=custom_id,
        horizon_hours=horizon_hours,
        chunksize=chunksize,
        workers=workers,
        max_in_flight=max_in_flight,
        pigz_workers=pigz_workers,
        logger=logger,
    )

    nurse_path = find_table(eicu_root, "nursecharting")
    ncols = detect_nursechart_columns(nurse_path)
    results["nursecharting"] = extract_eicu_long_source(
        source_path=nurse_path,
        source_table="nursecharting",
        cohort=cohort,
        output_path=output_dir / "eicu" / "nursecharting.parquet",
        stay_col=str(ncols["stay"]),
        offset_col=str(ncols["offset"]),
        variable_col=str(ncols["variable"]),
        value_col=str(ncols["value"]),
        unit_col=ncols["unit"],
        itemid_col=None,
        horizon_hours=horizon_hours,
        chunksize=chunksize,
        workers=workers,
        max_in_flight=max_in_flight,
        pigz_workers=pigz_workers,
        logger=logger,
    )

    periodic_path = find_table(eicu_root, "vitalperiodic")
    results["vitalperiodic"] = extract_eicu_wide_vitals(
        source_path=periodic_path,
        source_table="vitalperiodic",
        cohort=cohort,
        output_path=output_dir / "eicu" / "vitalperiodic.parquet",
        horizon_hours=horizon_hours,
        chunksize=chunksize,
        workers=workers,
        max_in_flight=max_in_flight,
        pigz_workers=pigz_workers,
        logger=logger,
    )

    aperiodic_path = find_table(eicu_root, "vitalaperiodic")
    results["vitalaperiodic"] = extract_eicu_wide_vitals(
        source_path=aperiodic_path,
        source_table="vitalaperiodic",
        cohort=cohort,
        output_path=output_dir / "eicu" / "vitalaperiodic.parquet",
        horizon_hours=horizon_hours,
        chunksize=chunksize,
        workers=workers,
        max_in_flight=max_in_flight,
        pigz_workers=pigz_workers,
        logger=logger,
    )

    results["demographics"] = build_eicu_demographics(
        cohort=cohort,
        output_dir=output_dir,
        logger=logger,
    )
    return results


def resolve_workers(requested: int) -> int:
    cpus = os.cpu_count() or 1

    if requested <= 0:
        return max(1, cpus - 2)

    # Always leave at least one CPU for the reader/writer/OS where possible.
    practical_max = max(1, cpus - 1)
    return min(requested, practical_max)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parallel extraction of first-48h numeric ICU events for the "
            "Stage-01 stroke cohorts."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=MIMIC_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--eicu-root",
        type=Path,
        default=EICU_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/01_cohorts",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/02_raw_events",
    )
    parser.add_argument(
        "--database",
        choices=["all", "mimic", "eicu"],
        default="all",
    )
    parser.add_argument(
        "--horizon-hours",
        type=float,
        default=HORIZON_HOURS_DEFAULT,
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=CHUNKSIZE_DEFAULT,
        help=f"CSV rows per reader chunk (default: {CHUNKSIZE_DEFAULT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAULT,
        help=(
            f"Worker processes per source (default: {WORKERS_DEFAULT}). "
            "Use 1 for serial debugging or 0 for cpu_count()-2."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help=(
            "Maximum submitted-but-not-yet-written chunks. "
            "0 means 2*workers."
        ),
    )
    parser.add_argument(
        "--pigz-workers",
        type=int,
        default=PIGZ_WORKERS_DEFAULT,
        help=(
            f"pigz decompression threads for .csv.gz "
            f"(default: {PIGZ_WORKERS_DEFAULT}; 0 disables). "
            "If pigz is unavailable, pandas gzip is used."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    require_pyarrow()

    project_root = args.project_root.expanduser().resolve()
    mimic_root = args.mimic_root.expanduser().resolve()
    eicu_root = args.eicu_root.expanduser().resolve()

    cohort_dir = (
        args.cohort_dir.expanduser().resolve()
        if args.cohort_dir is not None
        else project_root / "data" / "01_cohorts"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "02_raw_events"
    )

    if args.horizon_hours <= 0:
        raise ValueError("--horizon-hours must be positive.")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive.")
    if args.pigz_workers < 0:
        raise ValueError("--pigz-workers cannot be negative.")

    workers = resolve_workers(args.workers)
    max_in_flight = (
        args.max_in_flight
        if args.max_in_flight > 0
        else max(1, workers * 2)
    )

    logger = configure_logging(output_dir)

    logger.info("Project root: %s", project_root)
    logger.info("Cohort directory: %s", cohort_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Observation horizon: %.1f hours", args.horizon_hours)
    logger.info("Chunk size: %d", args.chunksize)
    logger.info("Detected CPUs: %d", os.cpu_count() or 1)
    logger.info("Worker processes: %d", workers)
    logger.info("Max in-flight chunks: %d", max_in_flight)
    logger.info("Requested pigz workers: %d", args.pigz_workers)
    logger.info("pigz executable: %s", shutil.which("pigz") or "not found")

    manifest = {
        "script": "02_extract_raw_events.py",
        "implementation": "bounded parallel producer-worker-writer",
        "project_root": str(project_root),
        "cohort_dir": str(cohort_dir),
        "output_dir": str(output_dir),
        "mimic_root": str(mimic_root),
        "eicu_root": str(eicu_root),
        "horizon_hours": float(args.horizon_hours),
        "chunksize": int(args.chunksize),
        "workers": int(workers),
        "max_in_flight": int(max_in_flight),
        "pigz_workers_requested": int(args.pigz_workers),
        "pigz_available": bool(shutil.which("pigz")),
        "time_rule": (
            "retain numeric events with 0 <= event time <= "
            "min(ICU exit, ICU admission + observation horizon)"
        ),
        "parallelism_rule": (
            "source file read once by parent; cheap cohort-ID prefilter in "
            "parent; cohort-relevant chunks processed by worker pool; "
            "single ordered Parquet writer in parent"
        ),
        "harmonization_performed": False,
        "aggregation_performed": False,
        "missing_values_imputed": False,
        "results": {},
    }

    if args.database in {"all", "mimic"}:
        mimic_cohort = load_mimic_cohort(cohort_dir)
        logger.info("Loaded MIMIC cohort: n=%d", len(mimic_cohort))

        chart_labels, chart_units, lab_labels = (
            build_mimic_reference_tables(mimic_root)
        )

        mimic_results: Dict[str, object] = {}
        mimic_results["chartevents"] = extract_mimic_chartevents(
            mimic_root=mimic_root,
            cohort=mimic_cohort,
            output_dir=output_dir,
            horizon_hours=args.horizon_hours,
            chunksize=args.chunksize,
            workers=workers,
            max_in_flight=max_in_flight,
            pigz_workers=args.pigz_workers,
            labels=chart_labels,
            units=chart_units,
            logger=logger,
        )
        mimic_results["labevents"] = extract_mimic_labevents(
            mimic_root=mimic_root,
            cohort=mimic_cohort,
            output_dir=output_dir,
            horizon_hours=args.horizon_hours,
            chunksize=args.chunksize,
            workers=workers,
            max_in_flight=max_in_flight,
            pigz_workers=args.pigz_workers,
            labels=lab_labels,
            logger=logger,
        )
        mimic_results["demographics"] = build_mimic_demographics(
            mimic_root=mimic_root,
            cohort=mimic_cohort,
            output_dir=output_dir,
            logger=logger,
        )
        manifest["results"]["mimic"] = mimic_results

    if args.database in {"all", "eicu"}:
        eicu_cohort = load_eicu_cohort(cohort_dir)
        logger.info("Loaded eICU cohort: n=%d", len(eicu_cohort))

        manifest["results"]["eicu"] = extract_eicu_sources(
            eicu_root=eicu_root,
            cohort=eicu_cohort,
            output_dir=output_dir,
            horizon_hours=args.horizon_hours,
            chunksize=args.chunksize,
            workers=workers,
            max_in_flight=max_in_flight,
            pigz_workers=args.pigz_workers,
            logger=logger,
        )

    manifest_path = output_dir / "02_raw_event_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Stage 02 completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
