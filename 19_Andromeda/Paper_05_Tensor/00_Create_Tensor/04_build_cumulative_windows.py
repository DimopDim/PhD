#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_build_cumulative_windows.py

Stage 04 of the clean multi-resolution tensor pipeline.

Purpose
-------
Build cumulative clinical summaries from the Stage-03 harmonized event stream.

For the primary pipeline:
    concepts       = 76
    aggregations   = mean, median, min, max
    clinical cols  = 76 * 4 = 304

Native temporal resolutions:
    o1: 1 h -> endpoints 1, 2, ..., 48       (48 rows/patient)
    o2: 2 h -> endpoints 2, 4, ..., 48       (24 rows/patient)
    o3: 3 h -> endpoints 3, 6, ..., 48       (16 rows/patient)
    o4: 4 h -> endpoints 4, 8, ..., 48       (12 rows/patient)

A key implementation detail is that the 48 one-hour cumulative endpoint
matrices are computed ONCE per patient. The 2 h, 3 h, and 4 h native grids are
strict subsamples of that same cumulative matrix. This guarantees that the
final [0, 48 h] clinical summary is identical across all four resolutions.

Important scientific rules
--------------------------
1. Cumulative windows are anchored to ICU admission:
       [0, 1 h], [0, 2 h], ..., [0, 48 h]
   and analogously for coarser resolutions.

2. An event at exactly the endpoint is included:
       event_offset_minutes <= endpoint_minutes

3. No carry-forward after ICU discharge:
   if endpoint > ICU LOS, ALL clinical descriptors at that endpoint are NaN.

4. Missing concepts within an otherwise available endpoint remain NaN.

5. No imputation, scaling, one-hot encoding, splitting, or tensor upsampling
   is performed here.

6. Demographics are NOT appended to the 304 clinical columns at this stage.
   The frozen MIMIC-training demographic schema is intentionally deferred to
   Stage 05. The later expected base dimensionality is:
       304 + 1 age + 3 gender + 33 race = 341.

7. Aggregations are read from imports/pipeline_config.json. The primary config
   is mean/median/min/max. The implementation also supports optional future
   sensitivity experiments with "std" and/or "last", but they are not enabled
   in the primary configuration.

Inputs
------
<project-root>/data/03_harmonized/
    mimic/harmonized_events.parquet
    mimic/harmonized_demographics.parquet
    eicu/harmonized_events.parquet
    eicu/harmonized_demographics.parquet

<project-root>/imports/
    concept_schema_76.csv
    pipeline_config.json

Outputs
-------
<project-root>/data/04_cumulative_windows/
    mimic/
        o1_01h.parquet
        o2_02h.parquet
        o3_03h.parquet
        o4_04h.parquet
    eicu/
        o1_01h.parquet
        o2_02h.parquet
        o3_03h.parquet
        o4_04h.parquet
    reports/
        window_summary.csv
        final_48h_consistency.csv
        feature_order.csv
    feature_order.json
    04_cumulative_windows_manifest.json
    04_build_cumulative_windows.log

Parallelism
-----------
The harmonized event stream is loaded once per database and represented as
compact read-only NumPy arrays. On Linux, worker processes are created with
"fork", so these arrays are shared copy-on-write rather than serialized for
every patient batch.

Example
-------
cd /home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

python 04_build_cumulative_windows.py \
    --database all \
    --workers 10 \
    --patients-per-task 64
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor"
)
INPUT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "03_harmonized"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "04_cumulative_windows"
IMPORT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "imports"

WORKERS_DEFAULT = 10
PATIENTS_PER_TASK_DEFAULT = 64

RESOLUTIONS = {
    "o1": 1,
    "o2": 2,
    "o3": 3,
    "o4": 4,
}

METADATA_COLUMNS = [
    "database",
    "patient_id",
    "stay_id",
    "native_resolution_hours",
    "native_step_index",
    "endpoint_hour",
    "endpoint_minutes",
    "icu_los_hours",
    "window_available",
]

# ---------------------------------------------------------------------------
# Worker-global, read-only arrays.
# Initialized once per database before the process pool is forked.
# ---------------------------------------------------------------------------

_G_EVENT_PATIENT: Optional[np.ndarray] = None
_G_EVENT_CONCEPT: Optional[np.ndarray] = None
_G_EVENT_OFFSET_MIN: Optional[np.ndarray] = None
_G_EVENT_VALUE: Optional[np.ndarray] = None
_G_PATIENT_BOUNDS: Dict[str, Tuple[int, int]] = {}
_G_PATIENT_META: Dict[str, Tuple[str, float]] = {}
_G_CONCEPTS: List[str] = []
_G_AGGREGATIONS: List[str] = []
_G_FEATURE_COLUMNS: List[str] = []


def require_pyarrow() -> None:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Stage 04 requires pyarrow. Install with: pip install pyarrow"
        ) from exc


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("build_cumulative_windows")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        output_dir / "04_build_cumulative_windows.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value: object) -> float:
    try:
        x = float(value)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def load_schema(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "concept_order",
        "canonical_concept",
        "target_unit",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Concept schema missing columns: {sorted(missing)}"
        )

    df["concept_order"] = pd.to_numeric(
        df["concept_order"], errors="raise"
    ).astype(int)
    df["canonical_concept"] = (
        df["canonical_concept"].astype(str).str.strip()
    )

    df = df.sort_values(
        "concept_order",
        kind="stable",
    ).reset_index(drop=True)

    if len(df) != 76:
        raise ValueError(
            f"Primary schema requires exactly 76 concepts; found {len(df)}."
        )

    if df["canonical_concept"].duplicated().any():
        duplicated = df.loc[
            df["canonical_concept"].duplicated(keep=False),
            "canonical_concept",
        ].tolist()
        raise ValueError(
            f"Duplicate canonical concepts: {duplicated}"
        )

    if "Chloride (serum)" in set(df["canonical_concept"]):
        raise ValueError(
            "'Chloride (serum)' must not appear in the locked 76-concept schema."
        )

    return df


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))

    aggregations = config.get("aggregations")
    if not isinstance(aggregations, list) or not aggregations:
        raise ValueError(
            "pipeline_config.json must define a non-empty 'aggregations' list."
        )

    supported = {"mean", "median", "min", "max", "std", "last"}
    unknown = [x for x in aggregations if x not in supported]
    if unknown:
        raise ValueError(
            f"Unsupported aggregations in pipeline config: {unknown}. "
            f"Supported: {sorted(supported)}"
        )

    if len(set(aggregations)) != len(aggregations):
        raise ValueError("Duplicate aggregation names in pipeline config.")

    if int(config.get("clinical_concept_count", -1)) != 76:
        raise ValueError(
            "pipeline_config clinical_concept_count must be 76."
        )

    primary = ["mean", "median", "min", "max"]
    if aggregations != primary:
        print(
            "WARNING: non-primary aggregation configuration detected: "
            f"{aggregations}. Primary manuscript configuration is {primary}.",
            file=sys.stderr,
        )

    return config


def feature_name(concept: str, aggregation: str) -> str:
    """
    Preserve the human-readable canonical concept and use a stable separator.

    Example:
        Heart Rate (bpm)__mean
        Heart Rate (bpm)__median
    """
    return f"{concept}__{aggregation}"


def build_feature_order(
    concepts: Sequence[str],
    aggregations: Sequence[str],
) -> List[str]:
    return [
        feature_name(concept, aggregation)
        for concept in concepts
        for aggregation in aggregations
    ]


class IncrementalWideParquetWriter:
    """Single-process incremental writer for one native resolution."""

    def __init__(
        self,
        path: Path,
        feature_columns: Sequence[str],
    ) -> None:
        require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.path = path
        self.feature_columns = list(feature_columns)
        self.columns = METADATA_COLUMNS + self.feature_columns
        self.writer = None
        self.rows = 0

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        missing = [
            col for col in self.columns
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Attempted to write frame missing columns: {missing[:10]}"
            )

        df = df[self.columns].copy()

        # Compact model matrix. Metadata remains explicit.
        for col in self.feature_columns:
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            ).astype("float32")

        df["native_resolution_hours"] = pd.to_numeric(
            df["native_resolution_hours"], errors="raise"
        ).astype("int16")
        df["native_step_index"] = pd.to_numeric(
            df["native_step_index"], errors="raise"
        ).astype("int16")
        df["endpoint_hour"] = pd.to_numeric(
            df["endpoint_hour"], errors="raise"
        ).astype("int16")
        df["endpoint_minutes"] = pd.to_numeric(
            df["endpoint_minutes"], errors="raise"
        ).astype("int32")
        df["icu_los_hours"] = pd.to_numeric(
            df["icu_los_hours"], errors="coerce"
        ).astype("float32")
        df["window_available"] = df["window_available"].astype(bool)

        table = self.pa.Table.from_pandas(
            df,
            preserve_index=False,
        )

        if self.writer is None:
            self.writer = self.pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
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

        # Deterministic empty output if needed.
        frame = pd.DataFrame(
            {
                "database": pd.Series(dtype="object"),
                "patient_id": pd.Series(dtype="object"),
                "stay_id": pd.Series(dtype="object"),
                "native_resolution_hours": pd.Series(dtype="int16"),
                "native_step_index": pd.Series(dtype="int16"),
                "endpoint_hour": pd.Series(dtype="int16"),
                "endpoint_minutes": pd.Series(dtype="int32"),
                "icu_los_hours": pd.Series(dtype="float32"),
                "window_available": pd.Series(dtype="bool"),
                **{
                    col: pd.Series(dtype="float32")
                    for col in self.feature_columns
                },
            }
        )
        frame.to_parquet(
            self.path,
            index=False,
            compression="zstd",
        )


def prepare_database_arrays(
    events_path: Path,
    demographics_path: Path,
    concepts: Sequence[str],
    logger: logging.Logger,
) -> Tuple[List[str], int, int]:
    """
    Load one database into compact global arrays.

    Events are sorted by:
        patient_id -> concept index -> event_offset_minutes

    Returns
    -------
    ordered_patient_ids, n_events, n_patients
    """
    global _G_EVENT_PATIENT
    global _G_EVENT_CONCEPT
    global _G_EVENT_OFFSET_MIN
    global _G_EVENT_VALUE
    global _G_PATIENT_BOUNDS
    global _G_PATIENT_META

    logger.info("Loading demographics: %s", demographics_path)
    demo = pd.read_parquet(demographics_path)

    required_demo = {
        "patient_id",
        "stay_id",
        "icu_los_days",
    }
    missing_demo = required_demo - set(demo.columns)
    if missing_demo:
        raise ValueError(
            f"{demographics_path} missing columns: {sorted(missing_demo)}"
        )

    demo["patient_id"] = (
        demo["patient_id"].astype(str).str.strip()
    )
    demo["stay_id"] = (
        demo["stay_id"].astype(str).str.strip()
    )
    demo["icu_los_days"] = pd.to_numeric(
        demo["icu_los_days"], errors="coerce"
    )

    if demo["patient_id"].duplicated().any():
        duplicates = demo.loc[
            demo["patient_id"].duplicated(keep=False),
            "patient_id",
        ].head(10).tolist()
        raise ValueError(
            "Stage 04 expects one selected ICU stay per patient. "
            f"Duplicate patient_id values include: {duplicates}"
        )

    if demo["icu_los_days"].isna().any():
        n = int(demo["icu_los_days"].isna().sum())
        raise ValueError(
            f"{n} patients have missing ICU LOS; cannot enforce "
            "post-discharge NaN windows."
        )

    if (demo["icu_los_days"] < 0).any():
        raise ValueError("Negative ICU LOS detected.")

    # Stable order for deterministic output.
    demo = demo.sort_values(
        ["patient_id", "stay_id"],
        kind="stable",
    ).reset_index(drop=True)

    ordered_patient_ids = demo["patient_id"].tolist()

    _G_PATIENT_META = {
        row.patient_id: (
            row.stay_id,
            float(row.icu_los_days) * 24.0,
        )
        for row in demo[
            ["patient_id", "stay_id", "icu_los_days"]
        ].itertuples(index=False)
    }

    logger.info("Loading harmonized events: %s", events_path)
    event_columns = [
        "patient_id",
        "canonical_concept",
        "event_offset_minutes",
        "value",
    ]
    events = pd.read_parquet(
        events_path,
        columns=event_columns,
    )

    events["patient_id"] = (
        events["patient_id"].astype(str).str.strip()
    )
    events["canonical_concept"] = (
        events["canonical_concept"].astype(str).str.strip()
    )
    events["event_offset_minutes"] = pd.to_numeric(
        events["event_offset_minutes"], errors="coerce"
    )
    events["value"] = pd.to_numeric(
        events["value"], errors="coerce"
    )

    concept_to_idx = {
        concept: idx
        for idx, concept in enumerate(concepts)
    }

    unknown_concepts = sorted(
        set(events["canonical_concept"].dropna().unique())
        - set(concept_to_idx)
    )
    if unknown_concepts:
        raise ValueError(
            "Harmonized event file contains concepts outside the locked "
            f"schema: {unknown_concepts}"
        )

    events["concept_idx"] = (
        events["canonical_concept"]
        .map(concept_to_idx)
        .astype("Int16")
    )

    valid_patients = set(ordered_patient_ids)
    unknown_patient_mask = ~events["patient_id"].isin(valid_patients)
    if unknown_patient_mask.any():
        n = int(unknown_patient_mask.sum())
        raise ValueError(
            f"{n} harmonized events refer to patient IDs absent from "
            "harmonized_demographics.parquet."
        )

    finite_mask = (
        events["event_offset_minutes"].notna()
        & events["value"].notna()
        & events["concept_idx"].notna()
        & np.isfinite(events["event_offset_minutes"].to_numpy(dtype=float))
        & np.isfinite(events["value"].to_numpy(dtype=float))
    )
    events = events.loc[finite_mask].copy()

    # Stage 03 should already enforce this, but Stage 04 independently guards
    # the cumulative observation horizon.
    events = events[
        (events["event_offset_minutes"] >= 0.0)
        & (events["event_offset_minutes"] <= 2880.0)
    ].copy()

    events["concept_idx"] = events["concept_idx"].astype("int16")

    events = events.sort_values(
        [
            "patient_id",
            "concept_idx",
            "event_offset_minutes",
        ],
        kind="stable",
    ).reset_index(drop=True)

    # Compact arrays inherited read-only by forked workers.
    _G_EVENT_PATIENT = events["patient_id"].to_numpy(
        dtype=object,
        copy=True,
    )
    _G_EVENT_CONCEPT = events["concept_idx"].to_numpy(
        dtype=np.int16,
        copy=True,
    )
    _G_EVENT_OFFSET_MIN = events["event_offset_minutes"].to_numpy(
        dtype=np.float64,
        copy=True,
    )
    _G_EVENT_VALUE = events["value"].to_numpy(
        dtype=np.float64,
        copy=True,
    )

    # Patient -> contiguous [start, end) event slice.
    _G_PATIENT_BOUNDS = {}
    if len(events):
        ids = _G_EVENT_PATIENT
        change = np.empty(len(ids), dtype=bool)
        change[0] = True
        change[1:] = ids[1:] != ids[:-1]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(ids)]

        for start, end in zip(starts, ends):
            _G_PATIENT_BOUNDS[str(ids[start])] = (
                int(start),
                int(end),
            )

    n_events = len(events)
    n_patients = len(demo)

    logger.info(
        "Prepared database arrays: patients=%d events=%d "
        "patients_with_events=%d",
        n_patients,
        n_events,
        len(_G_PATIENT_BOUNDS),
    )

    # The workers only need the arrays/dicts above.
    del events
    del demo
    gc.collect()

    return ordered_patient_ids, n_events, n_patients


def _worker_init(
    concepts: List[str],
    aggregations: List[str],
    feature_columns: List[str],
) -> None:
    global _G_CONCEPTS
    global _G_AGGREGATIONS
    global _G_FEATURE_COLUMNS

    _G_CONCEPTS = concepts
    _G_AGGREGATIONS = aggregations
    _G_FEATURE_COLUMNS = feature_columns

    # Prevent nested BLAS/OpenMP thread pools inside each process.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[variable] = "1"


def _aggregate_prefixes(
    offsets_minutes: np.ndarray,
    values: np.ndarray,
    endpoints_minutes: np.ndarray,
    aggregations: Sequence[str],
) -> Dict[str, np.ndarray]:
    """
    Cumulative summaries for one patient/concept on a fixed endpoint grid.

    offsets_minutes must be sorted ascending.
    """
    n_endpoints = len(endpoints_minutes)
    result = {
        agg: np.full(
            n_endpoints,
            np.nan,
            dtype=np.float64,
        )
        for agg in aggregations
    }

    if len(values) == 0:
        return result

    # Number of source observations available by each inclusive endpoint.
    counts = np.searchsorted(
        offsets_minutes,
        endpoints_minutes,
        side="right",
    ).astype(np.int64)

    has_data = counts > 0
    if not has_data.any():
        return result

    if "mean" in aggregations:
        prefix_sum = np.cumsum(values, dtype=np.float64)
        idx = counts[has_data] - 1
        result["mean"][has_data] = (
            prefix_sum[idx] / counts[has_data]
        )

    if "min" in aggregations:
        prefix_min = np.minimum.accumulate(values)
        idx = counts[has_data] - 1
        result["min"][has_data] = prefix_min[idx]

    if "max" in aggregations:
        prefix_max = np.maximum.accumulate(values)
        idx = counts[has_data] - 1
        result["max"][has_data] = prefix_max[idx]

    if "last" in aggregations:
        idx = counts[has_data] - 1
        result["last"][has_data] = values[idx]

    # Median is not linearly composable. Cache by unique prefix length so
    # endpoints with no new observation do not trigger repeated work.
    if "median" in aggregations:
        cache: Dict[int, float] = {}
        out = result["median"]
        for j, count in enumerate(counts):
            k = int(count)
            if k <= 0:
                continue
            if k not in cache:
                cache[k] = float(np.median(values[:k]))
            out[j] = cache[k]

    # Optional sensitivity only. Sample standard deviation (ddof=1), undefined
    # with fewer than two observations. This is intentionally disabled in the
    # primary manuscript configuration.
    if "std" in aggregations:
        prefix_sum = np.cumsum(values, dtype=np.float64)
        prefix_sq = np.cumsum(
            values * values,
            dtype=np.float64,
        )
        out = result["std"]
        eligible = counts >= 2
        if eligible.any():
            k = counts[eligible].astype(np.float64)
            idx = counts[eligible] - 1
            sums = prefix_sum[idx]
            sums_sq = prefix_sq[idx]
            numerator = sums_sq - (sums * sums / k)
            # Numerical round-off may yield tiny negative values.
            numerator = np.maximum(numerator, 0.0)
            out[eligible] = np.sqrt(
                numerator / (k - 1.0)
            )

    return result


def _patient_full_48h_matrix(
    patient_id: str,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Return the 48 x (concepts*aggregations) cumulative clinical matrix.

    Rows beyond ICU discharge remain entirely NaN.
    """
    if patient_id not in _G_PATIENT_META:
        raise KeyError(
            f"Missing metadata for patient_id={patient_id}"
        )

    stay_id, icu_los_hours = _G_PATIENT_META[patient_id]
    n_features = len(_G_FEATURE_COLUMNS)

    matrix = np.full(
        (48, n_features),
        np.nan,
        dtype=np.float32,
    )

    endpoints_hours = np.arange(
        1,
        49,
        dtype=np.float64,
    )
    endpoints_minutes = endpoints_hours * 60.0

    # Inclusive availability at discharge time; no rows after discharge.
    valid_endpoint_mask = (
        endpoints_hours <= (icu_los_hours + 1e-9)
    )

    if not valid_endpoint_mask.any():
        return matrix, valid_endpoint_mask, stay_id, icu_los_hours

    bounds = _G_PATIENT_BOUNDS.get(patient_id)
    if bounds is None:
        # Patient belongs to cohort but has no harmonized event.
        return matrix, valid_endpoint_mask, stay_id, icu_los_hours

    start, end = bounds

    concept_idx = _G_EVENT_CONCEPT[start:end]
    offsets = _G_EVENT_OFFSET_MIN[start:end]
    values = _G_EVENT_VALUE[start:end]

    # Data are sorted by concept_idx then offset.
    if len(concept_idx):
        changes = np.empty(len(concept_idx), dtype=bool)
        changes[0] = True
        changes[1:] = concept_idx[1:] != concept_idx[:-1]
        group_starts = np.flatnonzero(changes)
        group_ends = np.r_[group_starts[1:], len(concept_idx)]

        n_aggs = len(_G_AGGREGATIONS)

        for gs, ge in zip(group_starts, group_ends):
            cidx = int(concept_idx[gs])

            c_offsets = offsets[gs:ge]
            c_values = values[gs:ge]

            finite = (
                np.isfinite(c_offsets)
                & np.isfinite(c_values)
                & (c_offsets >= 0.0)
                & (c_offsets <= 2880.0)
            )
            if not finite.all():
                c_offsets = c_offsets[finite]
                c_values = c_values[finite]

            if len(c_values) == 0:
                continue

            # Sorting should already hold, but assert defensively.
            if len(c_offsets) > 1 and np.any(
                c_offsets[1:] < c_offsets[:-1]
            ):
                order = np.argsort(
                    c_offsets,
                    kind="stable",
                )
                c_offsets = c_offsets[order]
                c_values = c_values[order]

            summaries = _aggregate_prefixes(
                c_offsets,
                c_values,
                endpoints_minutes,
                _G_AGGREGATIONS,
            )

            base = cidx * n_aggs
            for agg_offset, agg in enumerate(
                _G_AGGREGATIONS
            ):
                matrix[
                    valid_endpoint_mask,
                    base + agg_offset,
                ] = summaries[agg][
                    valid_endpoint_mask
                ].astype(np.float32)

    # Critical no-carry-forward rule.
    matrix[~valid_endpoint_mask, :] = np.nan

    return (
        matrix,
        valid_endpoint_mask,
        stay_id,
        icu_los_hours,
    )


def _build_patient_batch(
    patient_ids: Sequence[str],
    database_label: str,
) -> Dict[str, pd.DataFrame]:
    """
    Worker task: compute all four native resolution outputs for a patient batch.

    The 48-hour cumulative matrix is calculated once per patient; coarser
    resolutions are strict row selections from that same matrix.
    """
    resolution_records: Dict[str, List[pd.DataFrame]] = {
        key: []
        for key in RESOLUTIONS
    }

    for patient_id in patient_ids:
        matrix48, valid48, stay_id, los_hours = (
            _patient_full_48h_matrix(patient_id)
        )

        for key, hours in RESOLUTIONS.items():
            endpoints = np.arange(
                hours,
                49,
                hours,
                dtype=np.int16,
            )
            row_index = endpoints.astype(int) - 1

            selected_matrix = matrix48[row_index, :]
            selected_valid = valid48[row_index]

            clinical = pd.DataFrame(
                selected_matrix,
                columns=_G_FEATURE_COLUMNS,
            )

            metadata = pd.DataFrame(
                {
                    "database": database_label,
                    "patient_id": patient_id,
                    "stay_id": stay_id,
                    "native_resolution_hours": int(hours),
                    "native_step_index": np.arange(
                        1,
                        len(endpoints) + 1,
                        dtype=np.int16,
                    ),
                    "endpoint_hour": endpoints,
                    "endpoint_minutes": (
                        endpoints.astype(np.int32) * 60
                    ),
                    "icu_los_hours": float(los_hours),
                    "window_available": selected_valid,
                }
            )

            frame = pd.concat(
                [
                    metadata.reset_index(drop=True),
                    clinical.reset_index(drop=True),
                ],
                axis=1,
            )
            resolution_records[key].append(frame)

    result = {}
    for key, frames in resolution_records.items():
        result[key] = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(
                columns=METADATA_COLUMNS + _G_FEATURE_COLUMNS
            )
        )

    return result


class OrderedPatientBatchRunner:
    """
    Bounded ordered process-pool runner.

    Output batches are consumed in submission order, making Parquet row order
    deterministic even though workers execute concurrently.
    """

    def __init__(
        self,
        *,
        workers: int,
        max_in_flight: int,
        writers: Mapping[str, IncrementalWideParquetWriter],
        concepts: List[str],
        aggregations: List[str],
        feature_columns: List[str],
        logger: logging.Logger,
    ) -> None:
        self.workers = workers
        self.max_in_flight = max_in_flight
        self.writers = dict(writers)
        self.logger = logger
        self.pending: Deque[
            Tuple[int, Future]
        ] = deque()

        self.completed_tasks = 0
        self.post_discharge_violations = 0
        self.available_rows = defaultdict(int)
        self.post_discharge_rows = defaultdict(int)
        self.rows_with_any_clinical = defaultdict(int)

        if workers <= 1:
            self.executor = None
            _worker_init(
                concepts,
                aggregations,
                feature_columns,
            )
        else:
            try:
                ctx = mp.get_context("fork")
            except ValueError as exc:
                raise RuntimeError(
                    "Parallel Stage 04 requires the Linux 'fork' "
                    "multiprocessing start method. Use --workers 1 "
                    "for a serial fallback."
                ) from exc

            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(
                    concepts,
                    aggregations,
                    feature_columns,
                ),
            )

    def submit(
        self,
        task_id: int,
        patient_ids: Sequence[str],
        database_label: str,
    ) -> None:
        if self.executor is None:
            result = _build_patient_batch(
                patient_ids,
                database_label,
            )
            self._consume_result(result)
            self.completed_tasks += 1
            return

        future = self.executor.submit(
            _build_patient_batch,
            list(patient_ids),
            database_label,
        )
        self.pending.append(
            (task_id, future)
        )

        if len(self.pending) >= self.max_in_flight:
            self._consume_oldest()

    def _consume_result(
        self,
        result: Dict[str, pd.DataFrame],
    ) -> None:
        for key, frame in result.items():
            if frame.empty:
                continue

            clinical = frame[_G_FEATURE_COLUMNS]
            unavailable = ~frame["window_available"].to_numpy(
                dtype=bool
            )

            if unavailable.any():
                values = clinical.loc[
                    unavailable
                ].to_numpy(dtype=np.float32)
                violations = int(
                    np.isfinite(values).sum()
                )
                self.post_discharge_violations += violations

            self.available_rows[key] += int(
                frame["window_available"].sum()
            )
            self.post_discharge_rows[key] += int(
                (~frame["window_available"]).sum()
            )

            any_clinical = clinical.notna().any(axis=1)
            self.rows_with_any_clinical[key] += int(
                any_clinical.sum()
            )

            self.writers[key].write(frame)

    def _consume_oldest(self) -> None:
        task_id, future = self.pending.popleft()
        try:
            result = future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Stage-04 worker failed on patient batch "
                f"task_id={task_id}"
            ) from exc

        self._consume_result(result)
        self.completed_tasks += 1

    def drain(self) -> None:
        while self.pending:
            self._consume_oldest()

    def close(self) -> None:
        try:
            self.drain()
        finally:
            if self.executor is not None:
                self.executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )


def patient_batches(
    patient_ids: Sequence[str],
    batch_size: int,
) -> Iterable[List[str]]:
    for start in range(
        0,
        len(patient_ids),
        batch_size,
    ):
        yield list(
            patient_ids[
                start:start + batch_size
            ]
        )


def compare_final_48h(
    *,
    database_key: str,
    database_label: str,
    paths: Mapping[str, Path],
    feature_columns: Sequence[str],
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> dict:
    """
    Independent post-write verification of the final [0,48h] summaries.
    """
    finals: Dict[str, pd.DataFrame] = {}

    columns = [
        "patient_id",
        "stay_id",
        "endpoint_hour",
        "window_available",
    ] + list(feature_columns)

    for key, path in paths.items():
        df = pd.read_parquet(
            path,
            columns=columns,
        )
        df = df[
            df["endpoint_hour"].eq(48)
        ].copy()

        if df["patient_id"].duplicated().any():
            raise ValueError(
                f"{database_key}/{key}: duplicate final-48h rows per patient."
            )

        df = df.sort_values(
            "patient_id",
            kind="stable",
        ).reset_index(drop=True)
        finals[key] = df

    reference = finals["o1"]
    reference_ids = reference["patient_id"].tolist()
    ref_values = reference[
        list(feature_columns)
    ].to_numpy(dtype=np.float32)

    total_compared = 0
    total_mask_mismatch = 0
    max_abs_error = 0.0
    all_close = True

    for key in ["o2", "o3", "o4"]:
        other = finals[key]

        if other["patient_id"].tolist() != reference_ids:
            raise ValueError(
                f"{database_key}: patient ordering/set differs "
                f"between o1 and {key} at 48h."
            )

        if not np.array_equal(
            reference["window_available"].to_numpy(dtype=bool),
            other["window_available"].to_numpy(dtype=bool),
        ):
            raise ValueError(
                f"{database_key}: window_available differs between "
                f"o1 and {key} at 48h."
            )

        values = other[
            list(feature_columns)
        ].to_numpy(dtype=np.float32)

        ref_nan = np.isnan(ref_values)
        other_nan = np.isnan(values)

        mask_mismatch = int(
            np.count_nonzero(ref_nan != other_nan)
        )
        total_mask_mismatch += mask_mismatch

        both = ~(ref_nan | other_nan)
        if both.any():
            diff = np.abs(
                ref_values[both] - values[both]
            )
            local_max = float(
                np.max(diff)
            ) if diff.size else 0.0
            max_abs_error = max(
                max_abs_error,
                local_max,
            )

            close = np.allclose(
                ref_values[both],
                values[both],
                atol=atol,
                rtol=rtol,
            )
        else:
            close = True

        total_compared += int(both.sum())
        all_close = (
            all_close
            and close
            and mask_mismatch == 0
        )

    return {
        "database": database_key,
        "database_label": database_label,
        "patients": int(len(reference)),
        "comparisons": "o1_vs_o2,o1_vs_o3,o1_vs_o4",
        "finite_cells_compared": int(total_compared),
        "nan_mask_mismatches": int(total_mask_mismatch),
        "max_abs_error": float(max_abs_error),
        "atol": float(atol),
        "rtol": float(rtol),
        "passed": bool(all_close),
    }


def validate_resolution_output(
    *,
    path: Path,
    expected_patients: int,
    hours: int,
    feature_columns: Sequence[str],
) -> dict:
    columns = METADATA_COLUMNS + list(feature_columns)
    df = pd.read_parquet(
        path,
        columns=columns,
    )

    expected_steps = 48 // hours
    expected_rows = expected_patients * expected_steps

    if len(df) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} rows "
            f"({expected_patients} patients * {expected_steps} steps), "
            f"found {len(df)}."
        )

    endpoints_expected = np.arange(
        hours,
        49,
        hours,
        dtype=int,
    )
    endpoints_observed = np.sort(
        df["endpoint_hour"].unique()
    ).astype(int)

    if not np.array_equal(
        endpoints_observed,
        endpoints_expected,
    ):
        raise ValueError(
            f"{path}: endpoint grid mismatch. "
            f"Expected {endpoints_expected.tolist()}, "
            f"observed {endpoints_observed.tolist()}."
        )

    counts_per_patient = df.groupby(
        "patient_id",
        sort=False,
    ).size()

    if len(counts_per_patient) != expected_patients:
        raise ValueError(
            f"{path}: expected {expected_patients} unique patients, "
            f"found {len(counts_per_patient)}."
        )

    if not (
        counts_per_patient.to_numpy()
        == expected_steps
    ).all():
        raise ValueError(
            f"{path}: some patients do not have exactly "
            f"{expected_steps} native rows."
        )

    clinical = df[list(feature_columns)]
    unavailable = ~df["window_available"].to_numpy(dtype=bool)

    post_discharge_finite = 0
    if unavailable.any():
        post_values = clinical.loc[
            unavailable
        ].to_numpy(dtype=np.float32)
        post_discharge_finite = int(
            np.isfinite(post_values).sum()
        )

    if post_discharge_finite != 0:
        raise ValueError(
            f"{path}: found {post_discharge_finite} finite clinical "
            "values after ICU discharge."
        )

    available = df["window_available"].to_numpy(dtype=bool)
    any_clinical = clinical.notna().any(axis=1).to_numpy(dtype=bool)

    return {
        "rows": int(len(df)),
        "patients": int(len(counts_per_patient)),
        "native_steps_per_patient": int(expected_steps),
        "available_rows": int(available.sum()),
        "post_discharge_rows": int((~available).sum()),
        "rows_with_any_clinical_value": int(any_clinical.sum()),
        "available_rows_without_any_clinical_value": int(
            np.count_nonzero(
                available & (~any_clinical)
            )
        ),
        "post_discharge_finite_cells": int(post_discharge_finite),
    }


def write_feature_order_reports(
    *,
    output_dir: Path,
    schema: pd.DataFrame,
    aggregations: Sequence[str],
    feature_columns: Sequence[str],
) -> None:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    feature_index = 0
    for concept_order, concept in zip(
        schema["concept_order"],
        schema["canonical_concept"],
    ):
        for aggregation in aggregations:
            rows.append(
                {
                    "feature_index_0based": feature_index,
                    "feature_index_1based": feature_index + 1,
                    "concept_order": int(concept_order),
                    "canonical_concept": concept,
                    "aggregation": aggregation,
                    "feature_column": feature_columns[
                        feature_index
                    ],
                }
            )
            feature_index += 1

    pd.DataFrame(rows).to_csv(
        reports_dir / "feature_order.csv",
        index=False,
    )

    payload = {
        "clinical_concepts": schema[
            "canonical_concept"
        ].tolist(),
        "aggregations": list(aggregations),
        "clinical_feature_count": len(feature_columns),
        "feature_columns": list(feature_columns),
        "naming_rule": "<canonical_concept>__<aggregation>",
    }

    (output_dir / "feature_order.json").write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def resolve_workers(requested: int) -> int:
    cpus = os.cpu_count() or 1

    if requested <= 0:
        return max(
            1,
            min(10, cpus - 2),
        )

    practical_max = max(
        1,
        cpus - 1,
    )
    return min(
        requested,
        practical_max,
    )


def run_database(
    *,
    database_key: str,
    input_dir: Path,
    output_dir: Path,
    concepts: List[str],
    aggregations: List[str],
    feature_columns: List[str],
    workers: int,
    patients_per_task: int,
    max_in_flight: int,
    logger: logging.Logger,
) -> Tuple[dict, List[dict], dict]:
    # The parent process also consumes worker results and performs immediate
    # post-discharge checks, so it needs the same feature-order globals as the
    # workers.
    global _G_CONCEPTS
    global _G_AGGREGATIONS
    global _G_FEATURE_COLUMNS
    _G_CONCEPTS = list(concepts)
    _G_AGGREGATIONS = list(aggregations)
    _G_FEATURE_COLUMNS = list(feature_columns)

    database_label = (
        "MIMIC-IV"
        if database_key == "mimic"
        else "eICU"
    )

    events_path = (
        input_dir
        / database_key
        / "harmonized_events.parquet"
    )
    demographics_path = (
        input_dir
        / database_key
        / "harmonized_demographics.parquet"
    )

    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    if not demographics_path.is_file():
        raise FileNotFoundError(demographics_path)

    patient_ids, n_events, n_patients = (
        prepare_database_arrays(
            events_path,
            demographics_path,
            concepts,
            logger,
        )
    )

    output_paths = {
        "o1": (
            output_dir
            / database_key
            / "o1_01h.parquet"
        ),
        "o2": (
            output_dir
            / database_key
            / "o2_02h.parquet"
        ),
        "o3": (
            output_dir
            / database_key
            / "o3_03h.parquet"
        ),
        "o4": (
            output_dir
            / database_key
            / "o4_04h.parquet"
        ),
    }

    writers = {
        key: IncrementalWideParquetWriter(
            output_paths[key],
            feature_columns,
        )
        for key in RESOLUTIONS
    }

    runner = OrderedPatientBatchRunner(
        workers=workers,
        max_in_flight=max_in_flight,
        writers=writers,
        concepts=concepts,
        aggregations=aggregations,
        feature_columns=feature_columns,
        logger=logger,
    )

    start = time.monotonic()
    submitted = 0

    logger.info(
        "%s Stage-04 build started | patients=%d events=%d "
        "workers=%d patients_per_task=%d max_in_flight=%d",
        database_key,
        n_patients,
        n_events,
        workers,
        patients_per_task,
        max_in_flight,
    )

    try:
        for task_id, batch in enumerate(
            patient_batches(
                patient_ids,
                patients_per_task,
            ),
            start=1,
        ):
            submitted += 1
            runner.submit(
                task_id,
                batch,
                database_label,
            )

            if submitted % 10 == 0:
                elapsed = time.monotonic() - start
                logger.info(
                    "%s progress | submitted_tasks=%d "
                    "completed_tasks=%d/%d | elapsed=%.1fs",
                    database_key,
                    submitted,
                    runner.completed_tasks,
                    math.ceil(
                        n_patients / patients_per_task
                    ),
                    elapsed,
                )

        runner.drain()
    finally:
        runner.close()
        for writer in writers.values():
            writer.close()

    if runner.post_discharge_violations != 0:
        raise RuntimeError(
            f"{database_key}: detected "
            f"{runner.post_discharge_violations} finite values "
            "in unavailable post-discharge windows before final validation."
        )

    elapsed = time.monotonic() - start

    # Strong independent post-write validation.
    resolution_summaries = []
    for key, hours in RESOLUTIONS.items():
        validation = validate_resolution_output(
            path=output_paths[key],
            expected_patients=n_patients,
            hours=hours,
            feature_columns=feature_columns,
        )

        resolution_summaries.append(
            {
                "database": database_key,
                "resolution": key,
                "resolution_hours": hours,
                "path": str(output_paths[key]),
                **validation,
            }
        )

    final_consistency = compare_final_48h(
        database_key=database_key,
        database_label=database_label,
        paths=output_paths,
        feature_columns=feature_columns,
    )

    if not final_consistency["passed"]:
        raise RuntimeError(
            f"{database_key}: final 48h consistency assertion failed: "
            f"{final_consistency}"
        )

    result = {
        "database": database_key,
        "database_label": database_label,
        "patients": int(n_patients),
        "harmonized_events_input": int(n_events),
        "workers": int(workers),
        "patients_per_task": int(patients_per_task),
        "max_in_flight": int(max_in_flight),
        "tasks": int(submitted),
        "elapsed_seconds": float(elapsed),
        "output_paths": {
            key: str(path)
            for key, path in output_paths.items()
        },
        "resolution_rows": {
            key: int(writers[key].rows)
            for key in RESOLUTIONS
        },
        "post_discharge_finite_cells": 0,
        "final_48h_consistency_passed": True,
    }

    logger.info(
        "%s Stage-04 complete | elapsed=%.1fs | "
        "rows o1=%d o2=%d o3=%d o4=%d | final48=PASS",
        database_key,
        elapsed,
        writers["o1"].rows,
        writers["o2"].rows,
        writers["o3"].rows,
        writers["o4"].rows,
    )

    return result, resolution_summaries, final_consistency


def run_self_test() -> None:
    """
    Small deterministic unit test for the cumulative prefix logic.
    """
    offsets = np.array(
        [10.0, 60.0, 61.0, 180.0],
        dtype=float,
    )
    values = np.array(
        [1.0, 3.0, 5.0, 9.0],
        dtype=float,
    )
    endpoints = np.array(
        [60.0, 120.0, 180.0],
        dtype=float,
    )

    result = _aggregate_prefixes(
        offsets,
        values,
        endpoints,
        ["mean", "median", "min", "max", "last", "std"],
    )

    # Inclusive endpoint semantics: at 60 min, values 1 and 3 are included.
    np.testing.assert_allclose(
        result["mean"],
        np.array([2.0, 3.0, 4.5]),
    )
    np.testing.assert_allclose(
        result["median"],
        np.array([2.0, 3.0, 4.0]),
    )
    np.testing.assert_allclose(
        result["min"],
        np.array([1.0, 1.0, 1.0]),
    )
    np.testing.assert_allclose(
        result["max"],
        np.array([3.0, 5.0, 9.0]),
    )
    np.testing.assert_allclose(
        result["last"],
        np.array([3.0, 5.0, 9.0]),
    )
    np.testing.assert_allclose(
        result["std"][0],
        np.std([1.0, 3.0], ddof=1),
    )

    print("Stage 04 self-test: PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build cumulative 1h/2h/3h/4h clinical windows "
            "from Stage-03 harmonized events."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/03_harmonized",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/04_cumulative_windows",
    )
    parser.add_argument(
        "--import-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/imports",
    )
    parser.add_argument(
        "--database",
        choices=["all", "mimic", "eicu"],
        default="all",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAULT,
        help=(
            f"Patient-batch worker processes "
            f"(default: {WORKERS_DEFAULT}; 1=serial; 0=auto)."
        ),
    )
    parser.add_argument(
        "--patients-per-task",
        type=int,
        default=PATIENTS_PER_TASK_DEFAULT,
        help=(
            "Number of patients processed per worker task "
            f"(default: {PATIENTS_PER_TASK_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=0,
        help="Bounded outstanding tasks. 0 means 2*workers.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the cumulative-aggregation unit test and exit.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    require_pyarrow()

    if args.self_test:
        run_self_test()
        return 0

    if args.patients_per_task <= 0:
        raise ValueError(
            "--patients-per-task must be positive."
        )

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    input_dir = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else project_root / "data" / "03_harmonized"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "04_cumulative_windows"
    )
    import_dir = (
        args.import_dir.expanduser().resolve()
        if args.import_dir is not None
        else project_root / "imports"
    )

    logger = configure_logging(output_dir)

    schema_path = (
        import_dir
        / "concept_schema_76.csv"
    )
    config_path = (
        import_dir
        / "pipeline_config.json"
    )

    schema = load_schema(schema_path)
    config = load_config(config_path)

    concepts = schema[
        "canonical_concept"
    ].tolist()
    aggregations = list(
        config["aggregations"]
    )

    feature_columns = build_feature_order(
        concepts,
        aggregations,
    )

    expected_clinical = (
        len(concepts)
        * len(aggregations)
    )

    if len(feature_columns) != expected_clinical:
        raise AssertionError(
            "Feature-order construction mismatch."
        )

    if aggregations == [
        "mean",
        "median",
        "min",
        "max",
    ]:
        if expected_clinical != 304:
            raise ValueError(
                f"Locked primary configuration must produce 304 "
                f"clinical descriptors, found {expected_clinical}."
            )

        expected_base = int(
            config.get(
                "expected_base_features_per_temporal_step",
                -1,
            )
        )
        if expected_base != 341:
            raise ValueError(
                "Primary pipeline_config must state "
                "expected_base_features_per_temporal_step=341."
            )

    workers = resolve_workers(
        args.workers
    )
    max_in_flight = (
        args.max_in_flight
        if args.max_in_flight > 0
        else max(1, workers * 2)
    )

    logger.info("Project root: %s", project_root)
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Concept schema: %s", schema_path)
    logger.info("Pipeline config: %s", config_path)
    logger.info("Clinical concepts: %d", len(concepts))
    logger.info("Aggregations: %s", aggregations)
    logger.info(
        "Clinical feature columns per endpoint: %d",
        len(feature_columns),
    )
    logger.info("Detected CPUs: %d", os.cpu_count() or 1)
    logger.info("Worker processes: %d", workers)
    logger.info(
        "Patients per task: %d",
        args.patients_per_task,
    )
    logger.info(
        "Max in-flight tasks: %d",
        max_in_flight,
    )

    write_feature_order_reports(
        output_dir=output_dir,
        schema=schema,
        aggregations=aggregations,
        feature_columns=feature_columns,
    )

    databases = (
        ["mimic", "eicu"]
        if args.database == "all"
        else [args.database]
    )

    manifest = {
        "script": "04_build_cumulative_windows.py",
        "project_root": str(project_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "concept_schema": str(schema_path),
        "pipeline_config": str(config_path),
        "clinical_concept_count": int(len(concepts)),
        "aggregations": aggregations,
        "clinical_feature_count_per_endpoint": int(
            len(feature_columns)
        ),
        "demographics_appended": False,
        "imputation_performed": False,
        "scaling_performed": False,
        "split_performed": False,
        "tensor_upsampling_performed": False,
        "window_mode": "cumulative",
        "window_definition": (
            "inclusive cumulative interval [0, endpoint], "
            "anchored to ICU admission"
        ),
        "post_discharge_rule": (
            "if endpoint > ICU LOS, all clinical descriptors "
            "at that endpoint are NaN; no carry-forward"
        ),
        "resolution_strategy": (
            "compute the 48 one-hour cumulative endpoint matrix once "
            "per patient; obtain 2h/3h/4h native grids by exact "
            "subsampling of rows"
        ),
        "resolutions": {
            key: {
                "hours": hours,
                "native_steps": int(48 // hours),
                "endpoints_hours": list(
                    range(hours, 49, hours)
                ),
            }
            for key, hours in RESOLUTIONS.items()
        },
        "later_expected_base_features_primary": (
            "304 clinical + 1 age + 3 gender + 33 race = 341"
        ),
        "workers": int(workers),
        "patients_per_task": int(
            args.patients_per_task
        ),
        "max_in_flight": int(
            max_in_flight
        ),
        "databases": {},
    }

    window_summary_rows: List[dict] = []
    consistency_rows: List[dict] = []

    for database_key in databases:
        result, resolution_rows, consistency = (
            run_database(
                database_key=database_key,
                input_dir=input_dir,
                output_dir=output_dir,
                concepts=concepts,
                aggregations=aggregations,
                feature_columns=feature_columns,
                workers=workers,
                patients_per_task=args.patients_per_task,
                max_in_flight=max_in_flight,
                logger=logger,
            )
        )

        manifest["databases"][
            database_key
        ] = result
        window_summary_rows.extend(
            resolution_rows
        )
        consistency_rows.append(
            consistency
        )

    reports_dir = (
        output_dir
        / "reports"
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_summary = pd.DataFrame(
        window_summary_rows
    )
    window_summary.to_csv(
        reports_dir / "window_summary.csv",
        index=False,
    )

    consistency_df = pd.DataFrame(
        consistency_rows
    )
    consistency_df.to_csv(
        reports_dir / "final_48h_consistency.csv",
        index=False,
    )

    if (
        not consistency_df.empty
        and not consistency_df["passed"].all()
    ):
        raise RuntimeError(
            "At least one database failed the final 48h "
            "multi-resolution consistency check."
        )

    manifest["all_final_48h_consistency_checks_passed"] = bool(
        consistency_df["passed"].all()
        if not consistency_df.empty
        else False
    )

    manifest_path = (
        output_dir
        / "04_cumulative_windows_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Saved window summary: %s",
        reports_dir / "window_summary.csv",
    )
    logger.info(
        "Saved final 48h consistency report: %s",
        reports_dir / "final_48h_consistency.csv",
    )
    logger.info(
        "Saved manifest: %s",
        manifest_path,
    )
    logger.info(
        "Stage 04 completed successfully. "
        "No demographics, split, imputation, scaling, or tensor "
        "upsampling was performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
