#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02b_audit_raw_features.py

Audit the raw event files produced by 02_extract_raw_events.py before any
cross-database harmonization or temporal aggregation.

Canonical project root
----------------------
/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

Inputs
------
<project-root>/data/02_raw_events/
    mimic/
        chartevents.parquet
        labevents.parquet
        demographics_outcomes.parquet
    eicu/
        lab.parquet
        customlab.parquet
        nursecharting.parquet
        vitalperiodic.parquet
        vitalaperiodic.parquet
        demographics_outcomes.parquet

Outputs
-------
<project-root>/data/02_raw_events/reports/
    mimic_feature_inventory.csv
    eicu_feature_inventory.csv
    unit_inventory.csv
    source_summary.csv
    patient_event_coverage.csv
    cohort_coverage_summary.csv
    demographics_summary.csv
    mapping_review_candidates.csv
    feature_mapping_template.csv
    02b_audit_manifest.json
    02b_audit_raw_features.log

Design goals
------------
- Memory bounded: scan Parquet files in record batches.
- Exact patient coverage per raw feature.
- Preserve source table, raw variable name, source item id, and observed units.
- No silent harmonization.
- Produce a review list for known high-risk mapping areas:
  troponin, glucose, pH, blood pressure, CRP, reticulocytes, GCS, temperature.
- Produce a blank mapping template that becomes the controlled input to Stage 03.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor"
)
RAW_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "02_raw_events"
REPORT_DIR_DEFAULT = RAW_DIR_DEFAULT / "reports"

BATCH_ROWS_DEFAULT = 250_000
WORKERS_DEFAULT = 4

EVENT_REQUIRED_COLUMNS = {
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
}

SOURCE_FILES = {
    "mimic": [
        ("chartevents", "mimic/chartevents.parquet"),
        ("labevents", "mimic/labevents.parquet"),
    ],
    "eicu": [
        ("lab", "eicu/lab.parquet"),
        ("customlab", "eicu/customlab.parquet"),
        ("nursecharting", "eicu/nursecharting.parquet"),
        ("vitalperiodic", "eicu/vitalperiodic.parquet"),
        ("vitalaperiodic", "eicu/vitalaperiodic.parquet"),
    ],
}

DEMOGRAPHIC_FILES = {
    "mimic": "mimic/demographics_outcomes.parquet",
    "eicu": "eicu/demographics_outcomes.parquet",
}


def require_pyarrow() -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "02b_audit_raw_features.py requires pyarrow.\n"
            "Install it with: pip install pyarrow"
        ) from exc


def configure_logging(report_dir: Path) -> logging.Logger:
    report_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("audit_raw_features")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        report_dir / "02b_audit_raw_features.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value: object) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _merge_min(current: Optional[float], new: object) -> Optional[float]:
    x = safe_float(new)
    if x is None:
        return current
    if current is None:
        return x
    return min(current, x)


def _merge_max(current: Optional[float], new: object) -> Optional[float]:
    x = safe_float(new)
    if x is None:
        return current
    if current is None:
        return x
    return max(current, x)


def _feature_key(
    source_table: object,
    source_variable: object,
    source_itemid: object,
) -> Tuple[str, str, str]:
    return (
        normalize_text(source_table),
        normalize_text(source_variable),
        normalize_text(source_itemid),
    )


def audit_event_file(
    database_key: str,
    expected_source: str,
    path_str: str,
    batch_rows: int,
) -> dict:
    """
    Worker function: audit one Parquet source file.

    Returns compact aggregated state, never raw events.
    """
    require_pyarrow()
    import pyarrow.parquet as pq

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(path)

    parquet = pq.ParquetFile(path)
    columns = parquet.schema.names
    missing = sorted(EVENT_REQUIRED_COLUMNS - set(columns))
    if missing:
        raise ValueError(
            f"{path} is missing Stage-02 columns: {missing}"
        )

    feature_stats: Dict[Tuple[str, str, str], dict] = {}
    unit_stats: Dict[Tuple[str, str, str, str], dict] = {}
    feature_patients: Dict[Tuple[str, str, str], Set[str]] = defaultdict(set)

    patient_stats: Dict[str, dict] = {}
    patient_feature_pairs: Set[Tuple[str, Tuple[str, str, str]]] = set()

    total_rows = 0
    min_offset: Optional[float] = None
    max_offset: Optional[float] = None
    observed_database_values: Set[str] = set()
    observed_source_values: Set[str] = set()

    columns_to_read = [
        "database",
        "patient_id",
        "source_table",
        "source_variable",
        "source_itemid",
        "event_offset_minutes",
        "value",
        "unit",
    ]

    start = time.monotonic()

    for batch in parquet.iter_batches(
        batch_size=batch_rows,
        columns=columns_to_read,
    ):
        df = batch.to_pandas()
        if df.empty:
            continue

        total_rows += len(df)

        for col in [
            "database",
            "patient_id",
            "source_table",
            "source_variable",
            "source_itemid",
            "unit",
        ]:
            df[col] = df[col].fillna("").astype(str).str.strip()

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["event_offset_minutes"] = pd.to_numeric(
            df["event_offset_minutes"], errors="coerce"
        )

        observed_database_values.update(
            x for x in df["database"].unique().tolist() if x
        )
        observed_source_values.update(
            x for x in df["source_table"].unique().tolist() if x
        )

        batch_min_offset = df["event_offset_minutes"].min(skipna=True)
        batch_max_offset = df["event_offset_minutes"].max(skipna=True)
        min_offset = _merge_min(min_offset, batch_min_offset)
        max_offset = _merge_max(max_offset, batch_max_offset)

        # Feature-level counts and numerical ranges.
        feature_grouped = (
            df.groupby(
                ["source_table", "source_variable", "source_itemid"],
                dropna=False,
                sort=False,
            )
            .agg(
                event_count=("value", "size"),
                value_min=("value", "min"),
                value_max=("value", "max"),
                offset_min_minutes=("event_offset_minutes", "min"),
                offset_max_minutes=("event_offset_minutes", "max"),
            )
            .reset_index()
        )

        for row in feature_grouped.itertuples(index=False):
            key = _feature_key(
                row.source_table,
                row.source_variable,
                row.source_itemid,
            )
            state = feature_stats.setdefault(
                key,
                {
                    "event_count": 0,
                    "value_min": None,
                    "value_max": None,
                    "offset_min_minutes": None,
                    "offset_max_minutes": None,
                    "units": set(),
                },
            )
            state["event_count"] += int(row.event_count)
            state["value_min"] = _merge_min(
                state["value_min"], row.value_min
            )
            state["value_max"] = _merge_max(
                state["value_max"], row.value_max
            )
            state["offset_min_minutes"] = _merge_min(
                state["offset_min_minutes"], row.offset_min_minutes
            )
            state["offset_max_minutes"] = _merge_max(
                state["offset_max_minutes"], row.offset_max_minutes
            )

        # Unit profile.
        unit_grouped = (
            df.groupby(
                [
                    "source_table",
                    "source_variable",
                    "source_itemid",
                    "unit",
                ],
                dropna=False,
                sort=False,
            )
            .agg(
                event_count=("value", "size"),
                value_min=("value", "min"),
                value_max=("value", "max"),
            )
            .reset_index()
        )

        for row in unit_grouped.itertuples(index=False):
            fkey = _feature_key(
                row.source_table,
                row.source_variable,
                row.source_itemid,
            )
            unit = normalize_text(row.unit)
            feature_stats[fkey]["units"].add(unit)

            ukey = (*fkey, unit)
            state = unit_stats.setdefault(
                ukey,
                {
                    "event_count": 0,
                    "value_min": None,
                    "value_max": None,
                },
            )
            state["event_count"] += int(row.event_count)
            state["value_min"] = _merge_min(
                state["value_min"], row.value_min
            )
            state["value_max"] = _merge_max(
                state["value_max"], row.value_max
            )

        # Exact distinct patients per feature.
        unique_patient_feature = df[
            [
                "patient_id",
                "source_table",
                "source_variable",
                "source_itemid",
            ]
        ].drop_duplicates()

        for row in unique_patient_feature.itertuples(index=False):
            patient_id = normalize_text(row.patient_id)
            if not patient_id:
                continue
            fkey = _feature_key(
                row.source_table,
                row.source_variable,
                row.source_itemid,
            )
            feature_patients[fkey].add(patient_id)
            patient_feature_pairs.add((patient_id, fkey))

        # Patient-level event count and time span.
        patient_grouped = (
            df.groupby("patient_id", sort=False)
            .agg(
                event_count=("value", "size"),
                first_event_minutes=("event_offset_minutes", "min"),
                last_event_minutes=("event_offset_minutes", "max"),
            )
            .reset_index()
        )

        for row in patient_grouped.itertuples(index=False):
            patient_id = normalize_text(row.patient_id)
            if not patient_id:
                continue

            state = patient_stats.setdefault(
                patient_id,
                {
                    "event_count": 0,
                    "first_event_minutes": None,
                    "last_event_minutes": None,
                },
            )
            state["event_count"] += int(row.event_count)
            state["first_event_minutes"] = _merge_min(
                state["first_event_minutes"], row.first_event_minutes
            )
            state["last_event_minutes"] = _merge_max(
                state["last_event_minutes"], row.last_event_minutes
            )

    # Compute exact distinct-feature counts per patient for this source.
    patient_feature_count: Dict[str, int] = defaultdict(int)
    for patient_id, _ in patient_feature_pairs:
        patient_feature_count[patient_id] += 1

    for patient_id, count in patient_feature_count.items():
        patient_stats.setdefault(
            patient_id,
            {
                "event_count": 0,
                "first_event_minutes": None,
                "last_event_minutes": None,
            },
        )
        patient_stats[patient_id]["distinct_features"] = int(count)

    for patient_id in patient_stats:
        patient_stats[patient_id].setdefault("distinct_features", 0)

    elapsed = time.monotonic() - start

    serial_feature_stats = []
    for key, state in feature_stats.items():
        source_table, source_variable, source_itemid = key
        serial_feature_stats.append(
            {
                "source_table": source_table,
                "source_variable": source_variable,
                "source_itemid": source_itemid,
                "event_count": int(state["event_count"]),
                "patient_ids": sorted(feature_patients.get(key, set())),
                "units": sorted(state["units"]),
                "value_min": state["value_min"],
                "value_max": state["value_max"],
                "offset_min_minutes": state["offset_min_minutes"],
                "offset_max_minutes": state["offset_max_minutes"],
            }
        )

    serial_unit_stats = []
    for key, state in unit_stats.items():
        source_table, source_variable, source_itemid, unit = key
        serial_unit_stats.append(
            {
                "source_table": source_table,
                "source_variable": source_variable,
                "source_itemid": source_itemid,
                "unit": unit,
                "event_count": int(state["event_count"]),
                "value_min": state["value_min"],
                "value_max": state["value_max"],
            }
        )

    serial_patient_stats = []
    for patient_id, state in patient_stats.items():
        serial_patient_stats.append(
            {
                "patient_id": patient_id,
                "event_count": int(state["event_count"]),
                "distinct_features": int(state["distinct_features"]),
                "first_event_minutes": state["first_event_minutes"],
                "last_event_minutes": state["last_event_minutes"],
            }
        )

    return {
        "database_key": database_key,
        "expected_source": expected_source,
        "path": str(path),
        "file_size_bytes": int(path.stat().st_size),
        "rows": int(total_rows),
        "observed_database_values": sorted(observed_database_values),
        "observed_source_values": sorted(observed_source_values),
        "min_offset_minutes": min_offset,
        "max_offset_minutes": max_offset,
        "feature_stats": serial_feature_stats,
        "unit_stats": serial_unit_stats,
        "patient_stats": serial_patient_stats,
        "elapsed_seconds": float(elapsed),
    }


def summarize_demographics(
    database: str,
    path: Path,
) -> Tuple[pd.DataFrame, int, Set[str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    if "patient_id" not in df.columns:
        raise ValueError(f"{path} does not contain patient_id")

    patient_ids = set(
        df["patient_id"].dropna().astype(str).str.strip().tolist()
    )
    rows: List[dict] = []

    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        missing = int(s.isna().sum())
        unique_count = int(s.nunique(dropna=True))

        row = {
            "database": database,
            "column": col,
            "dtype": str(s.dtype),
            "rows": int(len(df)),
            "non_null_count": non_null,
            "missing_count": missing,
            "missing_pct": (
                100.0 * missing / len(df) if len(df) else np.nan
            ),
            "unique_count": unique_count,
            "numeric_min": np.nan,
            "numeric_max": np.nan,
            "numeric_mean": np.nan,
            "top_values": "",
        }

        numeric = pd.to_numeric(s, errors="coerce")
        numeric_fraction = (
            float(numeric.notna().mean()) if len(s) else 0.0
        )

        if numeric_fraction >= 0.95 and numeric.notna().any():
            row["numeric_min"] = float(numeric.min())
            row["numeric_max"] = float(numeric.max())
            row["numeric_mean"] = float(numeric.mean())
        else:
            counts = (
                s.dropna()
                .astype(str)
                .value_counts()
                .head(10)
            )
            row["top_values"] = " | ".join(
                f"{value}:{count}"
                for value, count in counts.items()
            )

        rows.append(row)

    return pd.DataFrame(rows), int(len(df)), patient_ids


REVIEW_PATTERNS = [
    (
        "troponin",
        re.compile(r"troponin", re.IGNORECASE),
    ),
    (
        "glucose",
        re.compile(r"glucose", re.IGNORECASE),
    ),
    (
        "pH",
        re.compile(
            r"(^|[^a-z])ph([^a-z]|$)|arterial.*ph|venous.*ph",
            re.IGNORECASE,
        ),
    ),
    (
        "blood_pressure",
        re.compile(
            r"systolic|diastolic|blood.?pressure|non.?invasive|nibp",
            re.IGNORECASE,
        ),
    ),
    (
        "CRP",
        re.compile(
            r"\bcrp\b|c.?reactive",
            re.IGNORECASE,
        ),
    ),
    (
        "reticulocytes",
        re.compile(r"retic", re.IGNORECASE),
    ),
    (
        "GCS",
        re.compile(
            r"\bgcs\b|glasgow|eye.?opening|verbal|motor",
            re.IGNORECASE,
        ),
    ),
    (
        "temperature",
        re.compile(r"temperature|\btemp\b", re.IGNORECASE),
    ),
]


def make_review_candidates(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []

    for record in inventory.to_dict(orient="records"):
        searchable = " ".join(
            [
                str(record.get("source_table", "")),
                str(record.get("source_variable", "")),
                str(record.get("source_itemid", "")),
                str(record.get("observed_units", "")),
            ]
        )

        for topic, pattern in REVIEW_PATTERNS:
            if pattern.search(searchable):
                row = {"review_topic": topic}
                row.update(record)
                rows.append(row)

    if not rows:
        columns = ["review_topic"] + inventory.columns.tolist()
        return pd.DataFrame(columns=columns)

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["review_topic", "database", "source_table", "source_variable"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def build_mapping_template(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    cols = [
        "database",
        "source_table",
        "source_variable",
        "source_itemid",
        "observed_units",
        "event_count",
        "patient_count",
        "patient_coverage_pct",
    ]
    out = inventory[cols].copy()

    out["include"] = ""
    out["canonical_concept"] = ""
    out["target_unit"] = ""
    out["conversion_rule"] = ""
    out["priority_source"] = ""
    out["notes"] = ""

    return out.sort_values(
        ["database", "source_table", "source_variable", "source_itemid"],
        kind="stable",
    ).reset_index(drop=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit Stage-02 raw event Parquet files before harmonization."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/02_raw_events",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Default: <raw-dir>/reports",
    )
    parser.add_argument(
        "--database",
        choices=["all", "mimic", "eicu"],
        default="all",
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=BATCH_ROWS_DEFAULT,
        help=f"Parquet rows per scan batch (default: {BATCH_ROWS_DEFAULT}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=WORKERS_DEFAULT,
        help=(
            f"Independent source files audited concurrently "
            f"(default: {WORKERS_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected Stage-02 source file is missing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    require_pyarrow()

    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    project_root = args.project_root.expanduser().resolve()
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else project_root / "data" / "02_raw_events"
    )
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir is not None
        else raw_dir / "reports"
    )

    logger = configure_logging(report_dir)

    logger.info("Project root: %s", project_root)
    logger.info("Raw event directory: %s", raw_dir)
    logger.info("Report directory: %s", report_dir)
    logger.info("Batch rows: %d", args.batch_rows)
    logger.info("Source-level workers: %d", args.workers)

    databases = (
        ["mimic", "eicu"]
        if args.database == "all"
        else [args.database]
    )

    # Cohort denominators and demographic audit.
    demographics_frames: List[pd.DataFrame] = []
    cohort_sizes: Dict[str, int] = {}
    cohort_patients: Dict[str, Set[str]] = {}

    for db in databases:
        demo_path = raw_dir / DEMOGRAPHIC_FILES[db]
        demo_summary, n_rows, patients = summarize_demographics(
            db, demo_path
        )
        demographics_frames.append(demo_summary)
        cohort_sizes[db] = n_rows
        cohort_patients[db] = patients
        logger.info(
            "%s demographics: rows=%d unique patient_id=%d",
            db,
            n_rows,
            len(patients),
        )

    tasks: List[Tuple[str, str, Path]] = []
    missing_sources: List[str] = []

    for db in databases:
        for expected_source, rel_path in SOURCE_FILES[db]:
            path = raw_dir / rel_path
            if path.is_file():
                tasks.append((db, expected_source, path))
            else:
                missing_sources.append(str(path))

    if missing_sources:
        message = (
            "Missing expected Stage-02 source files:\n  - "
            + "\n  - ".join(missing_sources)
        )
        if args.strict:
            raise FileNotFoundError(message)
        logger.warning(message)

    if not tasks:
        raise RuntimeError("No Stage-02 event files were found to audit.")

    results: List[dict] = []
    max_workers = min(args.workers, len(tasks))

    logger.info(
        "Auditing %d source files with %d workers...",
        len(tasks),
        max_workers,
    )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                audit_event_file,
                db,
                expected_source,
                str(path),
                args.batch_rows,
            ): (db, expected_source, path)
            for db, expected_source, path in tasks
        }

        for future in as_completed(future_map):
            db, source, path = future_map[future]
            result = future.result()
            results.append(result)
            logger.info(
                "%s/%s complete: rows=%d features=%d patients=%d "
                "elapsed=%.1fs",
                db,
                source,
                result["rows"],
                len(result["feature_stats"]),
                len(result["patient_stats"]),
                result["elapsed_seconds"],
            )

    # Build feature inventory.
    inventory_rows: List[dict] = []
    unit_rows: List[dict] = []
    source_rows: List[dict] = []

    # Patient stats merged across source files.
    patient_aggregate: Dict[Tuple[str, str], dict] = {}
    patient_sources: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for result in results:
        db = result["database_key"]
        cohort_n = cohort_sizes[db]
        expected_source = result["expected_source"]

        source_patient_ids = {
            row["patient_id"]
            for row in result["patient_stats"]
        }

        source_rows.append(
            {
                "database": db,
                "source_table": expected_source,
                "path": result["path"],
                "file_size_bytes": result["file_size_bytes"],
                "event_count": result["rows"],
                "patient_count": len(source_patient_ids),
                "patient_coverage_pct": (
                    100.0 * len(source_patient_ids) / cohort_n
                    if cohort_n
                    else np.nan
                ),
                "raw_feature_count": len(result["feature_stats"]),
                "min_offset_minutes": result["min_offset_minutes"],
                "max_offset_minutes": result["max_offset_minutes"],
                "observed_database_values": "|".join(
                    result["observed_database_values"]
                ),
                "observed_source_values": "|".join(
                    result["observed_source_values"]
                ),
                "audit_elapsed_seconds": result["elapsed_seconds"],
            }
        )

        for feature in result["feature_stats"]:
            patient_count = len(feature["patient_ids"])
            inventory_rows.append(
                {
                    "database": db,
                    "source_table": feature["source_table"],
                    "source_variable": feature["source_variable"],
                    "source_itemid": feature["source_itemid"],
                    "observed_units": "|".join(
                        unit if unit else "<blank>"
                        for unit in feature["units"]
                    ),
                    "unit_count": len(feature["units"]),
                    "event_count": feature["event_count"],
                    "patient_count": patient_count,
                    "patient_coverage_pct": (
                        100.0 * patient_count / cohort_n
                        if cohort_n
                        else np.nan
                    ),
                    "value_min": feature["value_min"],
                    "value_max": feature["value_max"],
                    "offset_min_minutes": feature[
                        "offset_min_minutes"
                    ],
                    "offset_max_minutes": feature[
                        "offset_max_minutes"
                    ],
                }
            )

        for unit in result["unit_stats"]:
            unit_rows.append(
                {
                    "database": db,
                    "source_table": unit["source_table"],
                    "source_variable": unit["source_variable"],
                    "source_itemid": unit["source_itemid"],
                    "unit": (
                        unit["unit"] if unit["unit"] else "<blank>"
                    ),
                    "event_count": unit["event_count"],
                    "value_min": unit["value_min"],
                    "value_max": unit["value_max"],
                }
            )

        for patient in result["patient_stats"]:
            patient_id = patient["patient_id"]
            key = (db, patient_id)

            state = patient_aggregate.setdefault(
                key,
                {
                    "event_count": 0,
                    "distinct_features": 0,
                    "first_event_minutes": None,
                    "last_event_minutes": None,
                },
            )

            state["event_count"] += patient["event_count"]
            state["distinct_features"] += patient["distinct_features"]
            state["first_event_minutes"] = _merge_min(
                state["first_event_minutes"],
                patient["first_event_minutes"],
            )
            state["last_event_minutes"] = _merge_max(
                state["last_event_minutes"],
                patient["last_event_minutes"],
            )
            patient_sources[key].add(expected_source)

    inventory = pd.DataFrame(inventory_rows)
    inventory = inventory.sort_values(
        [
            "database",
            "source_table",
            "source_variable",
            "source_itemid",
        ],
        kind="stable",
    ).reset_index(drop=True)

    unit_inventory = pd.DataFrame(unit_rows)
    unit_inventory = unit_inventory.sort_values(
        [
            "database",
            "source_table",
            "source_variable",
            "source_itemid",
            "unit",
        ],
        kind="stable",
    ).reset_index(drop=True)

    source_summary = pd.DataFrame(source_rows)
    source_summary = source_summary.sort_values(
        ["database", "source_table"],
        kind="stable",
    ).reset_index(drop=True)

    # Include every cohort patient, including patients with zero retained events.
    patient_rows: List[dict] = []
    cohort_coverage_rows: List[dict] = []

    for db in databases:
        cohort_ids = cohort_patients[db]
        patients_with_any_event = 0

        source_names = [
            source
            for source, _ in SOURCE_FILES[db]
            if any(
                r["database_key"] == db
                and r["expected_source"] == source
                for r in results
            )
        ]

        per_source_counts = {
            source: 0 for source in source_names
        }

        for patient_id in sorted(cohort_ids):
            key = (db, patient_id)
            state = patient_aggregate.get(
                key,
                {
                    "event_count": 0,
                    "distinct_features": 0,
                    "first_event_minutes": None,
                    "last_event_minutes": None,
                },
            )
            sources = patient_sources.get(key, set())

            if state["event_count"] > 0:
                patients_with_any_event += 1

            for source in sources:
                if source in per_source_counts:
                    per_source_counts[source] += 1

            patient_rows.append(
                {
                    "database": db,
                    "patient_id": patient_id,
                    "event_count": int(state["event_count"]),
                    "distinct_raw_features": int(
                        state["distinct_features"]
                    ),
                    "source_count": len(sources),
                    "sources_present": "|".join(sorted(sources)),
                    "first_event_minutes": state[
                        "first_event_minutes"
                    ],
                    "last_event_minutes": state[
                        "last_event_minutes"
                    ],
                    "has_any_event": bool(
                        state["event_count"] > 0
                    ),
                }
            )

        cohort_n = len(cohort_ids)
        cohort_coverage_rows.append(
            {
                "database": db,
                "cohort_patients": cohort_n,
                "patients_with_any_event": patients_with_any_event,
                "patients_without_events": (
                    cohort_n - patients_with_any_event
                ),
                "any_event_coverage_pct": (
                    100.0 * patients_with_any_event / cohort_n
                    if cohort_n
                    else np.nan
                ),
                **{
                    f"patients_with_{source}": count
                    for source, count in per_source_counts.items()
                },
            }
        )

    patient_coverage = pd.DataFrame(patient_rows).sort_values(
        ["database", "patient_id"],
        kind="stable",
    ).reset_index(drop=True)

    cohort_coverage = pd.DataFrame(cohort_coverage_rows)

    demographics_summary = pd.concat(
        demographics_frames,
        ignore_index=True,
    )

    review_candidates = make_review_candidates(inventory)
    mapping_template = build_mapping_template(inventory)

    report_dir.mkdir(parents=True, exist_ok=True)

    # Requested per-database inventories.
    for db in databases:
        db_inventory = inventory[
            inventory["database"] == db
        ].reset_index(drop=True)

        filename = (
            "mimic_feature_inventory.csv"
            if db == "mimic"
            else "eicu_feature_inventory.csv"
        )
        db_inventory.to_csv(
            report_dir / filename,
            index=False,
        )

    unit_inventory.to_csv(
        report_dir / "unit_inventory.csv",
        index=False,
    )
    source_summary.to_csv(
        report_dir / "source_summary.csv",
        index=False,
    )
    patient_coverage.to_csv(
        report_dir / "patient_event_coverage.csv",
        index=False,
    )
    cohort_coverage.to_csv(
        report_dir / "cohort_coverage_summary.csv",
        index=False,
    )
    demographics_summary.to_csv(
        report_dir / "demographics_summary.csv",
        index=False,
    )
    review_candidates.to_csv(
        report_dir / "mapping_review_candidates.csv",
        index=False,
    )
    mapping_template.to_csv(
        report_dir / "feature_mapping_template.csv",
        index=False,
    )

    manifest = {
        "script": "02b_audit_raw_features.py",
        "project_root": str(project_root),
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "database_requested": args.database,
        "batch_rows": int(args.batch_rows),
        "workers": int(max_workers),
        "cohort_sizes": {
            db: int(cohort_sizes[db])
            for db in databases
        },
        "sources": [
            {
                "database": r["database_key"],
                "source_table": r["expected_source"],
                "path": r["path"],
                "file_size_bytes": r["file_size_bytes"],
                "event_count": r["rows"],
                "raw_feature_count": len(r["feature_stats"]),
                "observed_database_values": r[
                    "observed_database_values"
                ],
                "observed_source_values": r[
                    "observed_source_values"
                ],
                "min_offset_minutes": r["min_offset_minutes"],
                "max_offset_minutes": r["max_offset_minutes"],
                "audit_elapsed_seconds": r["elapsed_seconds"],
            }
            for r in sorted(
                results,
                key=lambda x: (
                    x["database_key"],
                    x["expected_source"],
                ),
            )
        ],
        "total_inventory_rows": int(len(inventory)),
        "review_candidate_rows": int(len(review_candidates)),
        "missing_expected_sources": missing_sources,
        "outputs": {
            "unit_inventory": str(
                report_dir / "unit_inventory.csv"
            ),
            "source_summary": str(
                report_dir / "source_summary.csv"
            ),
            "patient_event_coverage": str(
                report_dir / "patient_event_coverage.csv"
            ),
            "cohort_coverage_summary": str(
                report_dir / "cohort_coverage_summary.csv"
            ),
            "demographics_summary": str(
                report_dir / "demographics_summary.csv"
            ),
            "mapping_review_candidates": str(
                report_dir / "mapping_review_candidates.csv"
            ),
            "feature_mapping_template": str(
                report_dir / "feature_mapping_template.csv"
            ),
        },
        "important_note": (
            "No cross-database feature harmonization or unit conversion "
            "is performed by this audit."
        ),
    }

    manifest_path = report_dir / "02b_audit_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Audit complete: inventory rows=%d review candidates=%d",
        len(inventory),
        len(review_candidates),
    )
    logger.info("Saved reports to: %s", report_dir)
    logger.info("Saved manifest: %s", manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
