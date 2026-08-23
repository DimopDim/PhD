#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_harmonize_events.py

Stage 03: deterministic cross-database event and demographic harmonization.

Canonical root:
    /home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

Inputs:
    data/02_raw_events/{mimic,eicu}/*.parquet
    imports/concept_schema_76.csv
    imports/feature_mapping_76.csv
    imports/pipeline_config.json

Outputs:
    data/03_harmonized/
        mimic/harmonized_events.parquet
        mimic/harmonized_demographics.parquet
        eicu/harmonized_events.parquet
        eicu/harmonized_demographics.parquet
        reports/concept_coverage.csv
        reports/conversion_audit.csv
        reports/review_flags.csv
        03_harmonization_manifest.json
        03_harmonize_events.log

Scientific rules
----------------
- 76 time-varying concepts: the erroneous/redundant "Chloride (serum)" is absent.
- No temporal aggregation occurs here.
- No imputation occurs here.
- Missing values remain missing.
- One prespecified raw source is used per concept/database except MIMIC GCS,
  which is constructed from Eye + Motor + Verbal only when all three are
  available at the same timestamp.
- FiO2 is standardized to percentage (0-1 fractions -> 0-100%; >100 invalid).
- eICU CRP mg/dL -> mg/L.
- eICU ionized calcium mg/dL -> mmol/L.
- eICU ammonia mcg/dL -> umol/L.
- Exact duplicate harmonized observations are removed.
- Sentinel value 999999 is treated as invalid/missing.
- Demographics are harmonized but NOT one-hot encoded at this stage.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor")
RAW_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "02_raw_events"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "03_harmonized"
IMPORT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "imports"
BATCH_ROWS_DEFAULT = 250_000

RAW_SOURCES = {
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

DEMOGRAPHICS_FILES = {
    "mimic": "mimic/demographics_outcomes.parquet",
    "eicu": "eicu/demographics_outcomes.parquet",
}

OUTPUT_COLUMNS = [
    "database",
    "patient_id",
    "stay_id",
    "event_offset_minutes",
    "event_offset_hours",
    "canonical_concept",
    "value",
    "unit",
    "source_table",
    "source_variable",
    "source_itemid",
    "conversion_rule",
]


def require_pyarrow() -> None:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from exc


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("harmonize_events")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        output_dir / "03_harmonize_events.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def normalize_itemid(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        x = float(text)
        if math.isfinite(x) and x.is_integer():
            return str(int(x))
    except Exception:
        pass
    return text


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


class IncrementalParquetWriter:
    def __init__(self, path: Path):
        require_pyarrow()
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.pa = pa
        self.pq = pq
        self.path = path
        self.writer = None
        self.rows = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        df = df[OUTPUT_COLUMNS].copy()
        table = self.pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self.writer.schema:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.rows += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        elif not self.path.exists():
            empty = pd.DataFrame({
                "database": pd.Series(dtype="object"),
                "patient_id": pd.Series(dtype="object"),
                "stay_id": pd.Series(dtype="object"),
                "event_offset_minutes": pd.Series(dtype="float64"),
                "event_offset_hours": pd.Series(dtype="float64"),
                "canonical_concept": pd.Series(dtype="object"),
                "value": pd.Series(dtype="float64"),
                "unit": pd.Series(dtype="object"),
                "source_table": pd.Series(dtype="object"),
                "source_variable": pd.Series(dtype="object"),
                "source_itemid": pd.Series(dtype="object"),
                "conversion_rule": pd.Series(dtype="object"),
            })
            empty.to_parquet(self.path, index=False, compression="zstd")


def apply_conversion(values: pd.Series, rule: str) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce").astype("float64")
    x = x.where(np.isfinite(x))

    # Known sentinel from legacy extraction/data sources.
    x = x.mask(np.isclose(x, 999999.0, equal_nan=False))

    if rule in ("", "identity"):
        return x

    if rule == "crp_mgdl_to_mgl":
        return x * 10.0

    if rule == "ionized_calcium_mgdl_to_mmoll":
        # Calcium atomic weight 40.078 g/mol:
        # 1 mg/dL = 10 mg/L / 40.078 mg/mmol.
        return x / 4.0078

    if rule == "ammonia_mcgdl_to_umoll":
        # NH3 molecular weight ~17.031 g/mol:
        # 1 mcg/dL ~= 0.5872 umol/L.
        return x / 1.7031

    if rule == "fio2_to_percent":
        out = x.copy()
        fraction = (out > 0.0) & (out <= 1.0)
        percent = (out > 1.0) & (out <= 100.0)
        out.loc[fraction] = out.loc[fraction] * 100.0
        out.loc[~(fraction | percent)] = np.nan
        return out

    if rule.startswith("gcs_component_"):
        return x

    raise ValueError(f"Unknown conversion_rule: {rule}")


def load_schema(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"concept_order", "canonical_concept", "target_unit"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Schema missing columns: {sorted(missing)}")
    if len(df) != 76:
        raise ValueError(f"Expected exactly 76 concepts, found {len(df)}")
    if df["canonical_concept"].duplicated().any():
        raise ValueError("Duplicate canonical concepts in schema")
    if "Chloride (serum)" in set(df["canonical_concept"]):
        raise ValueError("Chloride (serum) must not be in the 76-concept schema")
    return df.sort_values("concept_order").reset_index(drop=True)


def load_mapping(path: Path, schema: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {
        "database", "source_table", "source_variable", "source_itemid",
        "canonical_concept", "target_unit", "conversion_rule", "priority",
        "special_role", "review_flag", "notes",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping missing columns: {sorted(missing)}")

    df["source_itemid"] = df["source_itemid"].map(normalize_itemid)
    df["database"] = df["database"].str.strip().str.lower()
    df["source_table"] = df["source_table"].str.strip()
    df["source_variable"] = df["source_variable"].str.strip()
    df["canonical_concept"] = df["canonical_concept"].str.strip()
    df["conversion_rule"] = df["conversion_rule"].str.strip().replace("", "identity")
    df["priority"] = pd.to_numeric(df["priority"], errors="coerce").fillna(1).astype(int)
    df["review_flag"] = pd.to_numeric(df["review_flag"], errors="coerce").fillna(0).astype(int)

    schema_set = set(schema["canonical_concept"])
    unknown = sorted(set(df["canonical_concept"]) - schema_set)
    if unknown:
        raise ValueError(f"Mapping contains concepts absent from schema: {unknown}")

    for database in ("mimic", "eicu"):
        present = set(df.loc[df["database"].eq(database), "canonical_concept"])
        absent = sorted(schema_set - present)
        if absent:
            raise ValueError(f"{database} mapping lacks concepts: {absent}")

    return df


def source_mapping_index(mapping: pd.DataFrame, database: str, source_table: str):
    sub = mapping[
        mapping["database"].eq(database)
        & mapping["source_table"].eq(source_table)
    ].copy()
    direct = sub[~sub["conversion_rule"].str.startswith("gcs_component_")].copy()

    index: Dict[Tuple[str, str], dict] = {}
    wildcard: Dict[str, dict] = {}

    for row in direct.sort_values("priority").to_dict(orient="records"):
        variable = row["source_variable"]
        itemid = normalize_itemid(row["source_itemid"])
        if itemid:
            key = (variable, itemid)
            index.setdefault(key, row)
        else:
            wildcard.setdefault(variable, row)

    return index, wildcard


def iter_parquet(path: Path, batch_rows: int, columns: Sequence[str]):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=list(columns)):
        yield batch.to_pandas()


def harmonize_source(
    *,
    database: str,
    source_table: str,
    path: Path,
    mapping: pd.DataFrame,
    writer: IncrementalParquetWriter,
    batch_rows: int,
    logger: logging.Logger,
    concept_events: Dict[str, int],
    concept_patients: Dict[str, Set[str]],
    conversion_counts: Dict[Tuple[str, str], int],
) -> dict:
    exact, wildcard = source_mapping_index(mapping, database, source_table)
    mapped_variables = set(v for v, _ in exact.keys()) | set(wildcard.keys())

    if not mapped_variables:
        logger.info("%s/%s: no direct mappings; skipping", database, source_table)
        return {"rows_scanned": 0, "rows_matched": 0, "rows_written": 0}

    columns = [
        "patient_id", "stay_id", "source_table", "source_variable",
        "source_itemid", "event_offset_minutes", "event_offset_hours",
        "value", "unit",
    ]
    scanned = matched = written = 0
    start = time.monotonic()

    for df in iter_parquet(path, batch_rows, columns):
        scanned += len(df)
        df["source_variable"] = df["source_variable"].fillna("").astype(str).str.strip()
        df = df[df["source_variable"].isin(mapped_variables)].copy()
        if df.empty:
            continue

        df["source_itemid_norm"] = df["source_itemid"].map(normalize_itemid)

        rows = []
        for record in df.to_dict(orient="records"):
            variable = normalize_text(record["source_variable"])
            itemid = normalize_itemid(record["source_itemid_norm"])
            rule = exact.get((variable, itemid)) or wildcard.get(variable)
            if rule is None:
                continue
            record["_mapping"] = rule
            rows.append(record)

        if not rows:
            continue

        matched += len(rows)
        selected = pd.DataFrame(rows)
        # Group by mapping rule to vectorize unit conversion.
        output_parts = []

        for (concept, conv, target_unit), g in selected.groupby(
            [
                selected["_mapping"].map(lambda x: x["canonical_concept"]),
                selected["_mapping"].map(lambda x: x["conversion_rule"]),
                selected["_mapping"].map(lambda x: x["target_unit"]),
            ],
            sort=False,
        ):
            values = apply_conversion(g["value"], conv)
            out = pd.DataFrame({
                "database": "MIMIC-IV" if database == "mimic" else "eICU",
                "patient_id": g["patient_id"].astype(str).values,
                "stay_id": g["stay_id"].astype(str).values,
                "event_offset_minutes": pd.to_numeric(
                    g["event_offset_minutes"], errors="coerce"
                ).values,
                "event_offset_hours": pd.to_numeric(
                    g["event_offset_hours"], errors="coerce"
                ).values,
                "canonical_concept": concept,
                "value": values.values,
                "unit": target_unit,
                "source_table": g["source_table"].astype(str).values,
                "source_variable": g["source_variable"].astype(str).values,
                "source_itemid": g["source_itemid"].map(normalize_itemid).values,
                "conversion_rule": conv,
            })
            out = out.dropna(subset=["value", "event_offset_minutes"])
            if out.empty:
                continue

            # Exact duplicates only. We intentionally do not collapse values that
            # occur at different times.
            out = out.drop_duplicates(
                [
                    "patient_id", "stay_id", "event_offset_minutes",
                    "canonical_concept", "value",
                ]
            )

            concept_events[concept] += len(out)
            concept_patients[concept].update(out["patient_id"].astype(str).unique())
            conversion_counts[(concept, conv)] += len(out)
            output_parts.append(out)

        if output_parts:
            result = pd.concat(output_parts, ignore_index=True)
            writer.write(result)
            written += len(result)

    elapsed = time.monotonic() - start
    logger.info(
        "%s/%s complete | scanned=%d matched=%d written=%d elapsed=%.1fs",
        database, source_table, scanned, matched, written, elapsed
    )
    return {
        "rows_scanned": int(scanned),
        "rows_matched": int(matched),
        "rows_written": int(written),
        "elapsed_seconds": float(elapsed),
    }


def build_mimic_gcs(
    path: Path,
    mapping: pd.DataFrame,
    batch_rows: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    sub = mapping[
        mapping["database"].eq("mimic")
        & mapping["source_table"].eq("chartevents")
        & mapping["conversion_rule"].str.startswith("gcs_component_")
    ].copy()

    variables = set(sub["source_variable"])
    if len(variables) != 3:
        raise ValueError("MIMIC GCS requires exactly Eye, Motor, Verbal mapping rows")

    role_map = {
        row["source_variable"]: row["conversion_rule"].replace("gcs_component_", "")
        for row in sub.to_dict(orient="records")
    }

    pieces = []
    cols = [
        "patient_id", "stay_id", "source_variable", "source_itemid",
        "event_offset_minutes", "event_offset_hours", "value",
    ]
    for df in iter_parquet(path, batch_rows, cols):
        df["source_variable"] = df["source_variable"].fillna("").astype(str).str.strip()
        df = df[df["source_variable"].isin(variables)].copy()
        if df.empty:
            continue
        df["component"] = df["source_variable"].map(role_map)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value", "event_offset_minutes", "component"])
        pieces.append(
            df[
                [
                    "patient_id", "stay_id", "event_offset_minutes",
                    "event_offset_hours", "component", "value",
                ]
            ]
        )

    if not pieces:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    x = pd.concat(pieces, ignore_index=True)
    # Duplicate component recordings at the same timestamp are reduced to their
    # median before the three components are combined.
    x = (
        x.groupby(
            [
                "patient_id", "stay_id", "event_offset_minutes",
                "event_offset_hours", "component"
            ],
            as_index=False,
            sort=False,
        )["value"]
        .median()
    )

    wide = x.pivot_table(
        index=["patient_id", "stay_id", "event_offset_minutes", "event_offset_hours"],
        columns="component",
        values="value",
        aggfunc="first",
    ).reset_index()

    required = {"eye", "motor", "verbal"}
    if not required.issubset(wide.columns):
        raise ValueError(f"Missing GCS components after pivot: {required - set(wide.columns)}")

    wide = wide.dropna(subset=["eye", "motor", "verbal"]).copy()
    wide["value"] = wide["eye"] + wide["motor"] + wide["verbal"]
    wide = wide[(wide["value"] >= 3) & (wide["value"] <= 15)].copy()

    out = pd.DataFrame({
        "database": "MIMIC-IV",
        "patient_id": wide["patient_id"].astype(str),
        "stay_id": wide["stay_id"].astype(str),
        "event_offset_minutes": pd.to_numeric(wide["event_offset_minutes"], errors="coerce"),
        "event_offset_hours": pd.to_numeric(wide["event_offset_hours"], errors="coerce"),
        "canonical_concept": "GCS",
        "value": wide["value"].astype(float),
        "unit": "score",
        "source_table": "chartevents",
        "source_variable": "GCS Eye + Motor + Verbal",
        "source_itemid": "220739|223901|223900",
        "conversion_rule": "gcs_complete_components_same_timestamp",
    })
    out = out.drop_duplicates(
        ["patient_id", "stay_id", "event_offset_minutes", "canonical_concept", "value"]
    )
    logger.info("Constructed MIMIC GCS events: %d", len(out))
    return out[OUTPUT_COLUMNS]


def harmonize_demographics(database: str, raw_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(raw_path)

    if database == "mimic":
        out = pd.DataFrame({
            "database": "MIMIC-IV",
            "patient_id": df["patient_id"].astype(str),
            "stay_id": df["stay_id_canonical"].astype(str),
            "age": pd.to_numeric(df["age"], errors="coerce"),
            "gender": df.get("gender", pd.Series(index=df.index, dtype=object))
                        .fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"}),
            "race": df.get("race", pd.Series(index=df.index, dtype=object))
                      .fillna("UNKNOWN").astype(str).str.strip().replace({"": "UNKNOWN"}),
            "icu_los_days": pd.to_numeric(df["icu_los_days"], errors="coerce"),
            "hospital_expire_flag": pd.to_numeric(
                df.get("hospital_expire_flag"), errors="coerce"
            ),
        })
        out["gender"] = out["gender"].replace(
            {"Female": "F", "Male": "M", "Unknown": "UNKNOWN"}
        )
        return out

    age_raw = df.get("age", pd.Series(index=df.index, dtype=object)).astype(str).str.strip()
    age = pd.to_numeric(age_raw.replace({"> 89": "90", ">89": "90"}), errors="coerce")

    gender_map = {
        "Female": "F",
        "Male": "M",
        "Unknown": "UNKNOWN",
        "": "UNKNOWN",
    }
    race_map = {
        "African American": "BLACK/AFRICAN AMERICAN",
        "Caucasian": "WHITE",
        "Hispanic": "HISPANIC OR LATINO",
        "Asian": "ASIAN",
        "Native American": "AMERICAN INDIAN/ALASKA NATIVE",
        "Other/Unknown": "UNKNOWN",
        "": "UNKNOWN",
    }

    gender_raw = df.get("gender", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()
    ethnicity_raw = df.get("ethnicity", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()
    status = df.get("hospitaldischargestatus", pd.Series(index=df.index, dtype=object)).fillna("").astype(str).str.strip()

    mortality = pd.Series(np.nan, index=df.index, dtype="float64")
    mortality.loc[status.str.casefold().eq("alive")] = 0.0
    mortality.loc[status.str.casefold().eq("expired")] = 1.0

    out = pd.DataFrame({
        "database": "eICU",
        "patient_id": df["patient_id"].astype(str),
        "stay_id": df["stay_id_canonical"].astype(str),
        "age": age,
        "gender": gender_raw.map(gender_map).fillna("UNKNOWN"),
        "race": ethnicity_raw.map(race_map).fillna("UNKNOWN"),
        "icu_los_days": pd.to_numeric(df["icu_los_days"], errors="coerce"),
        "hospital_expire_flag": mortality,
    })
    return out


def build_parser():
    p = argparse.ArgumentParser(description="Stage 03 cross-database harmonization.")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--import-dir", type=Path, default=None)
    p.add_argument("--database", choices=["all", "mimic", "eicu"], default="all")
    p.add_argument("--batch-rows", type=int, default=BATCH_ROWS_DEFAULT)
    p.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Abort if the mapping contains rows explicitly marked for review."
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    require_pyarrow()

    project_root = args.project_root.expanduser().resolve()
    raw_dir = (
        args.raw_dir.expanduser().resolve()
        if args.raw_dir is not None
        else project_root / "data" / "02_raw_events"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "03_harmonized"
    )
    import_dir = (
        args.import_dir.expanduser().resolve()
        if args.import_dir is not None
        else project_root / "imports"
    )

    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")

    logger = configure_logging(output_dir)

    schema_path = import_dir / "concept_schema_76.csv"
    mapping_path = import_dir / "feature_mapping_76.csv"
    config_path = import_dir / "pipeline_config.json"

    schema = load_schema(schema_path)
    mapping = load_mapping(mapping_path, schema)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("aggregations") != ["mean", "median", "min", "max"]:
        raise ValueError(
            "Primary pipeline must currently use exactly mean/median/min/max."
        )
    if int(config.get("clinical_concept_count", -1)) != 76:
        raise ValueError("pipeline_config clinical_concept_count must be 76")
    if int(config.get("expected_base_features_per_temporal_step", -1)) != 341:
        raise ValueError("pipeline_config expected base feature count must be 341")

    review = mapping[mapping["review_flag"].eq(1)].copy()
    if not review.empty:
        logger.warning(
            "Mapping contains %d review-flagged rows. See reports/review_flags.csv",
            len(review)
        )
        if args.fail_on_review:
            raise RuntimeError("Review-flagged mapping rows present; aborting.")

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(reports_dir / "review_flags.csv", index=False)

    databases = ["mimic", "eicu"] if args.database == "all" else [args.database]

    manifest = {
        "script": "03_harmonize_events.py",
        "project_root": str(project_root),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "schema": str(schema_path),
        "mapping": str(mapping_path),
        "pipeline_config": str(config_path),
        "clinical_concepts": 76,
        "chloride_serum_removed": True,
        "within_window_aggregations_for_next_stage": config["aggregations"],
        "expected_base_features_per_temporal_step_after_encoding": 341,
        "databases": {},
    }

    coverage_rows = []
    conversion_rows = []

    for db in databases:
        logger.info("Starting %s harmonization...", db)

        writer = IncrementalParquetWriter(
            output_dir / db / "harmonized_events.parquet"
        )

        concept_events = defaultdict(int)
        concept_patients = defaultdict(set)
        conversion_counts = defaultdict(int)
        source_results = {}

        try:
            for source_table, rel in RAW_SOURCES[db]:
                path = raw_dir / rel
                if not path.is_file():
                    raise FileNotFoundError(path)

                source_results[source_table] = harmonize_source(
                    database=db,
                    source_table=source_table,
                    path=path,
                    mapping=mapping,
                    writer=writer,
                    batch_rows=args.batch_rows,
                    logger=logger,
                    concept_events=concept_events,
                    concept_patients=concept_patients,
                    conversion_counts=conversion_counts,
                )

            if db == "mimic":
                gcs = build_mimic_gcs(
                    raw_dir / "mimic" / "chartevents.parquet",
                    mapping,
                    args.batch_rows,
                    logger,
                )
                if not gcs.empty:
                    writer.write(gcs)
                    concept_events["GCS"] += len(gcs)
                    concept_patients["GCS"].update(
                        gcs["patient_id"].astype(str).unique()
                    )
                    conversion_counts[
                        ("GCS", "gcs_complete_components_same_timestamp")
                    ] += len(gcs)
        finally:
            writer.close()

        demo_raw = raw_dir / DEMOGRAPHICS_FILES[db]
        demo = harmonize_demographics(db, demo_raw)
        demo_path = output_dir / db / "harmonized_demographics.parquet"
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        demo.to_parquet(demo_path, index=False, compression="zstd")

        cohort_n = len(demo)
        for concept in schema["canonical_concept"]:
            n_pat = len(concept_patients.get(concept, set()))
            coverage_rows.append({
                "database": db,
                "canonical_concept": concept,
                "event_count": int(concept_events.get(concept, 0)),
                "patient_count": int(n_pat),
                "cohort_patients": int(cohort_n),
                "patient_coverage_pct": (
                    100.0 * n_pat / cohort_n if cohort_n else np.nan
                ),
            })

        for (concept, rule), count in sorted(conversion_counts.items()):
            conversion_rows.append({
                "database": db,
                "canonical_concept": concept,
                "conversion_rule": rule,
                "event_count": int(count),
            })

        zero_concepts = [
            c for c in schema["canonical_concept"]
            if concept_events.get(c, 0) == 0
        ]
        if zero_concepts:
            logger.warning("%s concepts with zero events: %s", db, zero_concepts)

        manifest["databases"][db] = {
            "harmonized_event_file": str(
                output_dir / db / "harmonized_events.parquet"
            ),
            "harmonized_demographics_file": str(demo_path),
            "harmonized_events": int(writer.rows),
            "patients": int(cohort_n),
            "concepts_with_events": int(
                sum(concept_events.get(c, 0) > 0 for c in schema["canonical_concept"])
            ),
            "zero_event_concepts": zero_concepts,
            "source_results": source_results,
        }

        logger.info(
            "%s complete: patients=%d harmonized_events=%d concepts_with_events=%d/76",
            db,
            cohort_n,
            writer.rows,
            manifest["databases"][db]["concepts_with_events"],
        )

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(reports_dir / "concept_coverage.csv", index=False)

    conversion_df = pd.DataFrame(conversion_rows)
    conversion_df.to_csv(reports_dir / "conversion_audit.csv", index=False)

    # Verify the final schema configuration expected by Stage 04.
    manifest["schema_validation"] = {
        "clinical_concept_count": len(schema),
        "aggregations": config["aggregations"],
        "clinical_descriptors_per_step": len(schema) * len(config["aggregations"]),
        "age_features": 1,
        "gender_features_expected": 3,
        "race_features_expected": 33,
        "base_features_per_step_expected": (
            len(schema) * len(config["aggregations"]) + 1 + 3 + 33
        ),
    }

    manifest_path = output_dir / "03_harmonization_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved coverage: %s", reports_dir / "concept_coverage.csv")
    logger.info("Saved conversion audit: %s", reports_dir / "conversion_audit.csv")
    logger.info("Saved manifest: %s", manifest_path)
    logger.info(
        "Stage 03 complete. No temporal aggregation or one-hot encoding was performed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
