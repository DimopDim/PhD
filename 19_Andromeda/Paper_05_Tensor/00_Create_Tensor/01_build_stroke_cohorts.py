#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_build_stroke_cohorts.py

Build the stroke ICU cohorts for MIMIC-IV v3.1 and eICU v2.0 from ICD-code
lists stored in the project directory.

Default server layout
---------------------
Project/code:
    /home/ddimopoulos/Paper_05_Tensor/

MIMIC-IV:
    /home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1/
        derived/
        hosp/
        icu/
        note/

eICU:
    /home/ddimopoulos/Datasets/00_Datasets/eicu-2_0/

ICD lists:
    /home/ddimopoulos/Paper_05_Tensor/mimic_icd_stroke.csv
    /home/ddimopoulos/Paper_05_Tensor/eicu_icd_stroke.csv

Outputs:
    /home/ddimopoulos/Paper_05_Tensor/data/01_cohorts/

The script is intentionally independent of Jupyter and uses deterministic,
logged cohort construction.

Important:
- MIMIC matching is performed against hosp/diagnoses_icd.
- eICU matching is performed against diagnosis.icd9code. Entries containing
  multiple comma-separated codes are split before matching.
- ICD strings are normalized for whitespace/case. Decimal points are preserved.
- For MIMIC, the chronologically first eligible ICU stay per subject is retained.
- eICU does not provide a reliable absolute calendar ordering across separate
  hospital encounters for the same uniquepid. The script therefore retains one
  deterministic eligible ICU stay per uniquepid using:
      unitvisitnumber -> patienthealthsystemstayid -> patientunitstayid
  and records this rule in the manifest. The all-eligible-stays file is also
  saved, so this choice can be audited or changed before model training.
- ICU stays > max_icu_los_days are excluded by default (10 days, manuscript rule).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor")
MIMIC_ROOT_DEFAULT = Path("/home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1")
EICU_ROOT_DEFAULT = Path("/home/ddimopoulos/Datasets/00_Datasets/eicu-2_0")

MIMIC_ICD_DEFAULT = PROJECT_ROOT_DEFAULT / "mimic_icd_stroke.csv"
EICU_ICD_DEFAULT = PROJECT_ROOT_DEFAULT / "eicu_icd_stroke.csv"
OUTPUT_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "01_cohorts"


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stroke_cohorts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        output_dir / "01_build_stroke_cohorts.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def find_table(directory: Path, stem: str) -> Path:
    """
    Resolve a table using common PhysioNet formats.
    Preference: parquet -> csv.gz -> csv.
    """
    candidates = [
        directory / f"{stem}.parquet",
        directory / f"{stem}.csv.gz",
        directory / f"{stem}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find table '{stem}' in {directory}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def normalize_icd(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if not text:
        return None

    # Common artifact when a code was read numerically and converted to string.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    # Remove whitespace only; preserve clinically meaningful punctuation such as "."
    text = re.sub(r"\s+", "", text)
    return text or None


def load_icd_codes(path: Path, preferred_column: str) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"ICD list not found: {path}\n"
            f"Place the file in the project root or pass an explicit path."
        )

    df = pd.read_csv(path, dtype=str)
    if df.empty:
        raise ValueError(f"ICD list is empty: {path}")

    if preferred_column in df.columns:
        series = df[preferred_column]
    elif len(df.columns) == 1:
        series = df.iloc[:, 0]
    else:
        raise ValueError(
            f"{path} must contain column '{preferred_column}' "
            f"or exactly one column. Found: {df.columns.tolist()}"
        )

    codes = [normalize_icd(x) for x in series.tolist()]
    codes = sorted({x for x in codes if x is not None})
    if not codes:
        raise ValueError(f"No valid ICD codes were read from {path}")
    return codes


def read_table(path: Path, usecols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(usecols) if usecols else None)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def read_csv_in_chunks(
    path: Path,
    usecols: Sequence[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    if path.suffix == ".parquet":
        # Parquet is already columnar. Yield a single projected DataFrame.
        yield pd.read_parquet(path, columns=list(usecols))
        return

    yield from pd.read_csv(
        path,
        usecols=list(usecols),
        chunksize=chunksize,
        low_memory=False,
    )


def write_frame(df: pd.DataFrame, path_without_suffix: Path) -> Tuple[Path, str]:
    """
    Prefer Parquet with zstd. If pyarrow/fastparquet is unavailable,
    fall back to gzip-compressed CSV rather than aborting the cohort step.
    """
    parquet_path = path_without_suffix.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False, compression="zstd")
        return parquet_path, "parquet"
    except Exception as exc:
        csv_path = path_without_suffix.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        return csv_path, f"csv.gz fallback ({type(exc).__name__})"


def build_mimic_cohort(
    mimic_root: Path,
    icd_codes: Sequence[str],
    output_dir: Path,
    chunksize: int,
    max_icu_los_days: Optional[float],
    logger: logging.Logger,
) -> dict:
    hosp_dir = mimic_root / "hosp"
    icu_dir = mimic_root / "icu"

    diagnoses_path = find_table(hosp_dir, "diagnoses_icd")
    icustays_path = find_table(icu_dir, "icustays")

    logger.info("MIMIC diagnoses: %s", diagnoses_path)
    logger.info("MIMIC ICU stays: %s", icustays_path)

    code_set: Set[str] = set(icd_codes)
    matched_chunks: List[pd.DataFrame] = []

    diag_cols = ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"]
    logger.info("Scanning MIMIC diagnoses for %d ICD codes...", len(code_set))

    total_diag_rows = 0
    for chunk in read_csv_in_chunks(diagnoses_path, diag_cols, chunksize):
        total_diag_rows += len(chunk)
        norm = chunk["icd_code"].map(normalize_icd)
        keep = norm.isin(code_set)
        if keep.any():
            hit = chunk.loc[keep].copy()
            hit["matched_icd_code"] = norm.loc[keep].values
            matched_chunks.append(hit)

    if not matched_chunks:
        raise RuntimeError(
            "No MIMIC diagnoses matched the supplied ICD list. "
            "Check code formatting and dataset version."
        )

    matches = pd.concat(matched_chunks, ignore_index=True)
    matches = matches.drop_duplicates(
        ["subject_id", "hadm_id", "matched_icd_code"]
    ).reset_index(drop=True)

    # Admission-level code audit.
    code_audit = (
        matches.groupby(["subject_id", "hadm_id"], as_index=False)
        .agg(
            matched_icd_codes=(
                "matched_icd_code",
                lambda s: "|".join(sorted(set(map(str, s)))),
            ),
            n_matched_icd_codes=("matched_icd_code", "nunique"),
        )
    )

    icu_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "last_careunit",
        "intime",
        "outtime",
        "los",
    ]
    icustays = read_table(icustays_path, usecols=icu_cols)
    icustays["intime"] = pd.to_datetime(icustays["intime"], errors="coerce")
    icustays["outtime"] = pd.to_datetime(icustays["outtime"], errors="coerce")
    icustays["los"] = pd.to_numeric(icustays["los"], errors="coerce")

    eligible = icustays.merge(code_audit, on=["subject_id", "hadm_id"], how="inner")
    eligible = eligible.dropna(subset=["subject_id", "hadm_id", "stay_id", "intime"])

    n_before_los = len(eligible)
    if max_icu_los_days is not None:
        eligible = eligible[
            eligible["los"].notna()
            & (eligible["los"] > 0)
            & (eligible["los"] <= max_icu_los_days)
        ].copy()

    eligible = eligible.sort_values(
        ["subject_id", "intime", "hadm_id", "stay_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    # First eligible ICU stay for each patient.
    first = eligible.drop_duplicates("subject_id", keep="first").copy()
    first["database"] = "MIMIC-IV"
    first["cohort_patient_id"] = first["subject_id"].astype(str)
    first["cohort_stay_id"] = first["stay_id"].astype(str)

    matches_path, matches_format = write_frame(
        matches, output_dir / "mimic_stroke_diagnosis_matches"
    )
    eligible_path, eligible_format = write_frame(
        eligible, output_dir / "mimic_all_eligible_icu_stays"
    )
    first_path, first_format = write_frame(
        first, output_dir / "mimic_stroke_first_icu_stay"
    )

    logger.info(
        "MIMIC: scanned diagnosis rows=%d, matched diagnosis rows=%d, "
        "eligible ICU stays=%d, first-stay patients=%d",
        total_diag_rows,
        len(matches),
        len(eligible),
        len(first),
    )

    return {
        "diagnosis_rows_scanned": int(total_diag_rows),
        "matched_diagnosis_rows": int(len(matches)),
        "matched_hospital_admissions": int(
            matches[["subject_id", "hadm_id"]].drop_duplicates().shape[0]
        ),
        "eligible_icu_stays_before_los_filter": int(n_before_los),
        "eligible_icu_stays_after_los_filter": int(len(eligible)),
        "unique_patients_final": int(first["subject_id"].nunique()),
        "files": {
            "diagnosis_matches": str(matches_path),
            "all_eligible_stays": str(eligible_path),
            "first_icu_stay": str(first_path),
        },
        "formats": {
            "diagnosis_matches": matches_format,
            "all_eligible_stays": eligible_format,
            "first_icu_stay": first_format,
        },
        "selection_rule": "chronologically earliest eligible ICU intime per subject_id",
    }


def explode_eicu_codes(series: pd.Series) -> pd.DataFrame:
    """
    Return a two-column helper frame:
      original_index, normalized_code
    for comma-separated eICU diagnosis codes.
    """
    temp = series.fillna("").astype(str).str.split(r"\s*,\s*", regex=True)
    exploded = temp.explode()
    norm = exploded.map(normalize_icd)
    out = pd.DataFrame(
        {
            "original_index": exploded.index,
            "normalized_code": norm.values,
        }
    )
    return out.dropna(subset=["normalized_code"])


def build_eicu_cohort(
    eicu_root: Path,
    icd_codes: Sequence[str],
    output_dir: Path,
    chunksize: int,
    max_icu_los_days: Optional[float],
    logger: logging.Logger,
) -> dict:
    diagnosis_path = find_table(eicu_root, "diagnosis")
    patient_path = find_table(eicu_root, "patient")

    logger.info("eICU diagnosis: %s", diagnosis_path)
    logger.info("eICU patient: %s", patient_path)

    code_set: Set[str] = set(icd_codes)
    matched_chunks: List[pd.DataFrame] = []

    diag_cols = [
        "patientunitstayid",
        "diagnosisoffset",
        "diagnosisstring",
        "icd9code",
    ]

    logger.info("Scanning eICU diagnosis for %d ICD codes...", len(code_set))
    total_diag_rows = 0

    for chunk in read_csv_in_chunks(diagnosis_path, diag_cols, chunksize):
        total_diag_rows += len(chunk)
        exploded = explode_eicu_codes(chunk["icd9code"])
        exploded = exploded[exploded["normalized_code"].isin(code_set)]
        if exploded.empty:
            continue

        hit_rows = chunk.loc[exploded["original_index"].unique()].copy()
        # Attach all matching codes per original diagnosis row.
        per_row_codes = (
            exploded.groupby("original_index")["normalized_code"]
            .agg(lambda s: "|".join(sorted(set(map(str, s)))))
        )
        hit_rows["matched_icd_codes"] = hit_rows.index.map(per_row_codes)
        matched_chunks.append(hit_rows)

    if not matched_chunks:
        raise RuntimeError(
            "No eICU diagnoses matched the supplied ICD list. "
            "Check code formatting and the eICU diagnosis.icd9code field."
        )

    matches = pd.concat(matched_chunks, ignore_index=True)
    matches = matches.drop_duplicates(
        ["patientunitstayid", "matched_icd_codes", "diagnosisoffset"]
    ).reset_index(drop=True)

    stay_code_audit = (
        matches.groupby("patientunitstayid", as_index=False)
        .agg(
            matched_icd_codes=(
                "matched_icd_codes",
                lambda s: "|".join(
                    sorted(
                        {
                            code
                            for item in s.dropna().astype(str)
                            for code in item.split("|")
                            if code
                        }
                    )
                ),
            ),
            n_matching_diagnosis_rows=("patientunitstayid", "size"),
        )
    )

    # Read patient table and retain the identifiers/outcomes/demographics needed
    # by downstream preprocessing.
    patient_cols = [
        "uniquepid",
        "patienthealthsystemstayid",
        "patientunitstayid",
        "unitvisitnumber",
        "gender",
        "age",
        "ethnicity",
        "hospitaladmitoffset",
        "hospitaladmittime24",
        "hospitaldischargeoffset",
        "hospitaldischargetime24",
        "hospitaldischargestatus",
        "unitadmitoffset",
        "unitadmittime24",
        "unitdischargeoffset",
        "unitdischargetime24",
        "unitdischargestatus",
        "unitstaytype",
        "unittype",
        "unitadmitsource",
        "unitdischargelocation",
        "apacheadmissiondx",
        "admissionheight",
        "admissionweight",
        "dischargeweight",
    ]

    # Some eICU distributions may omit a nonessential column. Resolve safely.
    if patient_path.suffix == ".parquet":
        available_cols = pd.read_parquet(patient_path, columns=None).columns.tolist()
    else:
        available_cols = pd.read_csv(patient_path, nrows=0).columns.tolist()

    mandatory = {
        "uniquepid",
        "patienthealthsystemstayid",
        "patientunitstayid",
        "unitvisitnumber",
        "unitdischargeoffset",
    }
    missing_mandatory = sorted(mandatory - set(available_cols))
    if missing_mandatory:
        raise ValueError(
            f"eICU patient table is missing required columns: {missing_mandatory}"
        )

    selected_cols = [c for c in patient_cols if c in available_cols]
    patient = read_table(patient_path, usecols=selected_cols)

    eligible = patient.merge(stay_code_audit, on="patientunitstayid", how="inner")
    eligible["unitdischargeoffset"] = pd.to_numeric(
        eligible["unitdischargeoffset"], errors="coerce"
    )
    eligible["icu_los_days"] = eligible["unitdischargeoffset"] / 1440.0

    n_before_los = len(eligible)
    if max_icu_los_days is not None:
        eligible = eligible[
            eligible["icu_los_days"].notna()
            & (eligible["icu_los_days"] > 0)
            & (eligible["icu_los_days"] <= max_icu_los_days)
        ].copy()

    # eICU lacks a trustworthy absolute calendar date across separate encounters.
    # Use a documented deterministic rule and save all eligible stays for audit.
    for col in [
        "unitvisitnumber",
        "patienthealthsystemstayid",
        "patientunitstayid",
    ]:
        eligible[col] = pd.to_numeric(eligible[col], errors="coerce")

    eligible = eligible.sort_values(
        [
            "uniquepid",
            "unitvisitnumber",
            "patienthealthsystemstayid",
            "patientunitstayid",
        ],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    first = eligible.drop_duplicates("uniquepid", keep="first").copy()
    first["database"] = "eICU"
    first["cohort_patient_id"] = first["uniquepid"].astype(str)
    first["cohort_stay_id"] = first["patientunitstayid"].astype("Int64").astype(str)

    matches_path, matches_format = write_frame(
        matches, output_dir / "eicu_stroke_diagnosis_matches"
    )
    eligible_path, eligible_format = write_frame(
        eligible, output_dir / "eicu_all_eligible_icu_stays"
    )
    first_path, first_format = write_frame(
        first, output_dir / "eicu_stroke_first_icu_stay"
    )

    logger.info(
        "eICU: scanned diagnosis rows=%d, matched diagnosis rows=%d, "
        "eligible ICU stays=%d, unique final patients=%d",
        total_diag_rows,
        len(matches),
        len(eligible),
        len(first),
    )

    return {
        "diagnosis_rows_scanned": int(total_diag_rows),
        "matched_diagnosis_rows": int(len(matches)),
        "matched_unit_stays": int(matches["patientunitstayid"].nunique()),
        "eligible_icu_stays_before_los_filter": int(n_before_los),
        "eligible_icu_stays_after_los_filter": int(len(eligible)),
        "unique_patients_final": int(first["uniquepid"].nunique()),
        "files": {
            "diagnosis_matches": str(matches_path),
            "all_eligible_stays": str(eligible_path),
            "first_icu_stay": str(first_path),
        },
        "formats": {
            "diagnosis_matches": matches_format,
            "all_eligible_stays": eligible_format,
            "first_icu_stay": first_format,
        },
        "selection_rule": (
            "deterministic eICU rule: lowest unitvisitnumber, then "
            "patienthealthsystemstayid, then patientunitstayid per uniquepid; "
            "all eligible stays are retained separately for audit"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MIMIC-IV and eICU stroke ICU cohorts from ICD CSV lists."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help=f"Project root (default: {PROJECT_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=MIMIC_ROOT_DEFAULT,
        help=f"MIMIC-IV root (default: {MIMIC_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--eicu-root",
        type=Path,
        default=EICU_ROOT_DEFAULT,
        help=f"eICU root (default: {EICU_ROOT_DEFAULT})",
    )
    parser.add_argument(
        "--mimic-icd",
        type=Path,
        default=None,
        help=(
            "MIMIC stroke ICD CSV. Default: "
            "<project-root>/mimic_icd_stroke.csv"
        ),
    )
    parser.add_argument(
        "--eicu-icd",
        type=Path,
        default=None,
        help=(
            "eICU stroke ICD CSV. Default: "
            "<project-root>/eicu_icd_stroke.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<project-root>/data/01_cohorts"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="CSV chunk size for diagnosis scans (default: 500000).",
    )
    parser.add_argument(
        "--max-icu-los-days",
        type=float,
        default=10.0,
        help=(
            "Exclude ICU stays longer than this value. "
            "Use a negative value to disable. Default: 10 days."
        ),
    )
    parser.add_argument(
        "--database",
        choices=["all", "mimic", "eicu"],
        default="all",
        help="Run both databases or only one (default: all).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = args.project_root.expanduser().resolve()
    mimic_root = args.mimic_root.expanduser().resolve()
    eicu_root = args.eicu_root.expanduser().resolve()

    mimic_icd = (
        args.mimic_icd.expanduser().resolve()
        if args.mimic_icd is not None
        else project_root / "mimic_icd_stroke.csv"
    )
    eicu_icd = (
        args.eicu_icd.expanduser().resolve()
        if args.eicu_icd is not None
        else project_root / "eicu_icd_stroke.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "01_cohorts"
    )

    max_los = None if args.max_icu_los_days < 0 else args.max_icu_los_days

    logger = configure_logging(output_dir)
    logger.info("Project root: %s", project_root)
    logger.info("MIMIC root: %s", mimic_root)
    logger.info("eICU root: %s", eicu_root)
    logger.info("Output directory: %s", output_dir)
    logger.info("Maximum ICU LOS: %s", max_los)

    manifest = {
        "script": "01_build_stroke_cohorts.py",
        "project_root": str(project_root),
        "mimic_root": str(mimic_root),
        "eicu_root": str(eicu_root),
        "mimic_icd_file": str(mimic_icd),
        "eicu_icd_file": str(eicu_icd),
        "output_dir": str(output_dir),
        "chunksize": int(args.chunksize),
        "max_icu_los_days": max_los,
        "databases_requested": args.database,
        "results": {},
    }

    if args.database in {"all", "mimic"}:
        mimic_codes = load_icd_codes(mimic_icd, "icd_code")
        logger.info(
            "Loaded %d MIMIC ICD codes from %s: %s",
            len(mimic_codes),
            mimic_icd,
            ", ".join(mimic_codes),
        )
        manifest["mimic_icd_codes"] = mimic_codes
        manifest["results"]["mimic"] = build_mimic_cohort(
            mimic_root=mimic_root,
            icd_codes=mimic_codes,
            output_dir=output_dir,
            chunksize=args.chunksize,
            max_icu_los_days=max_los,
            logger=logger,
        )

    if args.database in {"all", "eicu"}:
        eicu_codes = load_icd_codes(eicu_icd, "icd9code")
        logger.info(
            "Loaded %d eICU ICD codes from %s: %s",
            len(eicu_codes),
            eicu_icd,
            ", ".join(eicu_codes),
        )
        manifest["eicu_icd_codes"] = eicu_codes
        manifest["results"]["eicu"] = build_eicu_cohort(
            eicu_root=eicu_root,
            icd_codes=eicu_codes,
            output_dir=output_dir,
            chunksize=args.chunksize,
            max_icu_los_days=max_los,
            logger=logger,
        )

    manifest_path = output_dir / "01_cohort_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    logger.info("Saved manifest: %s", manifest_path)
    logger.info("Cohort construction completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
