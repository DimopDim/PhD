#!/usr/bin/env python3
"""
05b_tripod_ai_cohort_characteristics.py

Generate reproducible cohort-characteristics outputs aligned with the participant-
reporting components of TRIPOD+AI (especially Items 20b and 20c) for the
Paper_05_Tensor dynamic-landmark study.

This script is DESCRIPTIVE ONLY. It does not change the modeling pipeline, the frozen
feature schema, risk sets, model fitting, calibration, or external evaluation.

Primary inputs (already produced by the canonical pipeline):
  data/03_harmonized/mimic/harmonized_demographics.parquet
  data/03_harmonized/eicu/harmonized_demographics.parquet
  data/05_splits/reports/full_cohort_split_assignments.csv

Outputs:
  tripod_ai_item20b_cohort_characteristics_long.csv
  tripod_ai_item20b_cohort_characteristics_wide.csv
  tripod_ai_race_distribution.csv
  tripod_ai_item20c_demographic_outcome_smd.csv
  tripod_ai_cohort_table.tex
  tripod_ai_manifest.json
  05b_tripod_ai_cohort_characteristics.log

Notes
-----
* MIMIC-IV development/test assignment is read from the frozen Stage-05 split; it is
  NOT recomputed here.
* eICU-CRD is treated as the external evaluation cohort.
* Race/ethnicity is collapsed ONLY FOR REPORTING into broad categories. This does not
  alter the model's frozen demographic encoder.
* Standardized mean differences (SMDs) are descriptive effect-size summaries; no
  p-values or significance tests are produced.
* For continuous variables the manuscript-facing table uses median (IQR), while SMDs
  use means and standard deviations by the conventional definition.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor")
EXPECTED_COUNTS = {
    "MIMIC-IV development": 2599,
    "MIMIC-IV test": 650,
    "eICU-CRD external": 4943,
}
COHORT_ORDER = [
    "MIMIC-IV development",
    "MIMIC-IV test",
    "eICU-CRD external",
]
RACE_ORDER = [
    "White",
    "Black/African American",
    "Hispanic/Latino",
    "Asian",
    "American Indian/Alaska Native",
    "Other",
    "Unknown",
]
SEX_ORDER = ["Female", "Male", "Unknown"]


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tripod_ai_cohort_characteristics")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        output_dir / "05b_tripod_ai_cohort_characteristics.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def broad_race(value: object) -> str:
    """Reporting-only cross-database race/ethnicity grouping."""
    if value is None or pd.isna(value):
        return "Unknown"

    text = re.sub(r"\s+", " ", str(value).strip().upper())
    if text in {
        "",
        "UNKNOWN",
        "UNK",
        "NAN",
        "NONE",
        "NULL",
        "UNABLE TO OBTAIN",
        "PATIENT DECLINED TO ANSWER",
        "OTHER/UNKNOWN",
    }:
        return "Unknown"

    if "BLACK" in text or "AFRICAN AMERICAN" in text:
        return "Black/African American"
    if "HISPANIC" in text or "LATINO" in text:
        return "Hispanic/Latino"
    if "AMERICAN INDIAN" in text or "ALASKA NATIVE" in text or "NATIVE AMERICAN" in text:
        return "American Indian/Alaska Native"
    if "ASIAN" in text:
        return "Asian"
    if "WHITE" in text or "CAUCASIAN" in text:
        return "White"
    return "Other"


def broad_sex(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value).strip().upper()
    if text in {"F", "FEMALE"}:
        return "Female"
    if text in {"M", "MALE"}:
        return "Male"
    return "Unknown"


def fmt_count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return "NA"
    return f"{n:,} ({100.0 * n / denom:.1f}%)"


def fmt_median_iqr(series: pd.Series, decimals: int = 1) -> str:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return "NA"
    q1 = float(x.quantile(0.25))
    med = float(x.quantile(0.50))
    q3 = float(x.quantile(0.75))
    iqr = q3 - q1
    return f"{med:.{decimals}f} ({iqr:.{decimals}f})"


def continuous_stats(series: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if x.empty:
        return {
            "n_nonmissing": 0,
            "n_missing": int(len(series)),
            "mean": np.nan,
            "sd": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return {
        "n_nonmissing": int(x.size),
        "n_missing": int(len(series) - x.size),
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)) if x.size > 1 else np.nan,
        "median": float(x.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def smd_continuous(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna().astype(float)
    y = pd.to_numeric(b, errors="coerce").dropna().astype(float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = math.sqrt((float(x.var(ddof=1)) + float(y.var(ddof=1))) / 2.0)
    if pooled == 0:
        return 0.0 if math.isclose(float(x.mean()), float(y.mean())) else np.nan
    return (float(y.mean()) - float(x.mean())) / pooled


def smd_binary(p_dev: float, p_eval: float) -> float:
    if any(pd.isna(v) for v in [p_dev, p_eval]):
        return np.nan
    pooled = math.sqrt((p_dev * (1.0 - p_dev) + p_eval * (1.0 - p_eval)) / 2.0)
    if pooled == 0:
        return 0.0 if math.isclose(p_dev, p_eval) else np.nan
    return (p_eval - p_dev) / pooled


def load_inputs(project_root: Path, logger: logging.Logger) -> Dict[str, pd.DataFrame]:
    harmonized_dir = project_root / "data" / "03_harmonized"
    split_path = project_root / "data" / "05_splits" / "reports" / "full_cohort_split_assignments.csv"
    mimic_path = harmonized_dir / "mimic" / "harmonized_demographics.parquet"
    eicu_path = harmonized_dir / "eicu" / "harmonized_demographics.parquet"

    for path in [mimic_path, eicu_path, split_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    mimic = pd.read_parquet(mimic_path)
    eicu = pd.read_parquet(eicu_path)
    splits = pd.read_csv(split_path)

    required_demo = {
        "patient_id", "stay_id", "age", "gender", "race",
        "icu_los_days", "hospital_expire_flag",
    }
    for name, df in [("mimic", mimic), ("eicu", eicu)]:
        missing = sorted(required_demo - set(df.columns))
        if missing:
            raise ValueError(f"{name} demographics missing columns: {missing}")
        df["patient_id"] = normalize_id(df["patient_id"])
        if df["patient_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate patient_id values detected")

    if not {"patient_id", "split"}.issubset(splits.columns):
        raise ValueError("full_cohort_split_assignments.csv must contain patient_id and split")
    splits["patient_id"] = normalize_id(splits["patient_id"])
    if splits["patient_id"].duplicated().any():
        raise ValueError("MIMIC split assignment contains duplicate patient_id values")

    mimic = mimic.merge(
        splits[["patient_id", "split"]],
        on="patient_id",
        how="left",
        validate="one_to_one",
    )
    if mimic["split"].isna().any():
        raise ValueError("Some MIMIC patients have no frozen Stage-05 split assignment")
    unexpected = sorted(set(mimic["split"].unique()) - {"train", "test"})
    if unexpected:
        raise ValueError(f"Unexpected MIMIC split labels: {unexpected}")

    mimic_dev = mimic.loc[mimic["split"].eq("train")].copy()
    mimic_test = mimic.loc[mimic["split"].eq("test")].copy()
    eicu_ext = eicu.copy()

    cohorts = {
        "MIMIC-IV development": mimic_dev,
        "MIMIC-IV test": mimic_test,
        "eICU-CRD external": eicu_ext,
    }

    for name, df in cohorts.items():
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df["icu_los_days"] = pd.to_numeric(df["icu_los_days"], errors="coerce")
        df["hospital_expire_flag"] = pd.to_numeric(df["hospital_expire_flag"], errors="coerce")
        df["sex_report"] = df["gender"].map(broad_sex)
        df["race_report"] = df["race"].map(broad_race)
        logger.info(
            "%s | n=%d age_missing=%d mortality_missing=%d",
            name,
            len(df),
            int(df["age"].isna().sum()),
            int(df["hospital_expire_flag"].isna().sum()),
        )

    return cohorts


def validate_counts(cohorts: Dict[str, pd.DataFrame], strict: bool, logger: logging.Logger) -> None:
    mismatches = []
    for name, expected in EXPECTED_COUNTS.items():
        observed = len(cohorts[name])
        if observed != expected:
            mismatches.append((name, observed, expected))
    if mismatches:
        msg = "; ".join(f"{n}: observed {o}, expected {e}" for n, o, e in mismatches)
        if strict:
            raise RuntimeError(f"Base-cohort count mismatch: {msg}")
        logger.warning("Base-cohort count mismatch: %s", msg)


def build_item20b_long(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []

    for cohort_name in COHORT_ORDER:
        df = cohorts[cohort_name]
        n = len(df)

        # Overall/sample size.
        rows.append({
            "cohort": cohort_name,
            "domain": "sample_size",
            "characteristic": "Patients",
            "category": "Overall",
            "n": n,
            "denominator": n,
            "percent": 100.0,
            "value": str(n),
        })

        # Continuous variables.
        for col, label, decimals in [
            ("age", "Age, years", 1),
            ("icu_los_days", "ICU LOS, days", 2),
        ]:
            stats = continuous_stats(df[col])
            rows.append({
                "cohort": cohort_name,
                "domain": "continuous",
                "characteristic": label,
                "category": "Median (IQR)",
                "n": int(stats["n_nonmissing"]),
                "denominator": n,
                "percent": 100.0 * stats["n_nonmissing"] / n if n else np.nan,
                "value": f"{stats['median']:.{decimals}f} ({stats['iqr']:.{decimals}f})" if stats["n_nonmissing"] else "NA",
                **{f"stat_{k}": v for k, v in stats.items()},
            })

        # Sex.
        for category in SEX_ORDER:
            count = int(df["sex_report"].eq(category).sum())
            rows.append({
                "cohort": cohort_name,
                "domain": "sex",
                "characteristic": "Sex",
                "category": category,
                "n": count,
                "denominator": n,
                "percent": 100.0 * count / n if n else np.nan,
                "value": fmt_count_pct(count, n),
            })

        # Broad race/ethnicity for reporting only.
        for category in RACE_ORDER:
            count = int(df["race_report"].eq(category).sum())
            rows.append({
                "cohort": cohort_name,
                "domain": "race_ethnicity",
                "characteristic": "Race/ethnicity",
                "category": category,
                "n": count,
                "denominator": n,
                "percent": 100.0 * count / n if n else np.nan,
                "value": fmt_count_pct(count, n),
            })

        # Mortality: prevalence among known plus missingness.
        known = df["hospital_expire_flag"].notna()
        known_n = int(known.sum())
        missing_n = int((~known).sum())
        deaths = int(df.loc[known, "hospital_expire_flag"].eq(1).sum())
        alive = known_n - deaths

        for category, count in [("Death", deaths), ("Alive", alive)]:
            rows.append({
                "cohort": cohort_name,
                "domain": "outcome",
                "characteristic": "In-hospital mortality",
                "category": category,
                "n": count,
                "denominator": known_n,
                "percent": 100.0 * count / known_n if known_n else np.nan,
                "value": fmt_count_pct(count, known_n),
            })
        rows.append({
            "cohort": cohort_name,
            "domain": "missingness",
            "characteristic": "In-hospital mortality status missing",
            "category": "Missing",
            "n": missing_n,
            "denominator": n,
            "percent": 100.0 * missing_n / n if n else np.nan,
            "value": fmt_count_pct(missing_n, n),
        })

        # Explicit missing/unknown reporting.
        missing_defs = [
            ("Age missing", int(df["age"].isna().sum())),
            ("Sex unknown", int(df["sex_report"].eq("Unknown").sum())),
            ("Race/ethnicity unknown", int(df["race_report"].eq("Unknown").sum())),
            ("ICU LOS missing", int(df["icu_los_days"].isna().sum())),
        ]
        for label, count in missing_defs:
            rows.append({
                "cohort": cohort_name,
                "domain": "missingness",
                "characteristic": label,
                "category": "Missing/unknown",
                "n": count,
                "denominator": n,
                "percent": 100.0 * count / n if n else np.nan,
                "value": fmt_count_pct(count, n),
            })

    return pd.DataFrame(rows)


def build_wide_table(long_df: pd.DataFrame) -> pd.DataFrame:
    display_order: List[Tuple[str, str, str]] = [
        ("sample_size", "Patients", "Overall"),
        ("continuous", "Age, years", "Median (IQR)"),
        ("sex", "Sex", "Female"),
        ("sex", "Sex", "Male"),
        ("sex", "Sex", "Unknown"),
    ]
    display_order += [("race_ethnicity", "Race/ethnicity", x) for x in RACE_ORDER]
    display_order += [
        ("outcome", "In-hospital mortality", "Death"),
        ("missingness", "In-hospital mortality status missing", "Missing"),
        ("continuous", "ICU LOS, days", "Median (IQR)"),
        ("missingness", "Age missing", "Missing/unknown"),
        ("missingness", "Sex unknown", "Missing/unknown"),
        ("missingness", "Race/ethnicity unknown", "Missing/unknown"),
        ("missingness", "ICU LOS missing", "Missing/unknown"),
    ]

    rows = []
    for domain, characteristic, category in display_order:
        subset = long_df[
            long_df["domain"].eq(domain)
            & long_df["characteristic"].eq(characteristic)
            & long_df["category"].eq(category)
        ]
        if subset.empty:
            continue
        label = characteristic
        if domain in {"sex", "race_ethnicity"}:
            label = f"{characteristic}: {category}"
        elif characteristic == "In-hospital mortality":
            label = "In-hospital mortality, n (%)"
        elif "missing" in characteristic.lower() or "unknown" in characteristic.lower():
            label = f"{characteristic}, n (%)"
        elif domain == "continuous":
            label = f"{characteristic}, median (IQR)"

        row = {"Characteristic": label}
        for cohort in COHORT_ORDER:
            hit = subset[subset["cohort"].eq(cohort)]
            row[cohort] = hit.iloc[0]["value"] if not hit.empty else "NA"
        rows.append(row)

    return pd.DataFrame(rows)


def build_race_distribution(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for cohort_name in COHORT_ORDER:
        df = cohorts[cohort_name]
        n = len(df)
        counts = df["race_report"].value_counts(dropna=False)
        for category in RACE_ORDER:
            count = int(counts.get(category, 0))
            rows.append({
                "cohort": cohort_name,
                "race_ethnicity_group": category,
                "n": count,
                "denominator": n,
                "percent": 100.0 * count / n if n else np.nan,
            })
    return pd.DataFrame(rows)


def build_item20c_smd(cohorts: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dev = cohorts["MIMIC-IV development"]
    comparisons = [
        ("MIMIC-IV test", cohorts["MIMIC-IV test"]),
        ("eICU-CRD external", cohorts["eICU-CRD external"]),
    ]
    rows = []

    for eval_name, ev in comparisons:
        # Continuous variables.
        for col, label in [("age", "Age"), ("icu_los_days", "ICU LOS")]:
            smd = smd_continuous(dev[col], ev[col])
            rows.append({
                "development_cohort": "MIMIC-IV development",
                "evaluation_cohort": eval_name,
                "variable": label,
                "level": "Continuous",
                "smd_eval_minus_development": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                "development_summary": fmt_median_iqr(dev[col], 2 if col == "icu_los_days" else 1),
                "evaluation_summary": fmt_median_iqr(ev[col], 2 if col == "icu_los_days" else 1),
            })

        # Sex categories as binary indicators.
        for category in SEX_ORDER:
            p_dev = float(dev["sex_report"].eq(category).mean())
            p_ev = float(ev["sex_report"].eq(category).mean())
            smd = smd_binary(p_dev, p_ev)
            rows.append({
                "development_cohort": "MIMIC-IV development",
                "evaluation_cohort": eval_name,
                "variable": "Sex",
                "level": category,
                "smd_eval_minus_development": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                "development_summary": f"{100*p_dev:.1f}%",
                "evaluation_summary": f"{100*p_ev:.1f}%",
            })

        # Race categories as binary indicators.
        for category in RACE_ORDER:
            p_dev = float(dev["race_report"].eq(category).mean())
            p_ev = float(ev["race_report"].eq(category).mean())
            smd = smd_binary(p_dev, p_ev)
            rows.append({
                "development_cohort": "MIMIC-IV development",
                "evaluation_cohort": eval_name,
                "variable": "Race/ethnicity",
                "level": category,
                "smd_eval_minus_development": smd,
                "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
                "development_summary": f"{100*p_dev:.1f}%",
                "evaluation_summary": f"{100*p_ev:.1f}%",
            })

        # Mortality among known outcomes only.
        dev_m = dev["hospital_expire_flag"].dropna().astype(float)
        ev_m = ev["hospital_expire_flag"].dropna().astype(float)
        p_dev = float(dev_m.mean()) if not dev_m.empty else np.nan
        p_ev = float(ev_m.mean()) if not ev_m.empty else np.nan
        smd = smd_binary(p_dev, p_ev)
        rows.append({
            "development_cohort": "MIMIC-IV development",
            "evaluation_cohort": eval_name,
            "variable": "In-hospital mortality",
            "level": "Death",
            "smd_eval_minus_development": smd,
            "abs_smd": abs(smd) if pd.notna(smd) else np.nan,
            "development_summary": f"{100*p_dev:.1f}%" if pd.notna(p_dev) else "NA",
            "evaluation_summary": f"{100*p_ev:.1f}%" if pd.notna(p_ev) else "NA",
        })

    return pd.DataFrame(rows)


def latex_escape(text: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def build_latex_table(wide: pd.DataFrame) -> str:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Base cohort characteristics reported in accordance with TRIPOD+AI participant-reporting recommendations. Continuous variables are median (IQR); categorical variables are $n$ (\%). Mortality percentages use patients with known hospital discharge status as the denominator. Race/ethnicity categories are reporting-only broad groups and do not alter the frozen model encoding.}",
        r"\label{tab:base_cohort_characteristics}",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"\textbf{Characteristic} & \textbf{MIMIC-IV development} & \textbf{MIMIC-IV test} & \textbf{eICU-CRD external} \\",
        r"\hline",
    ]
    for _, row in wide.iterrows():
        label = latex_escape(row["Characteristic"])
        values = [latex_escape(str(row[c])) for c in COHORT_ORDER]
        lines.append(f"{label} & {values[0]} & {values[1]} & {values[2]} \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize Age is fully observed in the retained cohorts. In MIMIC-IV, de-identified age at admission is reconstructed from anchor age/year and capped at 91 years in the extraction pipeline; in eICU-CRD, the de-identification category $>89$ is encoded as 90 years. Race/ethnicity is collapsed here only for cross-database descriptive reporting. SMD comparisons are provided separately and no hypothesis tests are used to assess baseline differences.",
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate TRIPOD+AI-aligned base cohort characteristics and development-evaluation comparisons."
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
        help=f"Tensor creation project root (default: {PROJECT_ROOT_DEFAULT})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <project-root>/data/05_splits/reports/tripod_ai)",
    )
    p.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="Warn rather than fail if base cohort counts differ from 2599/650/4943.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "data" / "05_splits" / "reports" / "tripod_ai"
    )
    logger = configure_logging(output_dir)
    logger.info("Project root: %s", project_root)
    logger.info("Output directory: %s", output_dir)

    cohorts = load_inputs(project_root, logger)
    validate_counts(cohorts, strict=not args.no_strict_counts, logger=logger)

    item20b_long = build_item20b_long(cohorts)
    item20b_wide = build_wide_table(item20b_long)
    race_dist = build_race_distribution(cohorts)
    item20c_smd = build_item20c_smd(cohorts)

    paths = {
        "item20b_long": output_dir / "tripod_ai_item20b_cohort_characteristics_long.csv",
        "item20b_wide": output_dir / "tripod_ai_item20b_cohort_characteristics_wide.csv",
        "race_distribution": output_dir / "tripod_ai_race_distribution.csv",
        "item20c_smd": output_dir / "tripod_ai_item20c_demographic_outcome_smd.csv",
        "latex_table": output_dir / "tripod_ai_cohort_table.tex",
        "manifest": output_dir / "tripod_ai_manifest.json",
    }

    item20b_long.to_csv(paths["item20b_long"], index=False)
    item20b_wide.to_csv(paths["item20b_wide"], index=False)
    race_dist.to_csv(paths["race_distribution"], index=False)
    item20c_smd.to_csv(paths["item20c_smd"], index=False)
    paths["latex_table"].write_text(build_latex_table(item20b_wide), encoding="utf-8")

    manifest = {
        "script": "05b_tripod_ai_cohort_characteristics.py",
        "purpose": "TRIPOD+AI-aligned descriptive cohort reporting; no model-development changes",
        "project_root": str(project_root),
        "inputs": {
            "mimic_harmonized_demographics": str(project_root / "data" / "03_harmonized" / "mimic" / "harmonized_demographics.parquet"),
            "eicu_harmonized_demographics": str(project_root / "data" / "03_harmonized" / "eicu" / "harmonized_demographics.parquet"),
            "frozen_mimic_split": str(project_root / "data" / "05_splits" / "reports" / "full_cohort_split_assignments.csv"),
        },
        "cohort_counts": {name: int(len(df)) for name, df in cohorts.items()},
        "mortality_missing": {
            name: int(df["hospital_expire_flag"].isna().sum())
            for name, df in cohorts.items()
        },
        "age_missing": {
            name: int(df["age"].isna().sum())
            for name, df in cohorts.items()
        },
        "reporting_race_groups": RACE_ORDER,
        "age_handling_note": {
            "MIMIC-IV": "age reconstructed upstream from anchor_age + (admission_year - anchor_year), capped at 91",
            "eICU-CRD": "raw >89 de-identification category encoded upstream as 90",
        },
        "item20c_note": "SMDs compare demographic variables, ICU LOS, and outcome distributions in each evaluation cohort with MIMIC-IV development; descriptive only, no p-values. Important clinical-predictor distributions should be reported separately rather than inferred by this cohort script.",
        "tripod_scope_note": "This script supports participant/cohort reporting for TRIPOD+AI Items 20b and the demographic/outcome component of 20c; it is not a complete TRIPOD+AI compliance checker.",
        "outputs": {k: str(v) for k, v in paths.items() if k != "manifest"},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %s", paths["item20b_long"])
    logger.info("Wrote %s", paths["item20b_wide"])
    logger.info("Wrote %s", paths["race_distribution"])
    logger.info("Wrote %s", paths["item20c_smd"])
    logger.info("Wrote %s", paths["latex_table"])
    logger.info("Wrote %s", paths["manifest"])

    logger.info("Completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
