#!/usr/bin/env python3
"""
06_bootstrap_landmark_analysis.py

Patient-level bootstrap uncertainty analysis for the dynamic-landmark
Paper_05_Tensor XGBoost pipeline.

The script DOES NOT refit models and DOES NOT rerun Optuna. It reads the
patient-level prediction parquet files already produced by 02_train_xgboost.py.

Outputs
-------
1. bootstrap_metric_ci.csv
   Percentile 95% CIs for each landmark/outcome/variant/cohort.

2. paired_representation_comparisons.csv
   Paired bootstrap differences between the reference representation and
   each comparator at the same landmark.

3. paired_landmark_common_riskset.csv
   Paired comparisons across landmarks on a common risk set. For LOS, each
   remaining-LOS prediction is converted to predicted total LOS before the
   cross-landmark comparison:
       predicted_total_LOS = landmark_hours / 24 + predicted_remaining_LOS

   This guarantees that all compared landmarks are evaluated against the
   same patient-level target in the same patients.

The prediction layout is expected to match 02_train_xgboost.py:

results/
  landmark_020h/
    los/
      full/
        predictions_mimic_test.parquet
        predictions_eicu_external.parquet
    mortality/
      full/
        predictions_mimic_test.parquet
        predictions_eicu_external.parquet

LOS columns:
    patient_id, stay_id, y_true, prediction

Mortality columns:
    patient_id, stay_id, y_true,
    prediction_raw, prediction_calibrated, frozen_threshold
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


DEFAULT_LANDMARKS = (1, 4, 8, 12, 16, 20, 24, 36, 48)
DEFAULT_VARIANTS = ("full", "o1", "o2", "o3", "o4", "static")
DEFAULT_COHORTS = ("mimic_test", "eicu_external")
DEFAULT_OUTCOMES = ("los", "mortality")
DEFAULT_COMMON_RISK_LANDMARKS = (16, 20, 24)

HIGHER_IS_BETTER = {
    "r2",
    "roc_auc",
    "average_precision",
}
LOWER_IS_BETTER = {
    "mse",
    "mae",
    "rmse",
    "brier",
    "logloss",
}
NEUTRAL_DIRECTION = {
    "bias",
}


def parse_csv_strings(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *[str(x) for x in parts]])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    # NumPy Generator accepts uint64, but constrain for portability.
    return value % (2**32 - 1)


def prediction_path(
    modeling_root: Path,
    landmark: int,
    outcome: str,
    variant: str,
    cohort: str,
) -> Path:
    return (
        modeling_root
        / "results"
        / f"landmark_{landmark:03d}h"
        / outcome
        / variant
        / f"predictions_{cohort}.parquet"
    )


def assert_unique_patient_rows(df: pd.DataFrame, label: str) -> None:
    required = {"patient_id", "stay_id", "y_true"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label}: missing columns: {missing}")

    if df["patient_id"].astype(str).duplicated().any():
        dup = (
            df.loc[df["patient_id"].astype(str).duplicated(keep=False), "patient_id"]
            .astype(str)
            .head(10)
            .tolist()
        )
        raise ValueError(
            f"{label}: expected one row per patient; duplicate patient_id values: {dup}"
        )

    if df[["patient_id", "stay_id"]].astype(str).duplicated().any():
        raise ValueError(f"{label}: duplicate patient_id + stay_id rows detected.")


def load_prediction_frame(
    path: Path,
    outcome: str,
    mortality_probability: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)
    label = str(path)
    assert_unique_patient_rows(df, label)

    out = df.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["stay_id"] = out["stay_id"].astype(str)
    out["y_true"] = pd.to_numeric(out["y_true"], errors="raise")

    if outcome == "los":
        if "prediction" not in out.columns:
            raise ValueError(f"{label}: LOS file is missing 'prediction'.")
        out["prediction"] = pd.to_numeric(out["prediction"], errors="raise")
        cols = ["patient_id", "stay_id", "y_true", "prediction"]
    elif outcome == "mortality":
        pred_col = (
            "prediction_calibrated"
            if mortality_probability == "calibrated"
            else "prediction_raw"
        )
        if pred_col not in out.columns:
            raise ValueError(f"{label}: mortality file is missing {pred_col!r}.")
        out["prediction"] = pd.to_numeric(out[pred_col], errors="raise")
        cols = ["patient_id", "stay_id", "y_true", "prediction"]
    else:
        raise ValueError(f"Unknown outcome: {outcome}")

    out = out[cols].copy()

    finite_mask = (
        np.isfinite(out["y_true"].to_numpy(dtype=float))
        & np.isfinite(out["prediction"].to_numpy(dtype=float))
    )
    if not finite_mask.all():
        bad_n = int((~finite_mask).sum())
        raise ValueError(f"{label}: {bad_n} non-finite y/prediction rows.")

    return out.sort_values(
        ["patient_id", "stay_id"],
        kind="stable",
    ).reset_index(drop=True)


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mse = float(mean_squared_error(y, p))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y, p)),
        # observed - predicted; positive = underprediction
        "bias": float(np.mean(y - p)),
    }


def classification_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-7, 1.0 - 1e-7)

    both_classes = np.unique(y).size == 2

    return {
        "roc_auc": (
            float(roc_auc_score(y, p))
            if both_classes
            else float("nan")
        ),
        "average_precision": (
            float(average_precision_score(y, p))
            if both_classes
            else float("nan")
        ),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
    }


def metric_function(outcome: str):
    if outcome == "los":
        return regression_metrics
    if outcome == "mortality":
        return classification_metrics
    raise ValueError(outcome)


def percentile_ci(
    values: Sequence[float],
    ci: float,
) -> Tuple[float, float, int]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), 0

    alpha = (100.0 - ci) / 2.0
    lo, hi = np.percentile(arr, [alpha, 100.0 - alpha])
    return float(lo), float(hi), int(arr.size)


def bootstrap_single_model(
    y: np.ndarray,
    p: np.ndarray,
    *,
    outcome: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> List[Dict[str, float]]:
    fn = metric_function(outcome)
    point = fn(y, p)
    n = len(y)
    rng = np.random.default_rng(seed)

    distributions: Dict[str, List[float]] = {
        metric: [] for metric in point
    }

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        values = fn(y[idx], p[idx])
        for metric, value in values.items():
            distributions[metric].append(value)

    rows = []
    for metric, estimate in point.items():
        lo, hi, valid = percentile_ci(distributions[metric], ci)
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "ci_low": lo,
                "ci_high": hi,
                "valid_bootstrap_replicates": valid,
            }
        )
    return rows


def align_two_frames(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    reference_label: str,
    comparator_label: str,
) -> pd.DataFrame:
    key = ["patient_id", "stay_id"]

    ref_keys = set(map(tuple, reference[key].to_numpy()))
    cmp_keys = set(map(tuple, comparator[key].to_numpy()))
    if ref_keys != cmp_keys:
        only_ref = list(ref_keys - cmp_keys)[:5]
        only_cmp = list(cmp_keys - ref_keys)[:5]
        raise ValueError(
            "Paired representation comparison requires identical patient sets. "
            f"{reference_label} n={len(ref_keys)}, "
            f"{comparator_label} n={len(cmp_keys)}, "
            f"only_reference={only_ref}, only_comparator={only_cmp}"
        )

    merged = reference.merge(
        comparator,
        on=key,
        how="inner",
        suffixes=("_reference", "_comparator"),
        validate="one_to_one",
    )

    y_ref = merged["y_true_reference"].to_numpy(dtype=float)
    y_cmp = merged["y_true_comparator"].to_numpy(dtype=float)

    if not np.allclose(y_ref, y_cmp, rtol=1e-6, atol=1e-7, equal_nan=True):
        max_abs = float(np.nanmax(np.abs(y_ref - y_cmp)))
        raise ValueError(
            f"Target mismatch between {reference_label} and {comparator_label}; "
            f"max_abs_difference={max_abs:.6g}"
        )

    return merged


def comparison_interpretation(
    metric: str,
    estimate: float,
    ci_low: float,
    ci_high: float,
    reference_name: str,
    comparator_name: str,
) -> Tuple[str, bool, str]:
    excludes_zero = bool(
        np.isfinite(ci_low)
        and np.isfinite(ci_high)
        and (ci_low > 0.0 or ci_high < 0.0)
    )

    if metric in HIGHER_IS_BETTER:
        favorable_sign = "positive"
        if ci_low > 0:
            conclusion = f"{reference_name}_better"
        elif ci_high < 0:
            conclusion = f"{comparator_name}_better"
        else:
            conclusion = "not_distinguishable"
    elif metric in LOWER_IS_BETTER:
        favorable_sign = "negative"
        if ci_high < 0:
            conclusion = f"{reference_name}_better"
        elif ci_low > 0:
            conclusion = f"{comparator_name}_better"
        else:
            conclusion = "not_distinguishable"
    else:
        favorable_sign = "closer_to_zero"
        conclusion = "direction_not_predefined"

    return favorable_sign, excludes_zero, conclusion


def bootstrap_paired_models(
    y: np.ndarray,
    p_reference: np.ndarray,
    p_comparator: np.ndarray,
    *,
    outcome: str,
    n_bootstrap: int,
    ci: float,
    seed: int,
    reference_name: str,
    comparator_name: str,
) -> List[Dict[str, object]]:
    fn = metric_function(outcome)
    point_ref = fn(y, p_reference)
    point_cmp = fn(y, p_comparator)

    common_metrics = [
        m for m in point_ref if m in point_cmp
    ]

    n = len(y)
    rng = np.random.default_rng(seed)
    distributions = {metric: [] for metric in common_metrics}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ref_values = fn(y[idx], p_reference[idx])
        cmp_values = fn(y[idx], p_comparator[idx])

        for metric in common_metrics:
            a = ref_values[metric]
            b = cmp_values[metric]
            distributions[metric].append(
                (a - b)
                if np.isfinite(a) and np.isfinite(b)
                else float("nan")
            )

    rows = []
    for metric in common_metrics:
        delta = float(point_ref[metric] - point_cmp[metric])
        lo, hi, valid = percentile_ci(distributions[metric], ci)
        favorable_sign, excludes_zero, conclusion = comparison_interpretation(
            metric,
            delta,
            lo,
            hi,
            reference_name,
            comparator_name,
        )
        rows.append(
            {
                "metric": metric,
                "reference_estimate": float(point_ref[metric]),
                "comparator_estimate": float(point_cmp[metric]),
                "delta_reference_minus_comparator": delta,
                "ci_low": lo,
                "ci_high": hi,
                "valid_bootstrap_replicates": valid,
                "favorable_delta_for_reference": favorable_sign,
                "ci_excludes_zero": excludes_zero,
                "conclusion": conclusion,
            }
        )
    return rows


def individual_worker(spec: Mapping[str, object]) -> List[Dict[str, object]]:
    modeling_root = Path(str(spec["modeling_root"]))
    landmark = int(spec["landmark"])
    outcome = str(spec["outcome"])
    variant = str(spec["variant"])
    cohort = str(spec["cohort"])
    mortality_probability = str(spec["mortality_probability"])

    path = prediction_path(
        modeling_root, landmark, outcome, variant, cohort
    )
    df = load_prediction_frame(
        path, outcome, mortality_probability
    )
    y = df["y_true"].to_numpy(
        dtype=int if outcome == "mortality" else float
    )
    p = df["prediction"].to_numpy(dtype=float)

    rows = bootstrap_single_model(
        y,
        p,
        outcome=outcome,
        n_bootstrap=int(spec["n_bootstrap"]),
        ci=float(spec["ci"]),
        seed=int(spec["seed"]),
    )

    for row in rows:
        row.update(
            {
                "landmark_hour": landmark,
                "outcome": outcome,
                "variant": variant,
                "cohort": cohort,
                "n": int(len(df)),
                "probability_source": (
                    mortality_probability
                    if outcome == "mortality"
                    else ""
                ),
                "prediction_file": str(path),
            }
        )
    return rows


def representation_worker(spec: Mapping[str, object]) -> List[Dict[str, object]]:
    modeling_root = Path(str(spec["modeling_root"]))
    landmark = int(spec["landmark"])
    outcome = str(spec["outcome"])
    reference_variant = str(spec["reference_variant"])
    comparator_variant = str(spec["comparator_variant"])
    cohort = str(spec["cohort"])
    mortality_probability = str(spec["mortality_probability"])

    ref_path = prediction_path(
        modeling_root,
        landmark,
        outcome,
        reference_variant,
        cohort,
    )
    cmp_path = prediction_path(
        modeling_root,
        landmark,
        outcome,
        comparator_variant,
        cohort,
    )

    ref = load_prediction_frame(
        ref_path, outcome, mortality_probability
    )
    cmp = load_prediction_frame(
        cmp_path, outcome, mortality_probability
    )
    merged = align_two_frames(
        ref,
        cmp,
        reference_label=reference_variant,
        comparator_label=comparator_variant,
    )

    y = merged["y_true_reference"].to_numpy(
        dtype=int if outcome == "mortality" else float
    )
    p_ref = merged["prediction_reference"].to_numpy(dtype=float)
    p_cmp = merged["prediction_comparator"].to_numpy(dtype=float)

    rows = bootstrap_paired_models(
        y,
        p_ref,
        p_cmp,
        outcome=outcome,
        n_bootstrap=int(spec["n_bootstrap"]),
        ci=float(spec["ci"]),
        seed=int(spec["seed"]),
        reference_name=reference_variant,
        comparator_name=comparator_variant,
    )

    for row in rows:
        row.update(
            {
                "landmark_hour": landmark,
                "outcome": outcome,
                "cohort": cohort,
                "n": int(len(merged)),
                "reference_variant": reference_variant,
                "comparator_variant": comparator_variant,
                "probability_source": (
                    mortality_probability
                    if outcome == "mortality"
                    else ""
                ),
                "reference_prediction_file": str(ref_path),
                "comparator_prediction_file": str(cmp_path),
            }
        )
    return rows


def build_common_risk_wide(
    *,
    modeling_root: Path,
    landmarks: Sequence[int],
    outcome: str,
    variant: str,
    cohort: str,
    mortality_probability: str,
) -> pd.DataFrame:
    key = ["patient_id", "stay_id"]
    merged: Optional[pd.DataFrame] = None

    for landmark in landmarks:
        path = prediction_path(
            modeling_root,
            landmark,
            outcome,
            variant,
            cohort,
        )
        frame = load_prediction_frame(
            path, outcome, mortality_probability
        )

        if outcome == "los":
            offset_days = float(landmark) / 24.0
            frame = frame.assign(
                **{
                    f"y_{landmark}": frame["y_true"] + offset_days,
                    f"p_{landmark}": frame["prediction"] + offset_days,
                }
            )[key + [f"y_{landmark}", f"p_{landmark}"]]
        else:
            frame = frame.rename(
                columns={
                    "y_true": f"y_{landmark}",
                    "prediction": f"p_{landmark}",
                }
            )[key + [f"y_{landmark}", f"p_{landmark}"]]

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(
                frame,
                on=key,
                how="inner",
                validate="one_to_one",
            )

    if merged is None or merged.empty:
        raise RuntimeError(
            f"No common-risk patients for {cohort}/{outcome}/{variant}/"
            f"landmarks={list(landmarks)}"
        )

    y_cols = [f"y_{landmark}" for landmark in landmarks]
    y_ref = merged[y_cols[0]].to_numpy(dtype=float)
    for col in y_cols[1:]:
        y_other = merged[col].to_numpy(dtype=float)
        if not np.allclose(
            y_ref, y_other, rtol=1e-5, atol=1e-5, equal_nan=True
        ):
            max_abs = float(np.nanmax(np.abs(y_ref - y_other)))
            raise ValueError(
                f"Common-risk target mismatch in {cohort}/{outcome}/{variant}: "
                f"{y_cols[0]} vs {col}, max_abs_difference={max_abs:.6g}"
            )

    if outcome == "los":
        max_landmark = max(landmarks)
        # y_ref is total LOS in days after conversion.
        if not np.all(y_ref * 24.0 > float(max_landmark) - 1e-6):
            raise ValueError(
                "Common-risk LOS set contains a patient whose total LOS "
                f"is not > max landmark ({max_landmark} h)."
            )

    return merged.sort_values(
        key,
        kind="stable",
    ).reset_index(drop=True)


def common_risk_worker(spec: Mapping[str, object]) -> List[Dict[str, object]]:
    modeling_root = Path(str(spec["modeling_root"]))
    landmarks = [int(x) for x in spec["landmarks"]]
    outcome = str(spec["outcome"])
    variant = str(spec["variant"])
    cohort = str(spec["cohort"])
    mortality_probability = str(spec["mortality_probability"])

    wide = build_common_risk_wide(
        modeling_root=modeling_root,
        landmarks=landmarks,
        outcome=outcome,
        variant=variant,
        cohort=cohort,
        mortality_probability=mortality_probability,
    )

    y = wide[f"y_{landmarks[0]}"].to_numpy(
        dtype=int if outcome == "mortality" else float
    )

    if str(spec["pair_mode"]) == "adjacent":
        pairs = list(zip(landmarks[:-1], landmarks[1:]))
    else:
        pairs = list(combinations(landmarks, 2))

    rows: List[Dict[str, object]] = []

    for earlier, later in pairs:
        p_earlier = wide[f"p_{earlier}"].to_numpy(dtype=float)
        p_later = wide[f"p_{later}"].to_numpy(dtype=float)

        # Define delta as later minus earlier, so the sign directly answers
        # whether waiting for the later landmark improved the metric.
        pair_rows = bootstrap_paired_models(
            y,
            p_later,
            p_earlier,
            outcome=outcome,
            n_bootstrap=int(spec["n_bootstrap"]),
            ci=float(spec["ci"]),
            seed=stable_seed(
                int(spec["seed"]),
                cohort,
                outcome,
                variant,
                "common-risk",
                earlier,
                later,
            ),
            reference_name=f"{later}h",
            comparator_name=f"{earlier}h",
        )

        for row in pair_rows:
            # Rename semantic columns for clarity in the output.
            row["later_estimate"] = row.pop("reference_estimate")
            row["earlier_estimate"] = row.pop("comparator_estimate")
            row["delta_later_minus_earlier"] = row.pop(
                "delta_reference_minus_comparator"
            )
            row["favorable_delta_for_later"] = row.pop(
                "favorable_delta_for_reference"
            )
            row.update(
                {
                    "common_riskset_max_landmark_hour": max(landmarks),
                    "common_riskset_landmarks": ",".join(map(str, landmarks)),
                    "earlier_landmark_hour": int(earlier),
                    "later_landmark_hour": int(later),
                    "outcome": outcome,
                    "variant": variant,
                    "cohort": cohort,
                    "n_common_riskset": int(len(wide)),
                    "target_scale": (
                        "total_los_days"
                        if outcome == "los"
                        else "in_hospital_mortality"
                    ),
                    "probability_source": (
                        mortality_probability
                        if outcome == "mortality"
                        else ""
                    ),
                }
            )

        rows.extend(pair_rows)

    return rows


def run_parallel(
    specs: Sequence[Mapping[str, object]],
    worker,
    workers: int,
    label: str,
) -> List[Dict[str, object]]:
    if not specs:
        return []

    print(f"[{label}] tasks={len(specs)} workers={workers}", flush=True)
    rows: List[Dict[str, object]] = []

    if workers <= 1:
        for i, spec in enumerate(specs, start=1):
            rows.extend(worker(spec))
            print(
                f"[{label}] completed {i}/{len(specs)}",
                flush=True,
            )
        return rows

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker, spec): spec
            for spec in specs
        }
        completed = 0
        for future in as_completed(futures):
            spec = futures[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"{label} task failed: {dict(spec)}"
                ) from exc
            completed += 1
            print(
                f"[{label}] completed {completed}/{len(specs)}",
                flush=True,
            )

    return rows


def available_prediction_specs(
    *,
    modeling_root: Path,
    landmarks: Sequence[int],
    outcomes: Sequence[str],
    variants: Sequence[str],
    cohorts: Sequence[str],
) -> List[Tuple[int, str, str, str, Path]]:
    specs = []
    for landmark in landmarks:
        for outcome in outcomes:
            for variant in variants:
                for cohort in cohorts:
                    path = prediction_path(
                        modeling_root,
                        landmark,
                        outcome,
                        variant,
                        cohort,
                    )
                    if path.is_file():
                        specs.append(
                            (landmark, outcome, variant, cohort, path)
                        )
    return specs


def sort_and_write(
    rows: Sequence[Mapping[str, object]],
    path: Path,
    sort_cols: Sequence[str],
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        present = [c for c in sort_cols if c in df.columns]
        df = df.sort_values(present, kind="stable").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} | rows={len(df)}", flush=True)
    return df


def make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Bootstrap CIs and paired dynamic-landmark comparisons using "
            "the patient-level predictions produced by 02_train_xgboost.py."
        )
    )
    p.add_argument(
        "--modeling-root",
        type=Path,
        default=Path("/home/ddimopoulos/Paper_05_Tensor/01_Modeling"),
    )
    p.add_argument(
        "--landmarks",
        default=",".join(map(str, DEFAULT_LANDMARKS)),
    )
    p.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
    )
    p.add_argument(
        "--outcomes",
        default=",".join(DEFAULT_OUTCOMES),
    )
    p.add_argument(
        "--cohorts",
        default=",".join(DEFAULT_COHORTS),
    )
    p.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of patient-level bootstrap replicates.",
    )
    p.add_argument(
        "--ci",
        type=float,
        default=95.0,
        help="Percentile confidence level.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel process workers. Use 1 for deterministic debugging.",
    )
    p.add_argument(
        "--mortality-probability",
        choices=("calibrated", "raw"),
        default="calibrated",
        help=(
            "Probability column used for mortality metrics. "
            "Default: calibrated (Platt fitted on MIMIC OOF only)."
        ),
    )
    p.add_argument(
        "--reference-variant",
        default="full",
        help=(
            "Reference representation for same-landmark paired comparisons. "
            "At 1 h, if 'full' is unavailable, o1 is automatically used."
        ),
    )
    p.add_argument(
        "--common-risk-landmarks",
        default=",".join(map(str, DEFAULT_COMMON_RISK_LANDMARKS)),
        help=(
            "Landmarks compared on one common risk set, e.g. 16,20,24. "
            "Leave empty to skip."
        ),
    )
    p.add_argument(
        "--common-risk-variant",
        default="full",
    )
    p.add_argument(
        "--common-risk-pairs",
        choices=("adjacent", "all"),
        default="adjacent",
        help="Compare adjacent landmarks or all pairwise landmark combinations.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--skip-individual-ci",
        action="store_true",
    )
    p.add_argument(
        "--skip-representation-comparisons",
        action="store_true",
    )
    p.add_argument(
        "--skip-common-risk",
        action="store_true",
    )
    return p


def main() -> int:
    args = make_arg_parser().parse_args()

    if args.bootstrap < 100:
        raise ValueError("--bootstrap should be at least 100.")
    if not (0.0 < args.ci < 100.0):
        raise ValueError("--ci must be between 0 and 100.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")

    modeling_root = args.modeling_root.resolve()
    if not modeling_root.is_dir():
        raise FileNotFoundError(modeling_root)

    landmarks = parse_csv_ints(args.landmarks)
    variants = parse_csv_strings(args.variants)
    outcomes = parse_csv_strings(args.outcomes)
    cohorts = parse_csv_strings(args.cohorts)

    invalid_outcomes = sorted(set(outcomes) - set(DEFAULT_OUTCOMES))
    if invalid_outcomes:
        raise ValueError(f"Invalid outcomes: {invalid_outcomes}")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else modeling_root / "bootstrap_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    available = available_prediction_specs(
        modeling_root=modeling_root,
        landmarks=landmarks,
        outcomes=outcomes,
        variants=variants,
        cohorts=cohorts,
    )
    if not available:
        raise RuntimeError(
            f"No prediction parquet files found under {modeling_root / 'results'}"
        )

    available_keys = {
        (landmark, outcome, variant, cohort)
        for landmark, outcome, variant, cohort, _ in available
    }

    # ------------------------------------------------------------------
    # 1) Individual model CIs
    # ------------------------------------------------------------------
    individual_rows: List[Dict[str, object]] = []
    if not args.skip_individual_ci:
        individual_specs = []
        for landmark, outcome, variant, cohort, _ in available:
            individual_specs.append(
                {
                    "modeling_root": str(modeling_root),
                    "landmark": landmark,
                    "outcome": outcome,
                    "variant": variant,
                    "cohort": cohort,
                    "mortality_probability": args.mortality_probability,
                    "n_bootstrap": args.bootstrap,
                    "ci": args.ci,
                    "seed": stable_seed(
                        args.seed,
                        "individual",
                        landmark,
                        outcome,
                        variant,
                        cohort,
                    ),
                }
            )

        individual_rows = run_parallel(
            individual_specs,
            individual_worker,
            args.workers,
            "individual-ci",
        )

    individual_df = sort_and_write(
        individual_rows,
        output_dir / "bootstrap_metric_ci.csv",
        [
            "outcome",
            "cohort",
            "landmark_hour",
            "variant",
            "metric",
        ],
    )

    # ------------------------------------------------------------------
    # 2) Same-landmark paired representation comparisons
    # ------------------------------------------------------------------
    representation_rows: List[Dict[str, object]] = []
    if not args.skip_representation_comparisons:
        representation_specs = []

        for landmark in landmarks:
            for outcome in outcomes:
                for cohort in cohorts:
                    # Full is the planned reference from 4 h onward.
                    # At 1 h, fall back to o1 because full is structurally
                    # unavailable in the current tensor design.
                    reference_variant = args.reference_variant
                    ref_key = (
                        landmark,
                        outcome,
                        reference_variant,
                        cohort,
                    )
                    if ref_key not in available_keys:
                        fallback_key = (
                            landmark,
                            outcome,
                            "o1",
                            cohort,
                        )
                        if fallback_key in available_keys:
                            reference_variant = "o1"
                        else:
                            continue

                    for comparator_variant in variants:
                        if comparator_variant == reference_variant:
                            continue

                        cmp_key = (
                            landmark,
                            outcome,
                            comparator_variant,
                            cohort,
                        )
                        if cmp_key not in available_keys:
                            continue

                        representation_specs.append(
                            {
                                "modeling_root": str(modeling_root),
                                "landmark": landmark,
                                "outcome": outcome,
                                "reference_variant": reference_variant,
                                "comparator_variant": comparator_variant,
                                "cohort": cohort,
                                "mortality_probability": args.mortality_probability,
                                "n_bootstrap": args.bootstrap,
                                "ci": args.ci,
                                "seed": stable_seed(
                                    args.seed,
                                    "representation",
                                    landmark,
                                    outcome,
                                    reference_variant,
                                    comparator_variant,
                                    cohort,
                                ),
                            }
                        )

        representation_rows = run_parallel(
            representation_specs,
            representation_worker,
            args.workers,
            "paired-representation",
        )

    representation_df = sort_and_write(
        representation_rows,
        output_dir / "paired_representation_comparisons.csv",
        [
            "outcome",
            "cohort",
            "landmark_hour",
            "reference_variant",
            "comparator_variant",
            "metric",
        ],
    )

    # ------------------------------------------------------------------
    # 3) Cross-landmark paired comparison on a common risk set
    # ------------------------------------------------------------------
    common_rows: List[Dict[str, object]] = []
    common_landmarks = (
        parse_csv_ints(args.common_risk_landmarks)
        if args.common_risk_landmarks.strip()
        else []
    )

    if (
        not args.skip_common_risk
        and len(common_landmarks) >= 2
    ):
        common_specs = []
        for outcome in outcomes:
            for cohort in cohorts:
                required = [
                    (
                        landmark,
                        outcome,
                        args.common_risk_variant,
                        cohort,
                    )
                    for landmark in common_landmarks
                ]
                if not all(key in available_keys for key in required):
                    print(
                        "[common-risk] skipping "
                        f"{cohort}/{outcome}/{args.common_risk_variant}: "
                        "one or more requested prediction files are missing.",
                        flush=True,
                    )
                    continue

                common_specs.append(
                    {
                        "modeling_root": str(modeling_root),
                        "landmarks": common_landmarks,
                        "outcome": outcome,
                        "variant": args.common_risk_variant,
                        "cohort": cohort,
                        "mortality_probability": args.mortality_probability,
                        "n_bootstrap": args.bootstrap,
                        "ci": args.ci,
                        "seed": stable_seed(
                            args.seed,
                            "common-risk",
                            outcome,
                            cohort,
                            args.common_risk_variant,
                            *common_landmarks,
                        ),
                        "pair_mode": args.common_risk_pairs,
                    }
                )

        common_rows = run_parallel(
            common_specs,
            common_risk_worker,
            min(args.workers, max(1, len(common_specs))),
            "common-risk",
        )

    common_df = sort_and_write(
        common_rows,
        output_dir / "paired_landmark_common_riskset.csv",
        [
            "outcome",
            "cohort",
            "variant",
            "earlier_landmark_hour",
            "later_landmark_hour",
            "metric",
        ],
    )

    manifest = {
        "script": Path(__file__).name,
        "created_unix_time": time.time(),
        "modeling_root": str(modeling_root),
        "output_dir": str(output_dir),
        "landmarks": landmarks,
        "variants": variants,
        "outcomes": outcomes,
        "cohorts": cohorts,
        "bootstrap_replicates": int(args.bootstrap),
        "ci_percent": float(args.ci),
        "seed": int(args.seed),
        "workers": int(args.workers),
        "mortality_probability": args.mortality_probability,
        "reference_variant": args.reference_variant,
        "common_risk_landmarks": common_landmarks,
        "common_risk_variant": args.common_risk_variant,
        "common_risk_pairs": args.common_risk_pairs,
        "available_prediction_files": len(available),
        "output_rows": {
            "bootstrap_metric_ci": int(len(individual_df)),
            "paired_representation_comparisons": int(len(representation_df)),
            "paired_landmark_common_riskset": int(len(common_df)),
        },
        "notes": [
            "Bootstrap unit is the patient because 02_train_xgboost.py produces one row per patient.",
            "Same-landmark representation comparisons use identical resampled patient indices.",
            "Common-risk landmark comparisons use the intersection of patients present at every requested landmark.",
            "For LOS common-risk comparisons, remaining LOS and its prediction are shifted by landmark/24 to total LOS, making the target identical across landmarks.",
            "No multiplicity-adjusted significance claims are made; CI-excluding-zero flags are estimation-based exploratory comparisons.",
        ],
        "elapsed_seconds": float(time.time() - start),
    }

    manifest_path = output_dir / "bootstrap_run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}", flush=True)

    print(
        f"Done in {manifest['elapsed_seconds']:.1f} s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
