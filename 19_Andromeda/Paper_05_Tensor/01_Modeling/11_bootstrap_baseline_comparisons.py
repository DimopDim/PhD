#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_bootstrap_baseline_comparisons.py

Patient-level bootstrap evaluation of the revised null/demographic baselines
against the main XGBoost Static and Full representations.

Positive paired effects always mean the XGBoost model performed better than
its baseline comparator. At 1 h, Full is structurally unavailable, so only
Static is compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor")
LANDMARKS_DEFAULT = (1, 4, 8, 12, 16, 20, 24, 36, 48)
OUTCOMES = ("los", "mortality")
BASELINES = ("null", "demographic")
MODELS = ("static", "full")
COHORTS = ("mimic_test", "eicu_external")
METRICS = {
    "los": ("mae", "rmse", "r2"),
    "mortality": ("roc_auc", "average_precision", "brier"),
}
HIGHER_IS_BETTER = {"r2", "roc_auc", "average_precision"}
LOWER_IS_BETTER = {"mae", "rmse", "brier"}


def parse_landmarks(text: str | None) -> List[int]:
    if text is None:
        return list(LANDMARKS_DEFAULT)
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    invalid = sorted(set(vals) - set(LANDMARKS_DEFAULT))
    if invalid:
        raise ValueError(f"Invalid landmarks={invalid}; allowed={list(LANDMARKS_DEFAULT)}")
    return vals


def parse_csv(text: str, allowed: Sequence[str], label: str) -> List[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    invalid = sorted(set(vals) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; allowed={list(allowed)}")
    if not vals:
        raise ValueError(f"No {label} selected.")
    return vals


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed)] + [str(x) for x in parts])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def percentile_ci(values: np.ndarray, ci: float) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    alpha = (100.0 - ci) / 2.0
    return float(np.percentile(values, alpha)), float(np.percentile(values, 100.0 - alpha))


def model_available(landmark: int, model: str) -> bool:
    return not (landmark == 1 and model == "full")


def main_prediction_path(root: Path, h: int, outcome: str, model: str, cohort: str) -> Path:
    return root / "results" / f"landmark_{h:03d}h" / outcome / model / f"predictions_{cohort}.parquet"


def baseline_prediction_path(root: Path, h: int, outcome: str, baseline: str, cohort: str) -> Path:
    return root / "baseline_results" / f"landmark_{h:03d}h" / outcome / baseline / f"predictions_{cohort}.parquet"


def load_canonical_summaries(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    training_path = root / "reports" / "training_task_summary.csv"
    baseline_path = root / "baseline_results" / "baseline_task_summary.csv"
    if not training_path.is_file():
        raise FileNotFoundError(training_path)
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    training = pd.read_csv(training_path)
    baseline = pd.read_csv(baseline_path)
    req_t = {"landmark_hour", "outcome", "variant", "status"}
    req_b = {"landmark_hour", "outcome", "baseline", "status"}
    if req_t - set(training.columns):
        raise ValueError(f"{training_path} missing columns: {sorted(req_t - set(training.columns))}")
    if req_b - set(baseline.columns):
        raise ValueError(f"{baseline_path} missing columns: {sorted(req_b - set(baseline.columns))}")
    if not training["status"].eq("PASS").all():
        raise ValueError("Main training summary contains non-PASS tasks.")
    if not baseline["status"].eq("PASS").all():
        raise ValueError("Baseline summary contains non-PASS tasks.")
    if training.duplicated(["landmark_hour", "outcome", "variant"]).any():
        raise ValueError("Duplicate main training tasks detected.")
    if baseline.duplicated(["landmark_hour", "outcome", "baseline"]).any():
        raise ValueError("Duplicate baseline tasks detected.")
    return training, baseline


def validate_task_coverage(training, baseline, landmarks, outcomes, baselines, models) -> None:
    training_keys = {(int(r.landmark_hour), str(r.outcome), str(r.variant)) for r in training.itertuples(index=False)}
    baseline_keys = {(int(r.landmark_hour), str(r.outcome), str(r.baseline)) for r in baseline.itertuples(index=False)}
    missing_main, missing_base = [], []
    for h in landmarks:
        for outcome in outcomes:
            for model in models:
                if model_available(h, model) and (h, outcome, model) not in training_keys:
                    missing_main.append((h, outcome, model))
            for base in baselines:
                if (h, outcome, base) not in baseline_keys:
                    missing_base.append((h, outcome, base))
    if missing_main:
        raise ValueError(f"Missing canonical main-model tasks: {missing_main}")
    if missing_base:
        raise ValueError(f"Missing canonical baseline tasks: {missing_base}")


def load_prediction(path: Path, outcome: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path).copy()
    required = {"patient_id", "stay_id", "y_true"}
    if required - set(df.columns):
        raise ValueError(f"{path} missing columns: {sorted(required - set(df.columns))}")
    df["patient_id"] = df["patient_id"].astype(str)
    df["stay_id"] = df["stay_id"].astype(str)
    if df["patient_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate patient_id rows.")
    if outcome == "los":
        if "prediction" not in df.columns:
            raise ValueError(f"{path}: missing prediction column.")
        out = df[["patient_id", "stay_id", "y_true", "prediction"]].copy()
        out[["y_true", "prediction"]] = out[["y_true", "prediction"]].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(out[["y_true", "prediction"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: non-finite LOS rows.")
        if (out["y_true"] <= 0).any():
            raise ValueError(f"{path}: remaining LOS target must be >0.")
        return out
    if "prediction_calibrated" not in df.columns:
        raise ValueError(f"{path}: missing prediction_calibrated.")
    out = df[["patient_id", "stay_id", "y_true", "prediction_calibrated"]].copy()
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["prediction_calibrated"] = pd.to_numeric(out["prediction_calibrated"], errors="coerce")
    known = np.isfinite(out["y_true"].to_numpy(dtype=float)) & np.isfinite(out["prediction_calibrated"].to_numpy(dtype=float))
    out = out.loc[known].copy()
    if out.empty:
        raise ValueError(f"{path}: no evaluable mortality rows.")
    labels = set(out["y_true"].astype(int).unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{path}: mortality labels not binary: {sorted(labels)}")
    if ((out["prediction_calibrated"] < 0) | (out["prediction_calibrated"] > 1)).any():
        raise ValueError(f"{path}: probabilities outside [0,1].")
    return out


def metric_value(outcome: str, metric: str, y: np.ndarray, p: np.ndarray) -> float:
    if outcome == "los":
        if metric == "mae":
            return float(mean_absolute_error(y, p))
        if metric == "rmse":
            return float(math.sqrt(mean_squared_error(y, p)))
        if metric == "r2":
            return float(r2_score(y, p))
        raise ValueError(metric)
    y = np.asarray(y, dtype=int)
    if metric in {"roc_auc", "average_precision"} and len(np.unique(y)) < 2:
        return float("nan")
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    if metric == "average_precision":
        return float(average_precision_score(y, p))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    raise ValueError(metric)


def oriented_difference(metric: str, model_value: float, baseline_value: float) -> float:
    if metric in HIGHER_IS_BETTER:
        return float(model_value - baseline_value)
    if metric in LOWER_IS_BETTER:
        return float(baseline_value - model_value)
    raise ValueError(metric)


def bootstrap_metric_worker(spec: dict) -> List[dict]:
    root = Path(spec["modeling_root"])
    h, outcome, baseline, cohort = int(spec["landmark"]), spec["outcome"], spec["baseline"], spec["cohort"]
    B, ci, seed = int(spec["n_bootstrap"]), float(spec["ci"]), int(spec["seed"])
    df = load_prediction(baseline_prediction_path(root, h, outcome, baseline, cohort), outcome)
    pred_col = "prediction" if outcome == "los" else "prediction_calibrated"
    y = df["y_true"].to_numpy(dtype=float)
    p = df[pred_col].to_numpy(dtype=float)
    n = len(df)
    rows = []
    for metric in METRICS[outcome]:
        rng = np.random.default_rng(stable_seed(seed, metric))
        estimate = metric_value(outcome, metric, y, p)
        boots = np.full(B, np.nan)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            boots[b] = metric_value(outcome, metric, y[idx], p[idx])
        finite = boots[np.isfinite(boots)]
        low, high = percentile_ci(finite, ci)
        rows.append({
            "landmark_hour": h, "outcome": outcome, "baseline": baseline, "cohort": cohort,
            "metric": metric, "n": n, "estimate": estimate, "ci_low": low, "ci_high": high,
            "bootstrap_replicates": B, "valid_bootstrap_replicates": int(len(finite)),
        })
    return rows


def paired_worker(spec: dict) -> List[dict]:
    root = Path(spec["modeling_root"])
    h, outcome, baseline, model, cohort = int(spec["landmark"]), spec["outcome"], spec["baseline"], spec["model"], spec["cohort"]
    B, ci, seed = int(spec["n_bootstrap"]), float(spec["ci"]), int(spec["seed"])
    bdf = load_prediction(baseline_prediction_path(root, h, outcome, baseline, cohort), outcome)
    mdf = load_prediction(main_prediction_path(root, h, outcome, model, cohort), outcome)
    pred_col = "prediction" if outcome == "los" else "prediction_calibrated"
    merged = bdf.merge(mdf, on=["patient_id", "stay_id"], how="inner", suffixes=("_baseline", "_model"), validate="one_to_one")
    if len(merged) != len(bdf) or len(merged) != len(mdf):
        raise ValueError(
            f"{h}h/{outcome}/{baseline}/{model}/{cohort}: patient-set mismatch "
            f"baseline={len(bdf)} model={len(mdf)} intersection={len(merged)}"
        )
    y_b = merged["y_true_baseline"].to_numpy(dtype=float)
    y_m = merged["y_true_model"].to_numpy(dtype=float)
    if not np.allclose(y_b, y_m, rtol=0.0, atol=1e-10, equal_nan=True):
        raise ValueError(f"{h}h/{outcome}/{baseline}/{model}/{cohort}: target mismatch.")
    y = y_b
    p_base = merged[f"{pred_col}_baseline"].to_numpy(dtype=float)
    p_model = merged[f"{pred_col}_model"].to_numpy(dtype=float)
    n = len(merged)
    rows = []
    for metric in METRICS[outcome]:
        base_est = metric_value(outcome, metric, y, p_base)
        model_est = metric_value(outcome, metric, y, p_model)
        effect = oriented_difference(metric, model_est, base_est)
        rng = np.random.default_rng(stable_seed(seed, metric))
        boots = np.full(B, np.nan)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            bb = metric_value(outcome, metric, y[idx], p_base[idx])
            mm = metric_value(outcome, metric, y[idx], p_model[idx])
            if np.isfinite(bb) and np.isfinite(mm):
                boots[b] = oriented_difference(metric, mm, bb)
        finite = boots[np.isfinite(boots)]
        low, high = percentile_ci(finite, ci)
        if np.isfinite(low) and np.isfinite(high):
            conclusion = "model_better" if low > 0 else ("baseline_better" if high < 0 else "indistinguishable")
        else:
            conclusion = "insufficient_bootstrap_information"
        rows.append({
            "landmark_hour": h, "outcome": outcome, "baseline": baseline, "model_variant": model,
            "cohort": cohort, "metric": metric, "n_paired": n,
            "baseline_estimate": base_est, "model_estimate": model_est,
            "oriented_model_minus_baseline": effect, "ci_low": low, "ci_high": high,
            "bootstrap_replicates": B, "valid_bootstrap_replicates": int(len(finite)),
            "ci_excludes_zero": bool(np.isfinite(low) and np.isfinite(high) and ((low > 0) or (high < 0))),
            "conclusion": conclusion,
        })
    return rows


def run_parallel(specs, worker, workers: int, label: str) -> List[dict]:
    print(f"[{label}] tasks={len(specs)} workers={workers}")
    rows = []
    if workers == 1:
        for i, spec in enumerate(specs, 1):
            rows.extend(worker(spec))
            print(f"[{label}] completed {i}/{len(specs)}")
        return rows
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(worker, spec): spec for spec in specs}
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result())
            print(f"[{label}] completed {i}/{len(specs)}")
    return rows


def write_sorted(df: pd.DataFrame, path: Path, sort_cols: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df = df.sort_values(list(sort_cols)).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"Wrote {path} | rows={len(df)}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bootstrap baseline CIs and paired baseline-vs-XGBoost comparisons.")
    p.add_argument("--project-root", type=Path, default=PROJECT_ROOT_DEFAULT)
    p.add_argument("--modeling-root", type=Path, default=None)
    p.add_argument("--landmarks", default=None)
    p.add_argument("--outcomes", default="los,mortality")
    p.add_argument("--baselines", default="null,demographic")
    p.add_argument("--models", default="static,full")
    p.add_argument("--cohorts", default="mimic_test,eicu_external")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--ci", type=float, default=95.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=8)
    return p


def main() -> int:
    args = build_parser().parse_args()
    t0 = time.time()
    project_root = args.project_root.expanduser().resolve()
    modeling_root = args.modeling_root.expanduser().resolve() if args.modeling_root is not None else project_root / "01_Modeling"
    if not modeling_root.is_dir():
        raise FileNotFoundError(modeling_root)
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be >=100.")
    if not (0 < args.ci < 100):
        raise ValueError("--ci must lie in (0,100).")
    if args.workers < 1:
        raise ValueError("--workers must be >=1.")

    landmarks = parse_landmarks(args.landmarks)
    outcomes = parse_csv(args.outcomes, OUTCOMES, "outcomes")
    baselines = parse_csv(args.baselines, BASELINES, "baselines")
    models = parse_csv(args.models, MODELS, "models")
    cohorts = parse_csv(args.cohorts, COHORTS, "cohorts")

    training_summary, baseline_summary = load_canonical_summaries(modeling_root)
    validate_task_coverage(training_summary, baseline_summary, landmarks, outcomes, baselines, models)

    output_dir = modeling_root / "baseline_results" / "bootstrap"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_specs = [
        {"modeling_root": str(modeling_root), "landmark": h, "outcome": outcome, "baseline": baseline,
         "cohort": cohort, "n_bootstrap": args.bootstrap, "ci": args.ci,
         "seed": stable_seed(args.seed, "baseline-ci", h, outcome, baseline, cohort)}
        for h in landmarks for outcome in outcomes for baseline in baselines for cohort in cohorts
    ]
    baseline_ci = pd.DataFrame(run_parallel(
        baseline_specs, bootstrap_metric_worker,
        min(args.workers, max(1, len(baseline_specs))), "baseline-ci"
    ))
    write_sorted(baseline_ci, output_dir / "baseline_metric_ci.csv",
                 ["landmark_hour", "outcome", "baseline", "cohort", "metric"])

    paired_specs = [
        {"modeling_root": str(modeling_root), "landmark": h, "outcome": outcome, "baseline": baseline,
         "model": model, "cohort": cohort, "n_bootstrap": args.bootstrap, "ci": args.ci,
         "seed": stable_seed(args.seed, "paired", h, outcome, baseline, model, cohort)}
        for h in landmarks for outcome in outcomes for baseline in baselines
        for model in models if model_available(h, model) for cohort in cohorts
    ]
    paired = pd.DataFrame(run_parallel(
        paired_specs, paired_worker,
        min(args.workers, max(1, len(paired_specs))), "paired-baseline-vs-model"
    ))
    write_sorted(paired, output_dir / "paired_baseline_vs_model.csv",
                 ["landmark_hour", "outcome", "baseline", "model_variant", "cohort", "metric"])

    expected_baseline_rows = sum(
        len(METRICS[o]) for _h in landmarks for o in outcomes for _b in baselines for _c in cohorts
    )
    expected_pair_tasks = sum(
        1 for h in landmarks for o in outcomes for _b in baselines
        for m in models if model_available(h, m) for _c in cohorts
    )
    expected_pair_rows = sum(
        len(METRICS[o]) for h in landmarks for o in outcomes for _b in baselines
        for m in models if model_available(h, m) for _c in cohorts
    )
    if len(baseline_ci) != expected_baseline_rows:
        raise RuntimeError(f"Baseline CI row mismatch: got={len(baseline_ci)} expected={expected_baseline_rows}")
    if len(paired_specs) != expected_pair_tasks:
        raise RuntimeError("Internal paired task-count mismatch.")
    if len(paired) != expected_pair_rows:
        raise RuntimeError(f"Paired row mismatch: got={len(paired)} expected={expected_pair_rows}")

    for label, df in (("baseline_metric_ci", baseline_ci), ("paired_baseline_vs_model", paired)):
        ratio = df["valid_bootstrap_replicates"].astype(float) / df["bootstrap_replicates"].astype(float)
        if (ratio < 0.95).any():
            raise RuntimeError(f"{label}: <95% valid bootstrap replicates for some rows.")

    counts = paired["conclusion"].value_counts(dropna=False).rename_axis("conclusion").reset_index(name="n_rows")
    counts.to_csv(output_dir / "paired_baseline_conclusion_counts.csv", index=False)

    manifest = {
        "script": Path(__file__).name,
        "modeling_root": str(modeling_root),
        "landmarks_hours": landmarks,
        "outcomes": outcomes,
        "baselines": baselines,
        "model_variants": models,
        "cohorts": cohorts,
        "bootstrap_replicates": args.bootstrap,
        "ci_percent": args.ci,
        "seed": args.seed,
        "mortality_probability_main_model": "prediction_calibrated",
        "mortality_probability_baseline": "direct baseline probability; no secondary Platt calibration",
        "effect_orientation": "positive means XGBoost better; model-baseline for higher-is-better metrics and baseline-model for lower-is-better metrics",
        "full_structurally_unavailable_at_1h": True,
        "expected_baseline_metric_rows": expected_baseline_rows,
        "expected_paired_tasks": expected_pair_tasks,
        "expected_paired_metric_rows": expected_pair_rows,
        "notes": [
            "Patient-level paired bootstrap; no model refitting.",
            "MIMIC test and eICU external only.",
            "Unknown eICU mortality outcomes are excluded from evaluable comparisons.",
            "CI-excluding-zero conclusions are descriptive; no multiplicity adjustment is applied.",
        ],
        "status": "PASS",
    }
    (output_dir / "11_bootstrap_baseline_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Expected paired tasks={expected_pair_tasks} | metric rows={expected_pair_rows}")
    print(f"Done in {time.time() - t0:.1f} s")
    print("PASS: Stage 11 bootstrap baseline comparisons complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
