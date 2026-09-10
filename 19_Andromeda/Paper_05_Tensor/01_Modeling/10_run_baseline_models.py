#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_run_baseline_models.py

Leakage-safe parsimonious baselines for the revised dynamic ICU landmark study.

Baselines
---------
1. null
   LOS:
       median remaining ICU LOS estimated from the MIMIC-IV development risk
       set at the current landmark.
   Mortality:
       event prevalence estimated from the MIMIC-IV development risk set at
       the current landmark.

2. demographic
   LOS:
       Ridge regression using only age + encoded gender.
   Mortality:
       L2-regularized logistic regression using only age + encoded gender.

Design constraints
------------------
- Baselines are refit independently at each landmark.
- Only the revised MIMIC-IV development split is used for fitting.
- MIMIC-IV internal test and eICU-CRD external data are prediction-only.
- No race predictors are used because database-specific race vocabularies are
  not directly harmonized.
- No clinical predictors are used in the demographic baseline.
- Fixed model hyperparameters are used; there is no Optuna/model selection.
- Development predictions are 5-fold out-of-fold (OOF).
- Test/external predictions come from a final model fit on the full MIMIC
  development risk set.
- Unknown eICU mortality outcomes are predicted but excluded from evaluable
  mortality metrics, matching the main modeling pipeline.

Inputs
------
Stage-01 cached model matrices under:
    <modeling-root>/data/model_matrices/landmark_XXXh/o1/

The o1 matrix is used only as a schema-safe carrier of age/gender and targets.
Clinical columns are explicitly excluded.

Outputs
-------
<modeling-root>/baseline_results/
    baseline_task_summary.csv
    baseline_run_manifest.json
    landmark_XXXh/
        los/
            null/
            demographic/
        mortality/
            null/
            demographic/

Each task directory contains:
    metrics.csv
    task_manifest.json
    predictions_mimic_train_oof.parquet
    predictions_mimic_test.parquet
    predictions_eicu_external.parquet
    [predictions_eicu_external_unknown_mortality.parquet]
    [model.joblib or baseline_parameters.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling_common import (
    load_model_matrix,
    matrix_cache_paths,
)



STAGE_PROTOCOL = "P5_BASELINES_FINAL_PROVENANCE_V1"
TASK_IDENTITY_VERSION = "P5_BASELINE_TASK_IDENTITY_V1"
RUN_IDENTITY_VERSION = "P5_BASELINE_RUN_IDENTITY_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def canonical_sha256(payload: dict) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def software_versions() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


PROJECT_ROOT_DEFAULT = Path("/home/ddimopoulos/Paper_05_Tensor")
MODELING_ROOT_DEFAULT = PROJECT_ROOT_DEFAULT / "01_Modeling"

LANDMARKS_DEFAULT = (1, 4, 8, 12, 16, 20, 24, 36, 48)
OUTCOMES = ("los", "mortality")
BASELINES = ("null", "demographic")
SOURCE_VARIANT = "o1"
N_FOLDS_DEFAULT = 5
SEED_DEFAULT = 42


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_landmarks(text: str | None) -> List[int]:
    if text is None:
        return list(LANDMARKS_DEFAULT)
    vals = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    invalid = sorted(set(vals) - set(LANDMARKS_DEFAULT))
    if invalid:
        raise ValueError(
            f"Invalid landmarks {invalid}; allowed={list(LANDMARKS_DEFAULT)}"
        )
    return vals


def parse_csv(text: str, allowed: Sequence[str], label: str) -> List[str]:
    vals = [x.strip() for x in text.split(",") if x.strip()]
    invalid = sorted(set(vals) - set(allowed))
    if invalid:
        raise ValueError(f"Invalid {label}: {invalid}; allowed={list(allowed)}")
    if not vals:
        raise ValueError(f"No {label} selected.")
    return vals


def safe_json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def task_dir(
    modeling_root: Path,
    landmark: int,
    outcome: str,
    baseline: str,
) -> Path:
    return (
        modeling_root
        / "baseline_results"
        / f"landmark_{landmark:03d}h"
        / outcome
        / baseline
    )


def load_split_matrix(
    modeling_root: Path,
    landmark: int,
    database: str,
    split_name: str,
):
    matrix_path, feature_path = matrix_cache_paths(
        modeling_root,
        landmark,
        SOURCE_VARIANT,
        database,
        split_name,
    )
    if not matrix_path.is_file():
        raise FileNotFoundError(matrix_path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    return load_model_matrix(matrix_path, feature_path)


def select_demographic_columns(feature_names: Sequence[str]) -> Tuple[List[int], List[str]]:
    """
    Select exactly age + gender one-hot columns from the o1 matrix.

    Race and all clinical/trajectory columns are intentionally excluded.
    """
    names = [str(x) for x in feature_names]
    lower = [x.lower() for x in names]

    age_idx = [i for i, x in enumerate(lower) if x == "age"]
    gender_idx = [i for i, x in enumerate(lower) if x.startswith("gender__")]

    if len(age_idx) != 1:
        raise ValueError(
            f"Expected exactly one age predictor, found "
            f"{[names[i] for i in age_idx]}"
        )
    if len(gender_idx) < 2:
        raise ValueError(
            f"Expected encoded gender columns, found "
            f"{[names[i] for i in gender_idx]}"
        )

    selected = age_idx + gender_idx
    selected_names = [names[i] for i in selected]

    forbidden = [
        x for x in selected_names
        if x.lower().startswith("race__")
        or "__last" in x.lower()
        or "__mean" in x.lower()
        or "__std" in x.lower()
        or "__slope" in x.lower()
    ]
    if forbidden:
        raise RuntimeError(
            f"Clinical/race columns entered demographic baseline: {forbidden}"
        )

    return selected, selected_names


def validate_matrix_alignment(train, test, external, landmark: int) -> None:
    if train.feature_names != test.feature_names:
        raise ValueError(f"{landmark}h: MIMIC train/test feature schema mismatch.")
    if train.feature_names != external.feature_names:
        raise ValueError(f"{landmark}h: MIMIC/eICU feature schema mismatch.")

    train_ids = set(train.patient_id.astype(str))
    test_ids = set(test.patient_id.astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(
            f"{landmark}h: MIMIC development/test patient overlap detected. "
            f"Examples={list(overlap)[:10]}"
        )

    for label, matrix in (
        ("mimic_train", train),
        ("mimic_test", test),
        ("eicu_external", external),
    ):
        if len(matrix.patient_id) != len(np.unique(matrix.patient_id.astype(str))):
            raise ValueError(f"{landmark}h/{label}: duplicate patient rows.")


def get_target(matrix, outcome: str) -> Tuple[np.ndarray, np.ndarray]:
    if outcome == "los":
        y = np.asarray(matrix.y_los_remaining_days, dtype=float)
        mask = np.isfinite(y) & (y > 0)
    elif outcome == "mortality":
        y = np.asarray(matrix.y_mortality, dtype=float)
        known = np.asarray(matrix.mortality_known, dtype=bool)
        mask = known & np.isfinite(y)
        if mask.any():
            labels = set(y[mask].astype(int).tolist())
            if not labels.issubset({0, 1}):
                raise ValueError(f"Non-binary mortality labels: {sorted(labels)}")
    else:
        raise ValueError(outcome)
    return y, mask


def regression_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mse = float(mean_squared_error(y, p))
    return {
        "mae": float(mean_absolute_error(y, p)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "r2": float(r2_score(y, p)),
        "bias": float(np.mean(p - y)),
    }


def classification_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("Non-finite mortality probabilities.")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Mortality probabilities outside [0,1].")

    if len(np.unique(y)) < 2:
        auc = float("nan")
        ap = float("nan")
    else:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))

    eps = 1e-7
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(
            log_loss(y, np.clip(p, eps, 1.0 - eps), labels=[0, 1])
        ),
    }


def calibration_slope_intercept(
    y: np.ndarray,
    p: np.ndarray,
) -> Tuple[float, float]:
    """
    Calibration intercept/slope from logistic regression of y on logit(p).

    Constant predictions (e.g. null prevalence baseline) do not identify a
    calibration slope, so NaN/NaN is returned.
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    if np.nanstd(p) < 1e-12:
        return float("nan"), float("nan")

    eps = 1e-6
    pc = np.clip(p, eps, 1.0 - eps)
    logit = np.log(pc / (1.0 - pc)).reshape(-1, 1)

    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(logit, y)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def build_demographic_model(outcome: str, seed: int) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if outcome == "los":
        estimator = Ridge(alpha=1.0)
    elif outcome == "mortality":
        estimator = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed,
        )
    else:
        raise ValueError(outcome)
    steps.append(("model", estimator))
    return Pipeline(steps)


def make_folds(y: np.ndarray, outcome: str, n_folds: int, seed: int):
    if outcome == "mortality":
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if counts.min() < n_folds:
            raise ValueError(
                f"Mortality class count too small for {n_folds} folds: "
                f"{counts.tolist()}"
            )
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(np.zeros(len(y)), y))

    splitter = KFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    return list(splitter.split(np.zeros(len(y))))


def predict_model(model: Pipeline, X: np.ndarray, outcome: str) -> np.ndarray:
    if outcome == "los":
        return model.predict(X).astype(np.float64)
    return model.predict_proba(X)[:, 1].astype(np.float64)


def fit_demographic_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_external: np.ndarray,
    *,
    outcome: str,
    n_folds: int,
    seed: int,
):
    folds = make_folds(y_train, outcome, n_folds, seed)
    oof = np.full(len(y_train), np.nan, dtype=np.float64)

    for fold, (fit_idx, val_idx) in enumerate(folds, start=1):
        model = build_demographic_model(outcome, seed + fold)
        model.fit(X_train[fit_idx], y_train[fit_idx])
        oof[val_idx] = predict_model(model, X_train[val_idx], outcome)

    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete/non-finite demographic OOF predictions.")

    final_model = build_demographic_model(outcome, seed)
    final_model.fit(X_train, y_train)
    test_pred = predict_model(final_model, X_test, outcome)
    external_pred = predict_model(final_model, X_external, outcome)

    return oof, test_pred, external_pred, final_model


def fit_null_predictions(
    y_train: np.ndarray,
    n_test: int,
    n_external: int,
    *,
    outcome: str,
    n_folds: int,
    seed: int,
):
    folds = make_folds(y_train, outcome, n_folds, seed)
    oof = np.full(len(y_train), np.nan, dtype=np.float64)

    for fit_idx, val_idx in folds:
        if outcome == "los":
            value = float(np.median(y_train[fit_idx]))
        else:
            value = float(np.mean(y_train[fit_idx]))
        oof[val_idx] = value

    if outcome == "los":
        final_value = float(np.median(y_train))
    else:
        final_value = float(np.mean(y_train))

    test_pred = np.full(n_test, final_value, dtype=np.float64)
    external_pred = np.full(n_external, final_value, dtype=np.float64)

    return oof, test_pred, external_pred, final_value


def make_prediction_table(
    *,
    patient_id: np.ndarray,
    stay_id: np.ndarray,
    y: np.ndarray,
    prediction: np.ndarray,
    outcome: str,
) -> pd.DataFrame:
    if outcome == "los":
        return pd.DataFrame(
            {
                "patient_id": patient_id.astype(str),
                "stay_id": stay_id.astype(str),
                "y_true": y.astype(float),
                "prediction": prediction.astype(float),
            }
        )

    return pd.DataFrame(
        {
            "patient_id": patient_id.astype(str),
            "stay_id": stay_id.astype(str),
            "y_true": y.astype(float),
            "prediction_raw": prediction.astype(float),
            # Baseline logistic/prevalence probabilities are already the
            # reported probabilities; no secondary Platt fit is applied.
            "prediction_calibrated": prediction.astype(float),
        }
    )


def write_metrics(
    *,
    task_output: Path,
    landmark: int,
    outcome: str,
    baseline: str,
    cohort_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: List[dict] = []

    for cohort, frame in cohort_frames.items():
        y = frame["y_true"].to_numpy(dtype=float)

        if outcome == "los":
            p = frame["prediction"].to_numpy(dtype=float)
            row = {
                "landmark_hour": landmark,
                "outcome": outcome,
                "baseline": baseline,
                "cohort": cohort,
                "n": int(len(frame)),
                **regression_metrics(y, p),
            }
        else:
            p = frame["prediction_calibrated"].to_numpy(dtype=float)
            y_int = y.astype(int)
            slope, intercept = calibration_slope_intercept(y_int, p)
            row = {
                "landmark_hour": landmark,
                "outcome": outcome,
                "baseline": baseline,
                "cohort": cohort,
                "n": int(len(frame)),
                "positives": int(np.sum(y_int == 1)),
                "prevalence": float(np.mean(y_int)),
                **classification_metrics(y_int, p),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            }
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(task_output / "metrics.csv", index=False)
    return out


def run_task(
    *,
    modeling_root: Path,
    landmark: int,
    outcome: str,
    baseline: str,
    n_folds: int,
    seed: int,
    overwrite: bool,
) -> dict:
    t0 = time.time()
    output = task_dir(modeling_root, landmark, outcome, baseline)
    output.mkdir(parents=True, exist_ok=True)

    manifest_path = output / "task_manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "PASS"
            and existing.get("protocol") == STAGE_PROTOCOL
            and existing.get("baseline_task_identity_sha256")
        ):
            return {
                "landmark_hour": landmark,
                "outcome": outcome,
                "baseline": baseline,
                "status": "PASS",
                "skipped_existing": True,
                "baseline_task_identity_sha256": existing[
                    "baseline_task_identity_sha256"
                ],
                "elapsed_seconds": 0.0,
            }
        raise RuntimeError(
            f"{manifest_path}: existing output is not FINAL provenance-bound; "
            "rerun with --overwrite."
        )

    if overwrite and output.exists():
        shutil.rmtree(output)
        output.mkdir(parents=True, exist_ok=True)

    train = load_split_matrix(modeling_root, landmark, "mimic", "train")
    test = load_split_matrix(modeling_root, landmark, "mimic", "test")
    external = load_split_matrix(modeling_root, landmark, "eicu", "external")

    validate_matrix_alignment(train, test, external, landmark)

    selected_idx, selected_names = select_demographic_columns(train.feature_names)

    train_y_all, train_mask = get_target(train, outcome)
    test_y_all, test_mask = get_target(test, outcome)
    ext_y_all, ext_mask = get_target(external, outcome)

    # LOS should be fully evaluable. Mortality may have unknown external labels.
    if outcome == "los":
        if not train_mask.all() or not test_mask.all() or not ext_mask.all():
            raise ValueError(f"{landmark}h: unexpected missing/invalid LOS target.")

    X_train = train.X[train_mask][:, selected_idx].astype(np.float64)
    X_test = test.X[test_mask][:, selected_idx].astype(np.float64)
    X_ext_eval = external.X[ext_mask][:, selected_idx].astype(np.float64)

    y_train = train_y_all[train_mask].astype(float)
    y_test = test_y_all[test_mask].astype(float)
    y_ext_eval = ext_y_all[ext_mask].astype(float)

    if baseline == "demographic":
        oof_pred, test_pred, ext_pred, final_model = fit_demographic_predictions(
            X_train,
            y_train,
            X_test,
            X_ext_eval,
            outcome=outcome,
            n_folds=n_folds,
            seed=seed,
        )
        joblib.dump(final_model, output / "model.joblib")
        model_meta = {
            "type": "ridge" if outcome == "los" else "logistic_regression_l2",
            "features": selected_names,
            "n_features": len(selected_names),
            "imputation": "median fit on MIMIC development data only",
            "scaling": "StandardScaler fit on MIMIC development data only",
            "hyperparameters": (
                {"alpha": 1.0}
                if outcome == "los"
                else {
                    "C": 1.0,
                    "penalty": "l2",
                    "solver": "lbfgs",
                    "max_iter": 2000,
                }
            ),
        }
    elif baseline == "null":
        oof_pred, test_pred, ext_pred, final_value = fit_null_predictions(
            y_train,
            len(y_test),
            len(y_ext_eval),
            outcome=outcome,
            n_folds=n_folds,
            seed=seed,
        )
        model_meta = {
            "type": "median_remaining_los" if outcome == "los" else "development_prevalence",
            "final_value_full_development": final_value,
            "oof_rule": "value estimated separately inside each training fold",
            "features": [],
            "n_features": 0,
        }
        safe_json_dump(model_meta, output / "baseline_parameters.json")
    else:
        raise ValueError(baseline)

    cohort_frames = {
        "mimic_train_oof": make_prediction_table(
            patient_id=train.patient_id[train_mask],
            stay_id=train.stay_id[train_mask],
            y=y_train,
            prediction=oof_pred,
            outcome=outcome,
        ),
        "mimic_test": make_prediction_table(
            patient_id=test.patient_id[test_mask],
            stay_id=test.stay_id[test_mask],
            y=y_test,
            prediction=test_pred,
            outcome=outcome,
        ),
        "eicu_external": make_prediction_table(
            patient_id=external.patient_id[ext_mask],
            stay_id=external.stay_id[ext_mask],
            y=y_ext_eval,
            prediction=ext_pred,
            outcome=outcome,
        ),
    }

    for cohort, frame in cohort_frames.items():
        frame.to_parquet(
            output / f"predictions_{cohort}.parquet",
            index=False,
        )

    unknown_external_n = 0
    if outcome == "mortality":
        unknown_mask = ~ext_mask
        unknown_external_n = int(np.sum(unknown_mask))
        if unknown_external_n:
            X_unknown = external.X[unknown_mask][:, selected_idx].astype(np.float64)
            if baseline == "demographic":
                unknown_pred = predict_model(final_model, X_unknown, outcome)
            else:
                unknown_pred = np.full(
                    unknown_external_n,
                    float(model_meta["final_value_full_development"]),
                    dtype=float,
                )
            unknown_frame = pd.DataFrame(
                {
                    "patient_id": external.patient_id[unknown_mask].astype(str),
                    "stay_id": external.stay_id[unknown_mask].astype(str),
                    "y_true": np.nan,
                    "prediction_raw": unknown_pred,
                    "prediction_calibrated": unknown_pred,
                }
            )
            unknown_frame.to_parquet(
                output / "predictions_eicu_external_unknown_mortality.parquet",
                index=False,
            )

    metrics_df = write_metrics(
        task_output=output,
        landmark=landmark,
        outcome=outcome,
        baseline=baseline,
        cohort_frames=cohort_frames,
    )

    # Exact provenance for the model-matrix carrier and the actual baseline inputs.
    split_objects = {
        "mimic/train": (train, train_mask, train_y_all),
        "mimic/test": (test, test_mask, test_y_all),
        "eicu/external": (external, ext_mask, ext_y_all),
    }
    input_bindings = {}
    for split_key, (matrix_obj, mask, target_all) in split_objects.items():
        db, split_name = split_key.split("/")
        matrix_path, feature_path = matrix_cache_paths(
            modeling_root,
            landmark,
            SOURCE_VARIANT,
            db,
            split_name,
        )
        x_selected = matrix_obj.X[mask][:, selected_idx].astype(np.float64)
        y_selected = np.asarray(target_all[mask], dtype=np.float64)
        input_bindings[split_key] = {
            "matrix_path": str(matrix_path),
            "matrix_sha256": sha256_file(matrix_path),
            "feature_file_path": str(feature_path),
            "feature_file_sha256": sha256_file(feature_path),
            "selected_X_sha256": sha256_array(x_selected),
            "target_sha256": sha256_array(y_selected),
            "patient_id_sha256": sha256_array(
                matrix_obj.patient_id[mask].astype(str)
            ),
            "stay_id_sha256": sha256_array(
                matrix_obj.stay_id[mask].astype(str)
            ),
        }

    output_files = {}
    for p in sorted(output.iterdir()):
        if (
            p.is_file()
            and p.name not in {"task_manifest.json", "baseline_task_identity.json"}
        ):
            output_files[p.name] = {
                "sha256": sha256_file(p),
                "bytes": int(p.stat().st_size),
            }

    source_hash = sha256_file(Path(__file__).resolve())
    identity_payload = {
        "identity_version": TASK_IDENTITY_VERSION,
        "protocol": STAGE_PROTOCOL,
        "landmark_hour": landmark,
        "outcome": outcome,
        "baseline": baseline,
        "source_variant": SOURCE_VARIANT,
        "n_folds": n_folds,
        "seed": seed,
        "selected_features": (
            selected_names if baseline == "demographic" else []
        ),
        "input_bindings": input_bindings,
        "output_files": output_files,
        "runner_source_sha256": source_hash,
        "software_versions": software_versions(),
    }
    task_identity = canonical_sha256(identity_payload)

    identity_document = dict(identity_payload)
    identity_document["baseline_task_identity_sha256"] = task_identity
    safe_json_dump(identity_document, output / "baseline_task_identity.json")

    manifest = {
        "script": Path(__file__).name,
        "protocol": STAGE_PROTOCOL,
        "identity_version": TASK_IDENTITY_VERSION,
        "baseline_task_identity_sha256": task_identity,
        "landmark_hour": landmark,
        "outcome": outcome,
        "baseline": baseline,
        "source_variant": SOURCE_VARIANT,
        "fit_source": "MIMIC-IV revised development risk set only",
        "evaluation_cohorts": ["mimic_test", "eicu_external"],
        "development_prediction_type": "5-fold out-of-fold",
        "n_folds": n_folds,
        "seed": seed,
        "selected_features": (
            selected_names if baseline == "demographic" else []
        ),
        "race_used": False,
        "clinical_predictors_used": False,
        "secondary_probability_calibration": False,
        "mortality_unknown_external_predictions_retained": True,
        "n_train_evaluable": int(train_mask.sum()),
        "n_test_evaluable": int(test_mask.sum()),
        "n_external_evaluable": int(ext_mask.sum()),
        "n_external_unknown_mortality": unknown_external_n,
        "model": model_meta,
        "input_bindings": input_bindings,
        "output_files": output_files,
        "runner_source_sha256": source_hash,
        "software_versions": software_versions(),
        "status": "PASS",
    }
    safe_json_dump(manifest, manifest_path)

    elapsed = time.time() - t0
    return {
        "landmark_hour": landmark,
        "outcome": outcome,
        "baseline": baseline,
        "status": "PASS",
        "skipped_existing": False,
        "n_train_evaluable": int(train_mask.sum()),
        "n_test_evaluable": int(test_mask.sum()),
        "n_external_evaluable": int(ext_mask.sum()),
        "n_external_unknown_mortality": unknown_external_n,
        "feature_count": 0 if baseline == "null" else len(selected_names),
        "elapsed_seconds": elapsed,
        "metrics_rows": int(len(metrics_df)),
        "baseline_task_identity_sha256": task_identity,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run leakage-safe null and demographic baseline models."
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--modeling-root",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--landmarks",
        type=str,
        default=None,
        help="Comma-separated subset; default is all revised landmarks.",
    )
    p.add_argument(
        "--outcomes",
        type=str,
        default="los,mortality",
    )
    p.add_argument(
        "--baselines",
        type=str,
        default="null,demographic",
    )
    p.add_argument(
        "--folds",
        type=int,
        default=N_FOLDS_DEFAULT,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    project_root = args.project_root.expanduser().resolve()
    modeling_root = (
        args.modeling_root.expanduser().resolve()
        if args.modeling_root is not None
        else project_root / "01_Modeling"
    )

    if not modeling_root.is_dir():
        raise FileNotFoundError(modeling_root)
    if args.folds < 2:
        raise ValueError("--folds must be >=2.")

    landmarks = parse_landmarks(args.landmarks)
    outcomes = parse_csv(args.outcomes, OUTCOMES, "outcomes")
    baselines = parse_csv(args.baselines, BASELINES, "baselines")

    print(f"Modeling root: {modeling_root}")
    print(f"Landmarks: {landmarks}")
    print(f"Outcomes: {outcomes}")
    print(f"Baselines: {baselines}")
    print(
        "Baseline design: null + age/gender only; "
        "MIMIC development fitting only; no race; no clinical features."
    )

    rows: List[dict] = []
    expected_tasks = len(landmarks) * len(outcomes) * len(baselines)
    completed = 0

    for landmark in landmarks:
        for outcome in outcomes:
            for baseline in baselines:
                print(f"[baseline] {landmark}h/{outcome}/{baseline}")
                row = run_task(
                    modeling_root=modeling_root,
                    landmark=landmark,
                    outcome=outcome,
                    baseline=baseline,
                    n_folds=args.folds,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
                rows.append(row)
                completed += 1
                print(
                    f"[baseline] completed {completed}/{expected_tasks} "
                    f"| {row['status']}"
                )

    summary = pd.DataFrame(rows).sort_values(
        ["landmark_hour", "outcome", "baseline"]
    )
    baseline_root = modeling_root / "baseline_results"
    baseline_root.mkdir(parents=True, exist_ok=True)
    summary_path = baseline_root / "baseline_task_summary.csv"
    summary.to_csv(summary_path, index=False)

    if len(summary) != expected_tasks:
        raise RuntimeError(
            f"Task-count mismatch: got {len(summary)}, expected {expected_tasks}."
        )
    if not summary["status"].eq("PASS").all():
        raise RuntimeError("One or more baseline tasks failed.")

    task_identities = (
        summary["baseline_task_identity_sha256"]
        .dropna()
        .astype(str)
        .tolist()
    )
    if len(task_identities) != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} task identities, got {len(task_identities)}."
        )
    if len(set(task_identities)) != expected_tasks:
        raise RuntimeError("Baseline task identities are not unique.")

    summary_hash = sha256_file(summary_path)
    source_hash = sha256_file(Path(__file__).resolve())
    run_identity_payload = {
        "identity_version": RUN_IDENTITY_VERSION,
        "protocol": STAGE_PROTOCOL,
        "task_identities": sorted(task_identities),
        "baseline_task_summary_sha256": summary_hash,
        "runner_source_sha256": source_hash,
        "software_versions": software_versions(),
    }
    run_identity = canonical_sha256(run_identity_payload)

    manifest = {
        "script": Path(__file__).name,
        "protocol": STAGE_PROTOCOL,
        "identity_version": RUN_IDENTITY_VERSION,
        "baseline_run_identity_sha256": run_identity,
        "modeling_root": str(modeling_root),
        "landmarks_hours": landmarks,
        "outcomes": outcomes,
        "baselines": baselines,
        "expected_tasks": expected_tasks,
        "completed_tasks": int(len(summary)),
        "folds": args.folds,
        "seed": args.seed,
        "task_identities": sorted(task_identities),
        "baseline_task_summary_sha256": summary_hash,
        "runner_source_sha256": source_hash,
        "software_versions": software_versions(),
        "design": {
            "null_los": "MIMIC development median remaining LOS at each landmark",
            "null_mortality": "MIMIC development mortality prevalence at each landmark",
            "demographic_predictors": ["age", "encoded gender"],
            "race_excluded": True,
            "clinical_predictors_excluded": True,
            "fit_database": "MIMIC-IV development only",
            "internal_test_role": "prediction/evaluation only",
            "eicu_role": "external prediction/evaluation only",
            "development_predictions": "5-fold OOF",
            "secondary_calibration": False,
        },
        "status": "PASS",
    }
    safe_json_dump(
        manifest,
        baseline_root / "baseline_run_manifest.json",
    )

    print(f"Wrote {summary_path} | rows={len(summary)}")
    print(f"Baseline run identity: {run_identity}")
    print(f"Unique baseline task identities: {len(set(task_identities))}")
    print("PASS: Stage 10 baseline models complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
