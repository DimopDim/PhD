#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_train_xgboost.py

Five-fold XGBoost training using frozen Optuna hyperparameters for dynamic-landmark LOS and mortality prediction.

Leakage-control design
----------------------
1. Training uses only MIMIC development data.
2. The 20% MIMIC internal test set is never used for:
   - fold construction
   - early stopping
   - model selection
   - calibration
   - threshold selection
3. eICU external data is never used for any fitting or tuning.
4. Each outer fold is untouched by early stopping:
   - outer training subset is split again into inner-fit / inner-ES
   - early stopping selects the number of boosting rounds
   - model is retrained on the entire outer-training subset using that round
     count
   - prediction is then made on the untouched outer validation fold
5. Test/external predictions are the mean of the five outer-fold models.
6. Mortality post-hoc sigmoid calibration and the classification threshold are
   fit/frozen using MIMIC OOF predictions only.

Default representations:
    full,o1,o2,o3,o4,static

Default outcomes:
    los,mortality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import shutil
import math
import os
import sys
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold,
    train_test_split,
)

from modeling_common import (
    VALID_OUTCOMES,
    VALID_VARIANTS,
    assert_no_identifier_predictors,
    assert_unique_patient_rows,
    load_json,
    load_model_matrix,
    matrix_cache_paths,
    parse_csv_arg,
    parse_landmarks,
    task_dir,
)


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)
MODELING_ROOT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "01_Modeling"
)

DEFAULT_PARAMS = {
    "eta": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "alpha": 0.0,
    "lambda": 1.0,
    "tree_method": "hist",
    "max_bin": 256,
}
DEFAULT_NUM_BOOST_ROUND = 4000
DEFAULT_EARLY_STOPPING = 200
DEFAULT_INNER_ES_FRACTION = 0.10
DEFAULT_TUNING_LANDMARK = 24


def sha256_json_payload(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_numpy_array(values: np.ndarray) -> str:
    arr = np.asarray(values)
    contiguous = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    header = {
        "dtype": contiguous.dtype.str,
        "shape": [int(x) for x in contiguous.shape],
        "order": "C",
    }
    h.update(
        json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    h.update(b"\n")
    h.update(memoryview(contiguous).cast("B"))
    return h.hexdigest()


def sha256_string_sequence(values) -> str:
    payload = [str(x) for x in values]
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def software_versions(xgb) -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": getattr(xgb, "__version__", "unknown"),
    }


def hyperparameter_file(
    modeling_root: Path,
    tuning_landmark: int,
    outcome: str,
    variant: str,
) -> Path:
    return (
        modeling_root
        / "hyperparameters"
        / f"landmark_{tuning_landmark:03d}h"
        / outcome
        / variant
        / "best_params.json"
    )


def load_task_params(
    modeling_root: Path,
    *,
    tuning_landmark: int,
    outcome: str,
    variant: str,
    params_source: str,
) -> Tuple[Dict, Optional[str], dict]:
    if params_source == "default":
        return dict(DEFAULT_PARAMS), None, {}

    path = hyperparameter_file(
        modeling_root,
        tuning_landmark,
        outcome,
        variant,
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen Optuna parameters: {path}. "
            "Run 02_optuna_tune_xgboost.py first, or explicitly use "
            "--params-source default."
        )

    payload = load_json(path)
    if int(payload.get("tuning_landmark_hour", -1)) != int(tuning_landmark):
        raise RuntimeError(f"Frozen parameter landmark mismatch in {path}.")
    if str(payload.get("outcome")) != str(outcome):
        raise RuntimeError(f"Frozen parameter outcome mismatch in {path}.")
    if str(payload.get("variant")) != str(variant):
        raise RuntimeError(f"Frozen parameter variant mismatch in {path}.")

    required = [
        "tuning_data_fingerprint_sha256",
        "patient_ids_sha256",
        "stay_ids_sha256",
        "feature_names_sha256",
        "x_values_sha256",
        "y_values_sha256",
        "matrix_file_sha256",
        "feature_names_file_sha256",
        "outer_fold_assignments_sha256",
        "tuning_config_sha256",
        "study_identity_sha256",
        "tuner_source_sha256",
        "modeling_common_source_sha256",
    ]
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise RuntimeError(
            f"Frozen parameters lack final provenance fields {missing}: {path}. "
            "Use the final provenance-aware 02_optuna_tune_xgboost.py outputs."
        )

    if int(payload.get("failed_trials", 0)) != 0:
        raise RuntimeError(
            f"Optuna study reports failed trials in {path}: "
            f"{payload.get('failed_trials')}"
        )
    target_trials = int(payload.get("target_total_trials", -1))
    total_trials = int(payload.get("total_trials_in_study", -1))
    if target_trials <= 0 or total_trials < target_trials:
        raise RuntimeError(
            f"Incomplete Optuna study in {path}: "
            f"total={total_trials}, target={target_trials}."
        )

    outer_fold_path = Path(
        payload.get("outer_fold_assignments_file", "")
    )
    if not outer_fold_path.is_file():
        raise FileNotFoundError(
            f"Missing Optuna outer-fold assignments: {outer_fold_path}"
        )
    actual_outer_hash = sha256_file(outer_fold_path)
    if actual_outer_hash != payload["outer_fold_assignments_sha256"]:
        raise RuntimeError(
            f"Optuna outer-fold assignment hash mismatch: {outer_fold_path}"
        )

    params = dict(payload["xgb_params"])
    for key in [
        "objective",
        "eval_metric",
        "nthread",
        "seed",
        "verbosity",
    ]:
        params.pop(key, None)

    provenance = {
        "hyperparameter_file_sha256": sha256_file(path),
        "tuning_data_fingerprint_sha256": payload["tuning_data_fingerprint_sha256"],
        "patient_ids_sha256": payload["patient_ids_sha256"],
        "stay_ids_sha256": payload["stay_ids_sha256"],
        "feature_names_sha256": payload["feature_names_sha256"],
        "x_values_sha256": payload["x_values_sha256"],
        "y_values_sha256": payload["y_values_sha256"],
        "matrix_file_sha256": payload["matrix_file_sha256"],
        "feature_names_file_sha256": payload["feature_names_file_sha256"],
        "outer_fold_assignments_file": str(outer_fold_path),
        "outer_fold_assignments_sha256": payload["outer_fold_assignments_sha256"],
        "tuning_config_sha256": payload["tuning_config_sha256"],
        "study_identity_sha256": payload["study_identity_sha256"],
        "study_identity_short": payload.get("study_identity_short"),
        "study_identity_version": payload.get("study_identity_version"),
        "tuner_source_sha256": payload["tuner_source_sha256"],
        "modeling_common_source_sha256": payload["modeling_common_source_sha256"],
        "target_total_trials": target_trials,
        "total_trials_in_study": total_trials,
        "complete_trials": int(payload.get("complete_trials", 0)),
        "pruned_trials": int(payload.get("pruned_trials", 0)),
        "failed_trials": int(payload.get("failed_trials", 0)),
        "software_versions": payload.get("software_versions", {}),
    }
    return params, str(path), provenance


def get_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required. Install/activate the project environment "
            "before running Stage 01 Modeling."
        ) from exc
    return xgb


def configure_logging(
    modeling_root: Path,
) -> logging.Logger:
    log_dir = (
        modeling_root
        / "logs"
    )
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "xgb_train"
    )
    logger.setLevel(
        logging.INFO
    )
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(
        sys.stdout
    )
    sh.setFormatter(
        formatter
    )
    logger.addHandler(
        sh
    )

    fh = logging.FileHandler(
        log_dir
        / "03_train_xgboost.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(
        formatter
    )
    logger.addHandler(
        fh
    )
    return logger


def load_cached(
    modeling_root: Path,
    landmark: int,
    variant: str,
    database: str,
    split_name: str,
):
    matrix_path, feature_path = (
        matrix_cache_paths(
            modeling_root,
            landmark,
            variant,
            database,
            split_name,
        )
    )
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Missing matrix cache: {matrix_path}. "
            "Run 01_prepare_model_matrices.py first."
        )
    return load_model_matrix(
        matrix_path,
        feature_path,
    )


def regression_metrics(
    y: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    mse = mean_squared_error(
        y,
        pred,
    )
    return {
        "mse": float(mse),
        "mae": float(
            mean_absolute_error(
                y,
                pred,
            )
        ),
        "rmse": float(
            math.sqrt(mse)
        ),
        "r2": float(
            r2_score(
                y,
                pred,
            )
        ),
    }


def safe_classification_metrics(
    y: np.ndarray,
    pred: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(y)
    pred = np.asarray(pred)

    if len(np.unique(y)) < 2:
        return {
            "roc_auc": np.nan,
            "average_precision": np.nan,
            "brier": np.nan,
            "logloss": np.nan,
        }

    eps = np.finfo(
        np.float64
    ).eps
    clipped = np.clip(
        pred,
        eps,
        1.0 - eps,
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y,
                pred,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y,
                pred,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                pred,
            )
        ),
        "logloss": float(
            log_loss(
                y,
                clipped,
                labels=[0, 1],
            )
        ),
    }


def calibration_slope_intercept(
    y: np.ndarray,
    pred: np.ndarray,
) -> Tuple[float, float]:
    eps = 1e-6
    p = np.clip(
        pred,
        eps,
        1.0 - eps,
    )
    logit = np.log(
        p / (1.0 - p)
    ).reshape(-1, 1)

    if len(np.unique(y)) < 2:
        return np.nan, np.nan

    # Very weak L2 regularization is used for broad scikit-learn
    # compatibility while remaining effectively unpenalized.
    model = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
    )
    model.fit(
        logit,
        y,
    )

    return (
        float(
            model.coef_[0, 0]
        ),
        float(
            model.intercept_[0]
        ),
    )


def youden_threshold(
    y: np.ndarray,
    pred: np.ndarray,
) -> float:
    if len(np.unique(y)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(
        y,
        pred,
    )
    j = tpr - fpr

    finite = np.isfinite(
        thresholds
    )
    if not finite.any():
        return 0.5

    idx_local = np.argmax(
        j[finite]
    )
    threshold = thresholds[
        np.flatnonzero(
            finite
        )[idx_local]
    ]

    return float(
        np.clip(
            threshold,
            0.0,
            1.0,
        )
    )


def threshold_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    yhat = (
        pred >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y == 1)
            & (yhat == 1)
        )
    )
    tn = int(
        np.sum(
            (y == 0)
            & (yhat == 0)
        )
    )
    fp = int(
        np.sum(
            (y == 0)
            & (yhat == 1)
        )
    )
    fn = int(
        np.sum(
            (y == 1)
            & (yhat == 0)
        )
    )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn
        else np.nan
    )
    specificity = (
        tn / (tn + fp)
        if tn + fp
        else np.nan
    )
    precision = (
        tp / (tp + fp)
        if tp + fp
        else np.nan
    )
    f1 = (
        2 * precision * sensitivity
        / (precision + sensitivity)
        if (
            np.isfinite(precision)
            and np.isfinite(sensitivity)
            and precision + sensitivity > 0
        )
        else np.nan
    )

    return {
        "threshold": float(
            threshold
        ),
        "sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "precision": float(
            precision
        ),
        "f1": float(
            f1
        ),
    }


def make_outer_splits(
    y: np.ndarray,
    groups: np.ndarray,
    outcome: str,
    seed: int,
):
    """
    Explicit patient-level five-fold CV.

    patient_id is used only as a grouping key. It is never passed to XGBoost
    as a predictor.
    """
    groups = np.asarray(groups).astype(str)

    if outcome == "mortality":
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        splits = list(
            splitter.split(
                np.zeros(len(y)),
                y,
                groups=groups,
            )
        )
    else:
        splitter = GroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed,
        )
        splits = list(
            splitter.split(
                np.zeros(len(y)),
                y,
                groups=groups,
            )
        )

    for fold, (train_idx, valid_idx) in enumerate(
        splits,
        start=1,
    ):
        overlap = set(groups[train_idx]) & set(groups[valid_idx])
        if overlap:
            raise RuntimeError(
                f"Patient leakage in outer CV fold {fold}: "
                f"{list(overlap)[:10]}"
            )

    return splits


def inner_split(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outcome: str,
    seed: int,
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Patient-level inner split used only for early stopping.
    """
    local = np.arange(len(indices))

    if len(
        np.unique(
            groups[indices].astype(str)
        )
    ) != len(indices):
        raise RuntimeError(
            "Duplicate patient groups detected before inner split."
        )

    if outcome == "mortality":
        local_fit, local_es = train_test_split(
            local,
            test_size=fraction,
            random_state=seed,
            shuffle=True,
            stratify=y[indices],
        )
    else:
        local_fit, local_es = train_test_split(
            local,
            test_size=fraction,
            random_state=seed,
            shuffle=True,
        )

    fit_idx = indices[local_fit]
    es_idx = indices[local_es]

    overlap = (
        set(groups[fit_idx].astype(str))
        & set(groups[es_idx].astype(str))
    )
    if overlap:
        raise RuntimeError(
            f"Patient leakage in inner early-stopping split: "
            f"{list(overlap)[:10]}"
        )

    return fit_idx, es_idx


def fit_fold(
    *,
    xgb,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outer_train_idx: np.ndarray,
    outer_valid_idx: np.ndarray,
    outcome: str,
    seed: int,
    fold: int,
    feature_names: Sequence[str],
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    inner_es_fraction: float,
    xgb_threads: int,
    model_path: Path,
):
    inner_fit_idx, inner_es_idx = (
        inner_split(
            outer_train_idx,
            y,
            groups,
            outcome,
            seed + 1000 + fold,
            inner_es_fraction,
        )
    )

    fold_params = dict(
        params
    )
    fold_params.update(
        {
            "seed": int(
                seed + fold
            ),
            "nthread": int(
                xgb_threads
            ),
            "verbosity": 0,
        }
    )

    if outcome == "los":
        fold_params[
            "objective"
        ] = "reg:squarederror"
        fold_params[
            "eval_metric"
        ] = "rmse"
    else:
        fold_params[
            "objective"
        ] = "binary:logistic"
        fold_params[
            "eval_metric"
        ] = "logloss"

    dfit = xgb.DMatrix(
        X[inner_fit_idx],
        label=y[inner_fit_idx],
        feature_names=list(
            feature_names
        ),
        missing=np.nan,
    )
    des = xgb.DMatrix(
        X[inner_es_idx],
        label=y[inner_es_idx],
        feature_names=list(
            feature_names
        ),
        missing=np.nan,
    )

    provisional = xgb.train(
        fold_params,
        dfit,
        num_boost_round=num_boost_round,
        evals=[(des, "inner_es")],
        early_stopping_rounds=(
            early_stopping_rounds
        ),
        verbose_eval=False,
    )

    best_iteration = getattr(
        provisional,
        "best_iteration",
        None,
    )
    if best_iteration is None:
        best_rounds = (
            num_boost_round
        )
    else:
        best_rounds = int(
            best_iteration
        ) + 1

    # Retrain on the entire outer-training subset with the frozen round count.
    douter = xgb.DMatrix(
        X[outer_train_idx],
        label=y[outer_train_idx],
        feature_names=list(
            feature_names
        ),
        missing=np.nan,
    )

    model = xgb.train(
        fold_params,
        douter,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model.save_model(
        model_path
    )

    dvalid = xgb.DMatrix(
        X[outer_valid_idx],
        feature_names=list(
            feature_names
        ),
        missing=np.nan,
    )
    pred_valid = model.predict(
        dvalid
    )

    return (
        model,
        pred_valid,
        best_rounds,
        len(inner_fit_idx),
        len(inner_es_idx),
    )


def run_task(task: dict) -> dict:
    xgb = get_xgboost()

    modeling_root = Path(
        task["modeling_root"]
    )
    landmark = int(
        task["landmark"]
    )
    variant = task[
        "variant"
    ]
    outcome = task[
        "outcome"
    ]
    seed = int(
        task["seed"]
    )

    train = load_cached(
        modeling_root,
        landmark,
        variant,
        "mimic",
        "train",
    )
    test = load_cached(
        modeling_root,
        landmark,
        variant,
        "mimic",
        "test",
    )
    external = load_cached(
        modeling_root,
        landmark,
        variant,
        "eicu",
        "external",
    )

    if (
        train.feature_names
        != test.feature_names
        or train.feature_names
        != external.feature_names
    ):
        raise RuntimeError(
            f"Feature schema mismatch in task {landmark}/{outcome}/{variant}"
        )

    if outcome == "los":
        y_train = (
            train.y_los_remaining_days
        ).astype(np.float32)
        y_test = (
            test.y_los_remaining_days
        ).astype(np.float32)
        y_external = (
            external.y_los_remaining_days
        ).astype(np.float32)

        train_mask = np.ones(
            len(y_train),
            dtype=bool,
        )
        test_mask = np.ones(
            len(y_test),
            dtype=bool,
        )
        external_mask = np.ones(
            len(y_external),
            dtype=bool,
        )
    else:
        train_mask = (
            train.mortality_known
        )
        test_mask = (
            test.mortality_known
        )
        external_mask = (
            external.mortality_known
        )

        if not train_mask.all():
            raise RuntimeError(
                "MIMIC training mortality is expected to be fully known."
            )

        y_train = (
            train.y_mortality[
                train_mask
            ]
        ).astype(np.int8)
        y_test = (
            test.y_mortality[
                test_mask
            ]
        ).astype(np.int8)
        y_external = (
            external.y_mortality[
                external_mask
            ]
        ).astype(np.int8)

    X_train = train.X[
        train_mask
    ]
    X_test = test.X[
        test_mask
    ]
    X_external = external.X[
        external_mask
    ]

    groups_train = train.patient_id[
        train_mask
    ].astype(str)

    # Bind this training task to the exact matrix/target bytes that were
    # loaded. At the 24 h Optuna anchor, these hashes must match the final
    # tuning study before frozen hyperparameters are accepted.
    train_matrix_path, train_feature_path = matrix_cache_paths(
        modeling_root, landmark, variant, "mimic", "train"
    )
    test_matrix_path, test_feature_path = matrix_cache_paths(
        modeling_root, landmark, variant, "mimic", "test"
    )
    external_matrix_path, external_feature_path = matrix_cache_paths(
        modeling_root, landmark, variant, "eicu", "external"
    )

    input_hashes = {
        "train_matrix_file_sha256": sha256_file(train_matrix_path),
        "test_matrix_file_sha256": sha256_file(test_matrix_path),
        "external_matrix_file_sha256": sha256_file(external_matrix_path),
        "train_feature_file_sha256": sha256_file(train_feature_path),
        "test_feature_file_sha256": sha256_file(test_feature_path),
        "external_feature_file_sha256": sha256_file(external_feature_path),
        "feature_names_sha256": sha256_string_sequence(train.feature_names),
        "x_train_values_sha256": sha256_numpy_array(X_train),
        "y_train_values_sha256": sha256_numpy_array(y_train),
        "x_test_values_sha256": sha256_numpy_array(X_test),
        "y_test_values_sha256": sha256_numpy_array(y_test),
        "x_external_values_sha256": sha256_numpy_array(X_external),
        "y_external_values_sha256": sha256_numpy_array(y_external),
        "train_patient_ids_sha256": sha256_string_sequence(
            train.patient_id[train_mask]
        ),
        "train_stay_ids_sha256": sha256_string_sequence(
            train.stay_id[train_mask]
        ),
        "test_patient_ids_sha256": sha256_string_sequence(
            test.patient_id[test_mask]
        ),
        "test_stay_ids_sha256": sha256_string_sequence(
            test.stay_id[test_mask]
        ),
        "external_patient_ids_sha256": sha256_string_sequence(
            external.patient_id[external_mask]
        ),
        "external_stay_ids_sha256": sha256_string_sequence(
            external.stay_id[external_mask]
        ),
    }

    hp_provenance = task.get("hyperparameter_provenance", {})
    anchor_tuning_input_match = None
    if task["params_source"] == "optuna" and landmark == int(task["tuning_landmark"]):
        anchor_checks = {
            "x_values_sha256": input_hashes["x_train_values_sha256"],
            "y_values_sha256": input_hashes["y_train_values_sha256"],
            "patient_ids_sha256": input_hashes["train_patient_ids_sha256"],
            "stay_ids_sha256": input_hashes["train_stay_ids_sha256"],
            "feature_names_sha256": input_hashes["feature_names_sha256"],
            "matrix_file_sha256": input_hashes["train_matrix_file_sha256"],
            "feature_names_file_sha256": input_hashes["train_feature_file_sha256"],
        }
        mismatches = {
            key: {"training": value, "tuning": hp_provenance.get(key)}
            for key, value in anchor_checks.items()
            if value != hp_provenance.get(key)
        }
        if mismatches:
            raise RuntimeError(
                "Final Optuna provenance does not match the actual 24 h "
                f"training inputs for {outcome}/{variant}: {mismatches}"
            )
        anchor_tuning_input_match = True

    assert_unique_patient_rows(
        groups_train,
        label=(
            f"training {landmark}h/{outcome}/{variant}/mimic_train"
        ),
    )
    assert_unique_patient_rows(
        test.patient_id[
            test_mask
        ],
        label=(
            f"training {landmark}h/{outcome}/{variant}/mimic_test"
        ),
    )
    assert_no_identifier_predictors(
        train.feature_names
    )

    # Internal train/test patient sets must remain disjoint.
    train_test_overlap = (
        set(
            train.patient_id[
                train_mask
            ].astype(str)
        )
        & set(
            test.patient_id[
                test_mask
            ].astype(str)
        )
    )
    if train_test_overlap:
        raise RuntimeError(
            f"MIMIC development/test patient leakage detected: "
            f"{list(train_test_overlap)[:10]}"
        )

    outer_splits = make_outer_splits(
        y_train,
        groups_train,
        outcome,
        seed,
    )

    outer_fold_assignment = np.full(len(groups_train), -1, dtype=np.int16)
    for fold, (_, valid_idx) in enumerate(outer_splits, start=1):
        if np.any(outer_fold_assignment[valid_idx] != -1):
            raise RuntimeError("Duplicate outer validation-fold assignment.")
        outer_fold_assignment[valid_idx] = fold
    if np.any(outer_fold_assignment == -1):
        raise RuntimeError("Incomplete outer validation-fold assignment.")

    anchor_tuning_fold_match = None
    if task["params_source"] == "optuna" and landmark == int(task["tuning_landmark"]):
        tuning_fold_path = Path(hp_provenance["outer_fold_assignments_file"])
        tuning_folds = pd.read_csv(tuning_fold_path)
        expected = pd.DataFrame({
            "patient_id": groups_train.astype(str),
            "fold": outer_fold_assignment.astype(int),
        })
        observed = tuning_folds[["patient_id", "fold"]].copy()
        observed["patient_id"] = observed["patient_id"].astype(str)
        observed["fold"] = observed["fold"].astype(int)
        if len(observed) != len(expected) or not observed.equals(expected):
            raise RuntimeError(
                f"Training outer folds do not exactly match final Optuna folds at "
                f"{landmark}h/{outcome}/{variant}."
            )
        anchor_tuning_fold_match = True

    oof = np.full(
        len(y_train),
        np.nan,
        dtype=np.float32,
    )
    test_fold_predictions = []
    external_fold_predictions = []
    fold_rows = []

    task_output = task_dir(
        modeling_root,
        landmark,
        outcome,
        variant,
    )

    # A canonical task run must not silently coexist with stale artifacts from
    # an earlier training execution. Each task has its own output directory,
    # so removing it here is safe under ProcessPoolExecutor.
    if task_output.exists():
        shutil.rmtree(task_output)
    model_dir = task_output / "models"
    task_output.mkdir(parents=True, exist_ok=True)

    outer_fold_path = task_output / "outer_fold_assignments.csv"
    pd.DataFrame({
        "patient_id": groups_train,
        "stay_id": train.stay_id[train_mask],
        "fold": outer_fold_assignment,
        "landmark_hour": landmark,
        "outcome": outcome,
        "variant": variant,
    }).to_csv(outer_fold_path, index=False)
    outer_fold_sha256 = sha256_file(outer_fold_path)

    source_path = Path(__file__).resolve()
    modeling_common_path = source_path.with_name("modeling_common.py")
    source_hashes = {
        "trainer_source_file": str(source_path),
        "trainer_source_sha256": sha256_file(source_path),
        "modeling_common_source_file": str(modeling_common_path),
        "modeling_common_source_sha256": sha256_file(modeling_common_path),
    }
    versions = software_versions(xgb)

    training_protocol = {
        "protocol_version": "P5_XGB_TRAINING_FINAL_PROVENANCE_V1",
        "landmark_hour": landmark,
        "outcome": outcome,
        "variant": variant,
        "seed": seed,
        "outer_folds": 5,
        "inner_early_stopping_fraction": float(task["inner_es_fraction"]),
        "max_boost_rounds": int(task["num_boost_round"]),
        "early_stopping_rounds": int(task["early_stopping_rounds"]),
        "xgb_threads": int(task["xgb_threads"]),
        "params_source": task["params_source"],
        "frozen_xgb_params": task["params"],
        "inner_split_seed_rule": "seed + 1000 + fold",
        "fold_model_seed_rule": "seed + fold",
        "test_external_prediction_rule": "arithmetic mean of five outer-fold model predictions",
        "mortality_calibration": "Platt sigmoid fit on MIMIC development OOF raw probabilities only",
        "mortality_threshold": "maximum Youden J on calibrated MIMIC development OOF probabilities",
    }
    training_protocol_sha256 = sha256_json_payload(training_protocol)

    identity_payload = {
        "identity_version": "P5_XGB_TRAINING_IDENTITY_V1",
        "input_hashes": input_hashes,
        "outer_fold_assignments_sha256": outer_fold_sha256,
        "hyperparameter_file_sha256": hp_provenance.get("hyperparameter_file_sha256"),
        "optuna_study_identity_sha256": hp_provenance.get("study_identity_sha256"),
        "training_protocol_sha256": training_protocol_sha256,
        "trainer_source_sha256": source_hashes["trainer_source_sha256"],
        "modeling_common_source_sha256": source_hashes["modeling_common_source_sha256"],
        "software_versions": versions,
    }
    training_identity_sha256 = sha256_json_payload(identity_payload)
    training_identity_record = {
        "training_identity_sha256": training_identity_sha256,
        "training_identity_short": training_identity_sha256[:16],
        "identity_payload": identity_payload,
        "training_protocol": training_protocol,
        "hyperparameter_provenance": hp_provenance,
        "anchor_tuning_input_match": anchor_tuning_input_match,
        "anchor_tuning_fold_match": anchor_tuning_fold_match,
    }
    (task_output / "training_identity.json").write_text(
        json.dumps(training_identity_record, indent=2), encoding="utf-8"
    )

    start = time.time()

    for fold, (
        outer_train_idx,
        outer_valid_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        model_path = (
            model_dir
            / f"fold_{fold}.json"
        )

        (
            model,
            pred_valid,
            best_rounds,
            inner_fit_n,
            inner_es_n,
        ) = fit_fold(
            xgb=xgb,
            X=X_train,
            y=y_train,
            groups=groups_train,
            outer_train_idx=outer_train_idx,
            outer_valid_idx=outer_valid_idx,
            outcome=outcome,
            seed=seed,
            fold=fold,
            feature_names=(
                train.feature_names
            ),
            params=task[
                "params"
            ],
            num_boost_round=int(
                task[
                    "num_boost_round"
                ]
            ),
            early_stopping_rounds=int(
                task[
                    "early_stopping_rounds"
                ]
            ),
            inner_es_fraction=float(
                task[
                    "inner_es_fraction"
                ]
            ),
            xgb_threads=int(
                task[
                    "xgb_threads"
                ]
            ),
            model_path=model_path,
        )

        oof[
            outer_valid_idx
        ] = pred_valid

        dtest = xgb.DMatrix(
            X_test,
            feature_names=(
                train.feature_names
            ),
            missing=np.nan,
        )
        dext = xgb.DMatrix(
            X_external,
            feature_names=(
                train.feature_names
            ),
            missing=np.nan,
        )

        test_fold_predictions.append(
            model.predict(
                dtest
            ).astype(
                np.float32
            )
        )
        external_fold_predictions.append(
            model.predict(
                dext
            ).astype(
                np.float32
            )
        )

        fold_rows.append(
            {
                "fold": fold,
                "outer_train_n": int(
                    len(
                        outer_train_idx
                    )
                ),
                "outer_valid_n": int(
                    len(
                        outer_valid_idx
                    )
                ),
                "outer_train_patients": int(
                    len(
                        np.unique(
                            groups_train[
                                outer_train_idx
                            ]
                        )
                    )
                ),
                "outer_valid_patients": int(
                    len(
                        np.unique(
                            groups_train[
                                outer_valid_idx
                            ]
                        )
                    )
                ),
                "patient_overlap_count": int(
                    len(
                        set(
                            groups_train[
                                outer_train_idx
                            ]
                        )
                        & set(
                            groups_train[
                                outer_valid_idx
                            ]
                        )
                    )
                ),
                "inner_fit_n": int(
                    inner_fit_n
                ),
                "inner_es_n": int(
                    inner_es_n
                ),
                "best_rounds": int(
                    best_rounds
                ),
                "model_path": str(
                    model_path
                ),
            }
        )

    if np.isnan(
        oof
    ).any():
        raise RuntimeError(
            f"OOF predictions incomplete: {landmark}/{outcome}/{variant}"
        )

    pred_test_raw = np.mean(
        np.stack(
            test_fold_predictions,
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    pred_external_raw = np.mean(
        np.stack(
            external_fold_predictions,
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)

    prediction_tables = {}
    metric_rows = []
    calibrator_meta = None

    if outcome == "los":
        cohort_specs = [
            (
                "mimic_train_oof",
                y_train,
                oof,
                train.patient_id[
                    train_mask
                ],
                train.stay_id[
                    train_mask
                ],
            ),
            (
                "mimic_test",
                y_test,
                pred_test_raw,
                test.patient_id[
                    test_mask
                ],
                test.stay_id[
                    test_mask
                ],
            ),
            (
                "eicu_external",
                y_external,
                pred_external_raw,
                external.patient_id[
                    external_mask
                ],
                external.stay_id[
                    external_mask
                ],
            ),
        ]

        for cohort, y, pred, pid, sid in cohort_specs:
            metrics = regression_metrics(
                y,
                pred,
            )
            metric_rows.append(
                {
                    "landmark_hour": landmark,
                    "outcome": outcome,
                    "variant": variant,
                    "cohort": cohort,
                    "n": int(
                        len(y)
                    ),
                    **metrics,
                }
            )

            prediction_tables[
                cohort
            ] = pd.DataFrame(
                {
                    "patient_id": pid,
                    "stay_id": sid,
                    "y_true": y,
                    "prediction": pred,
                }
            )
    else:
        # Fit Platt/sigmoid calibration on MIMIC OOF predictions only.
        eps = 1e-6
        oof_logit = np.log(
            np.clip(
                oof,
                eps,
                1.0 - eps,
            )
            / (
                1.0
                - np.clip(
                    oof,
                    eps,
                    1.0 - eps,
                )
            )
        ).reshape(-1, 1)

        # Near-unpenalized logistic calibration, compatible across
        # scikit-learn versions.
        calibrator = LogisticRegression(
            C=1e6,
            solver="lbfgs",
            max_iter=2000,
        )
        calibrator.fit(
            oof_logit,
            y_train,
        )

        def calibrate(p):
            p = np.clip(
                p,
                eps,
                1.0 - eps,
            )
            logit = np.log(
                p / (1.0 - p)
            ).reshape(-1, 1)
            return calibrator.predict_proba(
                logit
            )[:, 1].astype(
                np.float32
            )

        pred_oof_cal = calibrate(
            oof
        )
        pred_test_cal = calibrate(
            pred_test_raw
        )
        pred_external_cal = calibrate(
            pred_external_raw
        )

        frozen_threshold = (
            youden_threshold(
                y_train,
                pred_oof_cal,
            )
        )

        calibrator_meta = {
            "type": "sigmoid_platt",
            "fit_source": (
                "MIMIC train OOF predictions only"
            ),
            "coef": float(
                calibrator.coef_[0, 0]
            ),
            "intercept": float(
                calibrator.intercept_[0]
            ),
            "frozen_threshold": float(
                frozen_threshold
            ),
            "threshold_rule": (
                "maximum Youden J on calibrated MIMIC OOF predictions"
            ),
        }

        cohort_specs = [
            (
                "mimic_train_oof",
                y_train,
                oof,
                pred_oof_cal,
                train.patient_id[
                    train_mask
                ],
                train.stay_id[
                    train_mask
                ],
            ),
            (
                "mimic_test",
                y_test,
                pred_test_raw,
                pred_test_cal,
                test.patient_id[
                    test_mask
                ],
                test.stay_id[
                    test_mask
                ],
            ),
            (
                "eicu_external",
                y_external,
                pred_external_raw,
                pred_external_cal,
                external.patient_id[
                    external_mask
                ],
                external.stay_id[
                    external_mask
                ],
            ),
        ]

        for (
            cohort,
            y,
            pred_raw,
            pred_cal,
            pid,
            sid,
        ) in cohort_specs:
            raw_metrics = (
                safe_classification_metrics(
                    y,
                    pred_raw,
                )
            )
            calibrated_metrics = (
                safe_classification_metrics(
                    y,
                    pred_cal,
                )
            )
            slope, intercept = (
                calibration_slope_intercept(
                    y,
                    pred_cal,
                )
            )
            threshold_values = (
                threshold_metrics(
                    y,
                    pred_cal,
                    frozen_threshold,
                )
            )

            metric_rows.append(
                {
                    "landmark_hour": landmark,
                    "outcome": outcome,
                    "variant": variant,
                    "cohort": cohort,
                    "n": int(
                        len(y)
                    ),
                    "positives": int(
                        np.sum(
                            y == 1
                        )
                    ),
                    "prevalence": float(
                        np.mean(
                            y
                        )
                    ),
                    "roc_auc_raw": (
                        raw_metrics[
                            "roc_auc"
                        ]
                    ),
                    "average_precision_raw": (
                        raw_metrics[
                            "average_precision"
                        ]
                    ),
                    "brier_raw": (
                        raw_metrics[
                            "brier"
                        ]
                    ),
                    "logloss_raw": (
                        raw_metrics[
                            "logloss"
                        ]
                    ),
                    "roc_auc_calibrated": (
                        calibrated_metrics[
                            "roc_auc"
                        ]
                    ),
                    "average_precision_calibrated": (
                        calibrated_metrics[
                            "average_precision"
                        ]
                    ),
                    "brier_calibrated": (
                        calibrated_metrics[
                            "brier"
                        ]
                    ),
                    "logloss_calibrated": (
                        calibrated_metrics[
                            "logloss"
                        ]
                    ),
                    "calibration_slope": slope,
                    "calibration_intercept": (
                        intercept
                    ),
                    **threshold_values,
                }
            )

            prediction_tables[
                cohort
            ] = pd.DataFrame(
                {
                    "patient_id": pid,
                    "stay_id": sid,
                    "y_true": y,
                    "prediction_raw": (
                        pred_raw
                    ),
                    "prediction_calibrated": (
                        pred_cal
                    ),
                    "frozen_threshold": (
                        frozen_threshold
                    ),
                }
            )

        # Unknown external mortality rows are retained separately with
        # predictions so no patient is silently discarded.
        unknown_external = (
            ~external.mortality_known
        )
        if unknown_external.any():
            dunknown = xgb.DMatrix(
                external.X[
                    unknown_external
                ],
                feature_names=(
                    train.feature_names
                ),
                missing=np.nan,
            )
            fold_unknown = []
            for fold_row in fold_rows:
                model = xgb.Booster()
                model.load_model(
                    fold_row[
                        "model_path"
                    ]
                )
                fold_unknown.append(
                    model.predict(
                        dunknown
                    )
                )
            raw_unknown = np.mean(
                np.stack(
                    fold_unknown,
                    axis=0,
                ),
                axis=0,
            )
            cal_unknown = calibrate(
                raw_unknown
            )
            prediction_tables[
                "eicu_external_unknown_mortality"
            ] = pd.DataFrame(
                {
                    "patient_id": (
                        external.patient_id[
                            unknown_external
                        ]
                    ),
                    "stay_id": (
                        external.stay_id[
                            unknown_external
                        ]
                    ),
                    "y_true": np.nan,
                    "prediction_raw": (
                        raw_unknown
                    ),
                    "prediction_calibrated": (
                        cal_unknown
                    ),
                    "frozen_threshold": (
                        frozen_threshold
                    ),
                }
            )

    metrics_df = pd.DataFrame(
        metric_rows
    )
    metrics_df.to_csv(
        task_output
        / "metrics.csv",
        index=False,
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        task_output
        / "fold_training.csv",
        index=False,
    )

    prediction_file_hashes = {}
    for cohort, frame in prediction_tables.items():
        prediction_path = task_output / f"predictions_{cohort}.parquet"
        frame.to_parquet(prediction_path, index=False)
        prediction_file_hashes[cohort] = {
            "file": str(prediction_path),
            "sha256": sha256_file(prediction_path),
            "rows": int(len(frame)),
        }

    model_file_hashes = {}
    for fold_row in fold_rows:
        mp = Path(fold_row["model_path"])
        model_file_hashes[str(fold_row["fold"])] = {
            "file": str(mp),
            "sha256": sha256_file(mp),
            "best_rounds": int(fold_row["best_rounds"]),
        }

    (
        task_output
        / "feature_names.json"
    ).write_text(
        json.dumps(
            {
                "feature_names": (
                    train.feature_names
                ),
                "feature_count": len(
                    train.feature_names
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "landmark_hour": landmark,
        "outcome": outcome,
        "variant": variant,
        "seed": seed,
        "outer_cv": (
            "5-fold StratifiedGroupKFold grouped by patient_id"
            if outcome == "mortality"
            else "5-fold GroupKFold grouped by patient_id"
        ),
        "cross_validation_unit": "patient_id",
        "patient_overlap_between_outer_train_valid_allowed": False,
        "identifiers_used_as_predictors": False,
        "forbidden_identifier_predictors_checked": True,
        "outer_fold_never_used_for_early_stopping": True,
        "inner_early_stopping_fraction": float(
            task[
                "inner_es_fraction"
            ]
        ),
        "mimic_test_used_for_fitting": False,
        "eicu_used_for_fitting": False,
        "xgb_params": task[
            "params"
        ],
        "hyperparameter_source": task[
            "params_source"
        ],
        "optuna_tuning_landmark_hour": (
            int(
                task[
                    "tuning_landmark"
                ]
            )
            if task[
                "params_source"
            ] == "optuna"
            else None
        ),
        "hyperparameter_file": task.get(
            "hyperparameter_file"
        ),
        "hyperparameter_provenance": hp_provenance,
        "optuna_study_identity_sha256": hp_provenance.get(
            "study_identity_sha256"
        ),
        "anchor_tuning_input_match": anchor_tuning_input_match,
        "anchor_tuning_fold_match": anchor_tuning_fold_match,
        "training_identity_sha256": training_identity_sha256,
        "training_identity_file": str(task_output / "training_identity.json"),
        "input_hashes": input_hashes,
        "outer_fold_assignments_file": str(outer_fold_path),
        "outer_fold_assignments_sha256": outer_fold_sha256,
        "training_protocol_sha256": training_protocol_sha256,
        "training_protocol": training_protocol,
        **source_hashes,
        "software_versions": versions,
        "model_files": model_file_hashes,
        "prediction_files": prediction_file_hashes,
        "num_boost_round_max": int(
            task[
                "num_boost_round"
            ]
        ),
        "early_stopping_rounds": int(
            task[
                "early_stopping_rounds"
            ]
        ),
        "xgb_threads": int(
            task[
                "xgb_threads"
            ]
        ),
        "feature_count": int(
            train.X.shape[1]
        ),
        "calibration": calibrator_meta,
        "elapsed_seconds": float(
            time.time() - start
        ),
    }

    (
        task_output
        / "task_manifest.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "landmark_hour": landmark,
        "outcome": outcome,
        "variant": variant,
        "status": "PASS",
        "elapsed_seconds": metadata[
            "elapsed_seconds"
        ],
        "output_dir": str(
            task_output
        ),
        "training_identity_sha256": training_identity_sha256,
        "optuna_study_identity_sha256": hp_provenance.get(
            "study_identity_sha256"
        ),
        "anchor_tuning_input_match": anchor_tuning_input_match,
        "anchor_tuning_fold_match": anchor_tuning_fold_match,
    }


def build_parser():
    p = argparse.ArgumentParser()
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
    )
    p.add_argument(
        "--variants",
        type=str,
        default="full,o1,o2,o3,o4,static",
    )
    p.add_argument(
        "--outcomes",
        type=str,
        default="los,mortality",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel landmark/outcome/variant tasks. Keep workers * "
            "xgb-threads <= available CPUs."
        ),
    )
    p.add_argument(
        "--xgb-threads",
        type=int,
        default=8,
    )
    p.add_argument(
        "--num-boost-round",
        type=int,
        default=DEFAULT_NUM_BOOST_ROUND,
    )
    p.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=DEFAULT_EARLY_STOPPING,
    )
    p.add_argument(
        "--inner-es-fraction",
        type=float,
        default=DEFAULT_INNER_ES_FRACTION,
    )
    p.add_argument(
        "--params-source",
        choices=["optuna", "default"],
        default="optuna",
        help=(
            "Use frozen Optuna parameters by default. "
            "Use 'default' only for baseline/smoke testing."
        ),
    )
    p.add_argument(
        "--tuning-landmark",
        type=int,
        default=DEFAULT_TUNING_LANDMARK,
        help=(
            "Anchor landmark whose Optuna parameters are frozen across "
            "all prediction landmarks."
        ),
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    modeling_root = (
        args.modeling_root
        .expanduser()
        .resolve()
        if args.modeling_root is not None
        else project_root
        / "01_Modeling"
    )

    logger = configure_logging(
        modeling_root
    )
    if args.workers <= 0 or args.xgb_threads <= 0:
        raise ValueError("--workers and --xgb-threads must be positive.")
    if args.num_boost_round <= 0 or args.early_stopping_rounds <= 0:
        raise ValueError("Boosting and early-stopping rounds must be positive.")
    if not (0.0 < args.inner_es_fraction < 0.5):
        raise ValueError("--inner-es-fraction must be in (0, 0.5).")

    matrix_manifest_path = (
        modeling_root
        / "reports"
        / "model_matrix_manifest.csv"
    )
    if not matrix_manifest_path.is_file():
        raise FileNotFoundError(
            f"{matrix_manifest_path}. Run 01_prepare_model_matrices.py first."
        )

    matrix_manifest = pd.read_csv(
        matrix_manifest_path
    )

    tensor_manifest = load_json(
        project_root
        / "data"
        / "06_numpy_cubes"
        / "06_tensor_manifest.json"
    )

    landmarks = parse_landmarks(
        args.landmarks,
        tensor_manifest,
    )
    variants = parse_csv_arg(
        args.variants,
        VALID_VARIANTS,
    )
    outcomes = parse_csv_arg(
        args.outcomes,
        VALID_OUTCOMES,
    )

    tasks = []
    for landmark in landmarks:
        for variant in variants:
            row = matrix_manifest.loc[
                (
                    matrix_manifest[
                        "landmark_hour"
                    ].eq(landmark)
                )
                & (
                    matrix_manifest[
                        "variant"
                    ].eq(variant)
                )
            ]
            if row.empty:
                continue

            raw_available = row.iloc[0]["available"]
            if isinstance(raw_available, str):
                available = raw_available.strip().lower() in {"true", "1", "yes"}
            else:
                available = bool(raw_available)
            if not available:
                continue

            for outcome in outcomes:
                task_params, params_file, params_provenance = (
                    load_task_params(
                        modeling_root,
                        tuning_landmark=(
                            args.tuning_landmark
                        ),
                        outcome=outcome,
                        variant=variant,
                        params_source=(
                            args.params_source
                        ),
                    )
                )

                tasks.append(
                    {
                        "modeling_root": str(
                            modeling_root
                        ),
                        "landmark": landmark,
                        "variant": variant,
                        "outcome": outcome,
                        "seed": args.seed,
                        "xgb_threads": (
                            args.xgb_threads
                        ),
                        "num_boost_round": (
                            args.num_boost_round
                        ),
                        "early_stopping_rounds": (
                            args.early_stopping_rounds
                        ),
                        "inner_es_fraction": (
                            args.inner_es_fraction
                        ),
                        "params": task_params,
                        "params_source": (
                            args.params_source
                        ),
                        "tuning_landmark": (
                            args.tuning_landmark
                        ),
                        "hyperparameter_file": (
                            params_file
                        ),
                        "hyperparameter_provenance": params_provenance,
                    }
                )

    logger.info(
        "Tasks: %d | workers=%d | xgb_threads=%d",
        len(tasks),
        args.workers,
        args.xgb_threads,
    )
    logger.info(
        "Hyperparameters: source=%s | tuning_landmark=%dh",
        args.params_source,
        args.tuning_landmark,
    )

    detected_cpus = os.cpu_count()
    if (
        detected_cpus is not None
        and args.workers * args.xgb_threads > detected_cpus
    ):
        logger.warning(
            "workers*xgb_threads=%d exceeds detected CPUs=%s.",
            args.workers
            * args.xgb_threads,
            detected_cpus,
        )

    results = []

    if args.workers == 1:
        for i, task in enumerate(
            tasks,
            start=1,
        ):
            logger.info(
                "[%d/%d] %dh %s %s",
                i,
                len(tasks),
                task[
                    "landmark"
                ],
                task[
                    "outcome"
                ],
                task[
                    "variant"
                ],
            )
            result = run_task(
                task
            )
            results.append(
                result
            )
            logger.info(
                "PASS | %.1fs | %s",
                result[
                    "elapsed_seconds"
                ],
                result[
                    "output_dir"
                ],
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            futures = {
                pool.submit(
                    run_task,
                    task,
                ): task
                for task in tasks
            }

            for future in as_completed(
                futures
            ):
                task = futures[
                    future
                ]
                try:
                    result = future.result()
                except Exception:
                    logger.exception(
                        "FAILED | %dh %s %s",
                        task[
                            "landmark"
                        ],
                        task[
                            "outcome"
                        ],
                        task[
                            "variant"
                        ],
                    )
                    raise
                else:
                    results.append(
                        result
                    )
                    logger.info(
                        "PASS | %dh %s %s | %.1fs",
                        result[
                            "landmark_hour"
                        ],
                        result[
                            "outcome"
                        ],
                        result[
                            "variant"
                        ],
                        result[
                            "elapsed_seconds"
                        ],
                    )

    report_dir = (
        modeling_root
        / "reports"
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        results
    ).sort_values(
        [
            "landmark_hour",
            "outcome",
            "variant",
        ]
    ).to_csv(
        report_dir
        / "training_task_summary.csv",
        index=False,
    )

    logger.info(
        "Training complete. Run 04_collect_metrics.py next."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
