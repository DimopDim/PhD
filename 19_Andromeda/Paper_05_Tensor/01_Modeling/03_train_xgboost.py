#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_train_xgboost.py

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
import json
import logging
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
DEFAULT_EARLY_STOPPING = 100
DEFAULT_INNER_ES_FRACTION = 0.10
DEFAULT_TUNING_LANDMARK = 24


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
) -> Tuple[Dict, Optional[str]]:
    if params_source == "default":
        return dict(DEFAULT_PARAMS), None

    path = hyperparameter_file(
        modeling_root,
        tuning_landmark,
        outcome,
        variant,
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen Optuna parameters: {path}. "
            "Run 01_optuna_tune_xgboost.py first, or explicitly use "
            "--params-source default."
        )

    payload = load_json(path)
    params = dict(
        payload["xgb_params"]
    )

    # Runtime-only parameters are set later per fold.
    for key in [
        "objective",
        "eval_metric",
        "nthread",
        "seed",
        "verbosity",
    ]:
        params.pop(
            key,
            None,
        )

    return params, str(path)


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
        / "01_train_xgboost.log",
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
            "Run 00_prepare_model_matrices.py first."
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
    model_dir = (
        task_output
        / "models"
    )
    task_output.mkdir(
        parents=True,
        exist_ok=True,
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

    for cohort, frame in (
        prediction_tables.items()
    ):
        frame.to_parquet(
            task_output
            / f"predictions_{cohort}.parquet",
            index=False,
        )

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

    matrix_manifest_path = (
        modeling_root
        / "reports"
        / "model_matrix_manifest.csv"
    )
    if not matrix_manifest_path.is_file():
        raise FileNotFoundError(
            f"{matrix_manifest_path}. Run 00_prepare_model_matrices.py first."
        )

    matrix_manifest = pd.read_csv(
        matrix_manifest_path
    )

    tensor_manifest = load_json(
        project_root
        / "00_Create_Tensor"
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

            available = bool(
                row.iloc[0][
                    "available"
                ]
            )
            if not available:
                continue

            for outcome in outcomes:
                task_params, params_file = (
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

    if (
        args.workers
        * args.xgb_threads
        > os.cpu_count()
    ):
        logger.warning(
            "workers*xgb_threads=%d exceeds detected CPUs=%s.",
            args.workers
            * args.xgb_threads,
            os.cpu_count(),
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
        "Training complete. Run 02_collect_metrics.py next."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
