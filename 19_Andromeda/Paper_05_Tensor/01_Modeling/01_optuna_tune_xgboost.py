#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_optuna_tune_xgboost.py

Leakage-safe Optuna hyperparameter optimization for the multi-landmark
XGBoost pipeline.

Tuning policy
-------------
Hyperparameters are tuned ONCE per:

    outcome × representation

at a single fully aligned tuning landmark (default: 24 h), using only the
MIMIC development partition.

The resulting hyperparameters are then frozen and reused unchanged at every
prediction landmark. This avoids giving each landmark a separately optimized
model and keeps the performance-vs-landmark analysis focused on information
availability rather than landmark-specific hyperparameter searching.

Default tuning jobs:
    LOS       × full,o1,o2,o3,o4,static
    Mortality × full,o1,o2,o3,o4,static

No MIMIC test or eICU record is loaded by this script.

Cross-validation inside each Optuna trial
-----------------------------------------
Each trial uses five fixed outer folds inside MIMIC development.

For each outer fold:
    outer training subset
        ↓
    inner fit / inner early-stopping split
        ↓
    choose number of boosting rounds
        ↓
    retrain with trial hyperparameters on the full outer-training subset
        ↓
    evaluate on untouched outer validation fold

Objectives:
    LOS       -> minimize mean outer-fold RMSE
    Mortality -> maximize mean outer-fold Average Precision

This means even the fold used to score an Optuna trial is not used for early
stopping.

Optuna studies are persistent SQLite databases and can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
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
    load_model_matrix,
    matrix_cache_paths,
    parse_csv_arg,
)


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)
MODELING_ROOT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "01_Modeling"
)

DEFAULT_TUNING_LANDMARK = 24
DEFAULT_TRIALS = 40
DEFAULT_OUTER_FOLDS = 5
DEFAULT_INNER_ES_FRACTION = 0.10
DEFAULT_NUM_BOOST_ROUND = 4000
DEFAULT_EARLY_STOPPING = 100

FIXED_XGB_PARAMS = {
    "tree_method": "hist",
    "verbosity": 0,
}

SEARCH_SPACE_DESCRIPTION = {
    "eta": "loguniform[0.01, 0.15]",
    "max_depth": "int[3, 10]",
    "min_child_weight": "loguniform[0.5, 20]",
    "subsample": "uniform[0.60, 1.00]",
    "colsample_bytree": "uniform[0.50, 1.00]",
    "gamma": "uniform[0, 5]",
    "alpha": "loguniform[1e-5, 10]",
    "lambda": "loguniform[1e-3, 30]",
    "max_bin": "categorical[128,256,512]",
}


def get_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required."
        ) from exc
    return xgb


def get_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "optuna is required. Install it in the active environment, e.g. "
            "`python -m pip install optuna`."
        ) from exc
    return optuna


def configure_logging(
    modeling_root: Path,
) -> logging.Logger:
    log_dir = modeling_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("optuna_tune")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        log_dir / "01_optuna_tune_xgboost.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_training_matrix(
    modeling_root: Path,
    landmark: int,
    variant: str,
):
    matrix_path, feature_path = matrix_cache_paths(
        modeling_root,
        landmark,
        variant,
        "mimic",
        "train",
    )
    if not matrix_path.is_file():
        raise FileNotFoundError(
            f"Missing model matrix {matrix_path}. "
            "Run 00_prepare_model_matrices.py first."
        )
    return load_model_matrix(
        matrix_path,
        feature_path,
    )


def hyperparameter_dir(
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
    )


def fixed_outer_splits(
    y: np.ndarray,
    groups: np.ndarray,
    outcome: str,
    seed: int,
    n_splits: int,
):
    """
    Five-fold CV is explicitly patient-grouped.

    groups = patient_id. A patient can therefore never occur in both the
    training and validation partition of the same fold, even if a future
    representation were to contain multiple rows per patient.
    """
    groups = np.asarray(groups).astype(str)

    if outcome == "mortality":
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
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
        # GroupKFold in current sklearn supports shuffle/random_state.
        splitter = GroupKFold(
            n_splits=n_splits,
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
                f"Patient leakage in Optuna outer fold {fold}: "
                f"{list(overlap)[:10]}"
            )

    return splits


def make_inner_split(
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    outcome: str,
    seed: int,
    fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inner early-stopping split is also patient-level.

    The current matrices contain exactly one row per patient, which is
    asserted before tuning. We nevertheless split using patient_id metadata
    and verify disjointness explicitly.
    """
    local = np.arange(len(outer_train_idx))
    outer_groups = np.asarray(
        groups[outer_train_idx]
    ).astype(str)

    if len(np.unique(outer_groups)) != len(outer_groups):
        raise RuntimeError(
            "Inner split received duplicate patient groups."
        )

    if outcome == "mortality":
        fit_local, es_local = train_test_split(
            local,
            test_size=fraction,
            random_state=seed,
            shuffle=True,
            stratify=y[outer_train_idx],
        )
    else:
        fit_local, es_local = train_test_split(
            local,
            test_size=fraction,
            random_state=seed,
            shuffle=True,
        )

    fit_idx = outer_train_idx[fit_local]
    es_idx = outer_train_idx[es_local]

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


def suggest_params(
    trial,
) -> Dict:
    params = {
        "eta": trial.suggest_float(
            "eta",
            0.01,
            0.15,
            log=True,
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            0.5,
            20.0,
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.50,
            1.00,
        ),
        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            5.0,
        ),
        "alpha": trial.suggest_float(
            "alpha",
            1e-5,
            10.0,
            log=True,
        ),
        "lambda": trial.suggest_float(
            "lambda",
            1e-3,
            30.0,
            log=True,
        ),
        "max_bin": trial.suggest_categorical(
            "max_bin",
            [128, 256, 512],
        ),
    }
    params.update(FIXED_XGB_PARAMS)
    return params


def fold_score(
    outcome: str,
    y_true: np.ndarray,
    pred: np.ndarray,
) -> float:
    if outcome == "los":
        return float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    pred,
                )
            )
        )

    return float(
        average_precision_score(
            y_true,
            pred,
        )
    )


def evaluate_trial(
    *,
    xgb,
    trial,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    outer_splits,
    outcome: str,
    seed: int,
    xgb_threads: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    inner_es_fraction: float,
):
    params = suggest_params(
        trial
    )
    params.update(
        {
            "nthread": int(xgb_threads),
            "seed": int(seed + trial.number),
        }
    )

    if outcome == "los":
        params["objective"] = "reg:squarederror"
        params["eval_metric"] = "rmse"
    else:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = "logloss"

    scores: List[float] = []
    best_rounds: List[int] = []

    for fold, (
        outer_train_idx,
        outer_valid_idx,
    ) in enumerate(
        outer_splits,
        start=1,
    ):
        fit_idx, es_idx = make_inner_split(
            outer_train_idx,
            y,
            groups,
            outcome,
            seed=(
                seed
                + 10000
                + trial.number * 100
                + fold
            ),
            fraction=inner_es_fraction,
        )

        dfit = xgb.DMatrix(
            X[fit_idx],
            label=y[fit_idx],
            feature_names=feature_names,
            missing=np.nan,
        )
        des = xgb.DMatrix(
            X[es_idx],
            label=y[es_idx],
            feature_names=feature_names,
            missing=np.nan,
        )

        provisional = xgb.train(
            params,
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
        rounds = (
            int(best_iteration) + 1
            if best_iteration is not None
            else int(num_boost_round)
        )
        best_rounds.append(rounds)

        douter = xgb.DMatrix(
            X[outer_train_idx],
            label=y[outer_train_idx],
            feature_names=feature_names,
            missing=np.nan,
        )
        model = xgb.train(
            params,
            douter,
            num_boost_round=rounds,
            verbose_eval=False,
        )

        dvalid = xgb.DMatrix(
            X[outer_valid_idx],
            feature_names=feature_names,
            missing=np.nan,
        )
        pred = model.predict(dvalid)

        score = fold_score(
            outcome,
            y[outer_valid_idx],
            pred,
        )
        scores.append(score)

        running = float(
            np.mean(scores)
        )
        trial.report(
            running,
            step=fold,
        )

        if trial.should_prune():
            trial.set_user_attr(
                "fold_scores_partial",
                [float(x) for x in scores],
            )
            trial.set_user_attr(
                "best_rounds_partial",
                [int(x) for x in best_rounds],
            )
            optuna = get_optuna()
            raise optuna.TrialPruned()

    trial.set_user_attr(
        "fold_scores",
        [float(x) for x in scores],
    )
    trial.set_user_attr(
        "best_rounds",
        [int(x) for x in best_rounds],
    )
    trial.set_user_attr(
        "mean_best_rounds",
        float(np.mean(best_rounds)),
    )
    trial.set_user_attr(
        "median_best_rounds",
        float(np.median(best_rounds)),
    )

    return float(
        np.mean(scores)
    )


def run_tuning_job(
    job: dict,
) -> dict:
    optuna = get_optuna()
    xgb = get_xgboost()

    # Keep Optuna worker logs concise.
    optuna.logging.set_verbosity(
        optuna.logging.WARNING
    )

    modeling_root = Path(
        job["modeling_root"]
    )
    tuning_landmark = int(
        job["tuning_landmark"]
    )
    outcome = str(
        job["outcome"]
    )
    variant = str(
        job["variant"]
    )
    seed = int(
        job["seed"]
    )

    matrix = load_training_matrix(
        modeling_root,
        tuning_landmark,
        variant,
    )

    if outcome == "los":
        X = matrix.X
        y = matrix.y_los_remaining_days.astype(
            np.float32
        )
        direction = "minimize"
        objective_name = "mean_outer_cv_rmse"
    else:
        if not matrix.mortality_known.all():
            raise RuntimeError(
                "MIMIC development mortality must be fully known."
            )
        X = matrix.X[
            matrix.mortality_known
        ]
        y = matrix.y_mortality[
            matrix.mortality_known
        ].astype(np.int8)
        direction = "maximize"
        objective_name = "mean_outer_cv_average_precision"

    groups = matrix.patient_id.astype(str)

    assert_unique_patient_rows(
        groups,
        label=(
            f"Optuna {tuning_landmark}h/{outcome}/{variant}"
        ),
    )
    assert_no_identifier_predictors(
        matrix.feature_names
    )

    if len(groups) != len(y):
        raise RuntimeError(
            "Patient-group vector and outcome vector differ in length."
        )

    outer_splits = fixed_outer_splits(
        y,
        groups,
        outcome,
        seed,
        int(job["outer_folds"]),
    )

    out_dir = hyperparameter_dir(
        modeling_root,
        tuning_landmark,
        outcome,
        variant,
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    db_path = out_dir / "study.db"
    storage = (
        "sqlite:///"
        + str(db_path.resolve())
    )
    study_name = (
        f"xgb_{outcome}_{variant}_{tuning_landmark:03d}h"
    )

    sampler = optuna.samplers.TPESampler(
        seed=(
            seed
            + tuning_landmark * 10
            + sum(ord(c) for c in outcome + variant)
        ),
        multivariate=True,
        n_startup_trials=10,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=8,
        n_warmup_steps=2,
        interval_steps=1,
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    target_trials = int(
        job["trials"]
    )
    existing = len(
        study.trials
    )
    remaining = max(
        0,
        target_trials - existing,
    )

    start = time.time()

    if remaining > 0:
        def objective(trial):
            return evaluate_trial(
                xgb=xgb,
                trial=trial,
                X=X,
                y=y,
                groups=groups,
                feature_names=(
                    matrix.feature_names
                ),
                outer_splits=outer_splits,
                outcome=outcome,
                seed=seed,
                xgb_threads=int(
                    job["xgb_threads"]
                ),
                num_boost_round=int(
                    job["num_boost_round"]
                ),
                early_stopping_rounds=int(
                    job[
                        "early_stopping_rounds"
                    ]
                ),
                inner_es_fraction=float(
                    job[
                        "inner_es_fraction"
                    ]
                ),
            )

        study.optimize(
            objective,
            n_trials=remaining,
            gc_after_trial=True,
            show_progress_bar=False,
        )

    complete_trials = [
        t
        for t in study.trials
        if t.state
        == optuna.trial.TrialState.COMPLETE
    ]
    if not complete_trials:
        raise RuntimeError(
            f"No complete Optuna trial for {outcome}/{variant}."
        )

    best_trial = study.best_trial

    best_xgb_params = dict(
        best_trial.params
    )
    best_xgb_params.update(
        FIXED_XGB_PARAMS
    )

    fold_scores = best_trial.user_attrs.get(
        "fold_scores",
        []
    )
    best_rounds = best_trial.user_attrs.get(
        "best_rounds",
        []
    )

    result = {
        "tuning_landmark_hour": tuning_landmark,
        "tuning_landmark_policy": (
            "single fully aligned anchor landmark; hyperparameters frozen "
            "and reused across all prediction landmarks"
        ),
        "outcome": outcome,
        "variant": variant,
        "objective": objective_name,
        "direction": direction,
        "best_value": float(
            best_trial.value
        ),
        "best_trial_number": int(
            best_trial.number
        ),
        "xgb_params": best_xgb_params,
        "best_trial_fold_scores": [
            float(x)
            for x in fold_scores
        ],
        "best_trial_best_rounds": [
            int(x)
            for x in best_rounds
        ],
        "best_trial_median_best_rounds": (
            float(np.median(best_rounds))
            if best_rounds
            else None
        ),
        "development_patients": int(
            len(y)
        ),
        "feature_count": int(
            X.shape[1]
        ),
        "outer_folds": int(
            job["outer_folds"]
        ),
        "cross_validation_unit": "patient_id",
        "patient_group_overlap_allowed": False,
        "identifiers_used_as_predictors": False,
        "inner_early_stopping_fraction": float(
            job[
                "inner_es_fraction"
            ]
        ),
        "max_boost_rounds": int(
            job["num_boost_round"]
        ),
        "early_stopping_rounds": int(
            job[
                "early_stopping_rounds"
            ]
        ),
        "search_space": (
            SEARCH_SPACE_DESCRIPTION
        ),
        "mimic_test_used": False,
        "eicu_used": False,
        "study_name": study_name,
        "study_database": str(
            db_path
        ),
        "target_total_trials": target_trials,
        "total_trials_in_study": int(
            len(study.trials)
        ),
        "complete_trials": int(
            sum(
                t.state
                == optuna.trial.TrialState.COMPLETE
                for t in study.trials
            )
        ),
        "pruned_trials": int(
            sum(
                t.state
                == optuna.trial.TrialState.PRUNED
                for t in study.trials
            )
        ),
        "failed_trials": int(
            sum(
                t.state
                == optuna.trial.TrialState.FAIL
                for t in study.trials
            )
        ),
        "elapsed_seconds_this_run": float(
            time.time() - start
        ),
    }

    (
        out_dir / "best_params.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    trials_df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
            "state",
        )
    )
    trials_df.to_csv(
        out_dir / "trials.csv",
        index=False,
    )

    return {
        "tuning_landmark_hour": tuning_landmark,
        "outcome": outcome,
        "variant": variant,
        "best_value": float(
            best_trial.value
        ),
        "best_trial_number": int(
            best_trial.number
        ),
        "total_trials": int(
            len(study.trials)
        ),
        "complete_trials": int(
            result["complete_trials"]
        ),
        "pruned_trials": int(
            result["pruned_trials"]
        ),
        "feature_count": int(
            X.shape[1]
        ),
        "best_params_file": str(
            out_dir / "best_params.json"
        ),
        "status": "PASS",
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
        "--tuning-landmark",
        type=int,
        default=DEFAULT_TUNING_LANDMARK,
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
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=(
            "Target total number of Optuna trials per outcome×variant study. "
            "Persistent studies resume to this total."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Parallel outcome×variant studies. Keep workers*xgb-threads "
            "below available CPUs."
        ),
    )
    p.add_argument(
        "--xgb-threads",
        type=int,
        default=8,
    )
    p.add_argument(
        "--outer-folds",
        type=int,
        default=DEFAULT_OUTER_FOLDS,
    )
    p.add_argument(
        "--inner-es-fraction",
        type=float,
        default=DEFAULT_INNER_ES_FRACTION,
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
        "--seed",
        type=int,
        default=42,
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
        else project_root / "01_Modeling"
    )

    logger = configure_logging(
        modeling_root
    )

    variants = parse_csv_arg(
        args.variants,
        VALID_VARIANTS,
    )
    outcomes = parse_csv_arg(
        args.outcomes,
        VALID_OUTCOMES,
    )

    # Check that the requested tuning landmark has cached matrices.
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

    jobs = []
    for outcome in outcomes:
        for variant in variants:
            row = matrix_manifest.loc[
                (
                    matrix_manifest[
                        "landmark_hour"
                    ].eq(
                        args.tuning_landmark
                    )
                )
                & (
                    matrix_manifest[
                        "variant"
                    ].eq(
                        variant
                    )
                )
            ]

            if row.empty:
                raise RuntimeError(
                    f"No cached matrix manifest entry for "
                    f"{args.tuning_landmark}h/{variant}."
                )

            raw_available = row.iloc[0][
                "available"
            ]
            if isinstance(
                raw_available,
                str,
            ):
                available = (
                    raw_available
                    .strip()
                    .lower()
                    in {
                        "true",
                        "1",
                        "yes",
                    }
                )
            else:
                available = bool(
                    raw_available
                )

            if not available:
                raise RuntimeError(
                    f"{args.tuning_landmark}h/{variant} is unavailable. "
                    "Choose a fully aligned tuning landmark."
                )

            jobs.append(
                {
                    "modeling_root": str(
                        modeling_root
                    ),
                    "tuning_landmark": int(
                        args.tuning_landmark
                    ),
                    "outcome": outcome,
                    "variant": variant,
                    "trials": int(
                        args.trials
                    ),
                    "workers": int(
                        args.workers
                    ),
                    "xgb_threads": int(
                        args.xgb_threads
                    ),
                    "outer_folds": int(
                        args.outer_folds
                    ),
                    "inner_es_fraction": float(
                        args.inner_es_fraction
                    ),
                    "num_boost_round": int(
                        args.num_boost_round
                    ),
                    "early_stopping_rounds": int(
                        args.early_stopping_rounds
                    ),
                    "seed": int(
                        args.seed
                    ),
                }
            )

    logger.info(
        "Tuning landmark: %dh",
        args.tuning_landmark,
    )
    logger.info(
        "Studies: %d | target trials/study=%d | workers=%d | "
        "xgb_threads/study=%d",
        len(jobs),
        args.trials,
        args.workers,
        args.xgb_threads,
    )
    logger.info(
        "Objectives | LOS=minimize RMSE | mortality=maximize Average Precision"
    )
    logger.info(
        "MIMIC test/eICU are not loaded by the tuner."
    )

    detected_cpus = os.cpu_count()
    if (
        detected_cpus is not None
        and args.workers
        * args.xgb_threads
        > detected_cpus
    ):
        logger.warning(
            "workers*xgb_threads=%d exceeds detected CPUs=%d.",
            args.workers
            * args.xgb_threads,
            detected_cpus,
        )

    results = []

    if args.workers == 1:
        for i, job in enumerate(
            jobs,
            start=1,
        ):
            logger.info(
                "[%d/%d] tuning %s/%s",
                i,
                len(jobs),
                job["outcome"],
                job["variant"],
            )
            result = run_tuning_job(
                job
            )
            results.append(
                result
            )
            logger.info(
                "PASS %s/%s | best=%g | trials=%d",
                result["outcome"],
                result["variant"],
                result["best_value"],
                result["total_trials"],
            )
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers
        ) as pool:
            future_to_job = {
                pool.submit(
                    run_tuning_job,
                    job,
                ): job
                for job in jobs
            }

            for future in as_completed(
                future_to_job
            ):
                job = future_to_job[
                    future
                ]
                try:
                    result = future.result()
                except Exception:
                    logger.exception(
                        "FAILED tuning %s/%s",
                        job["outcome"],
                        job["variant"],
                    )
                    raise
                else:
                    results.append(
                        result
                    )
                    logger.info(
                        "PASS %s/%s | best=%g | trials=%d "
                        "(complete=%d pruned=%d)",
                        result["outcome"],
                        result["variant"],
                        result["best_value"],
                        result["total_trials"],
                        result["complete_trials"],
                        result["pruned_trials"],
                    )

    report_dir = modeling_root / "reports"
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = pd.DataFrame(
        results
    ).sort_values(
        [
            "outcome",
            "variant",
        ]
    )
    summary.to_csv(
        report_dir
        / "optuna_tuning_summary.csv",
        index=False,
    )

    master_manifest = {
        "tuning_landmark_hour": int(
            args.tuning_landmark
        ),
        "policy": (
            "one hyperparameter set per outcome×representation tuned only on "
            "MIMIC development at the anchor landmark, then frozen across "
            "all prediction landmarks"
        ),
        "objectives": {
            "los": "minimize mean five-fold RMSE",
            "mortality": (
                "maximize mean five-fold Average Precision"
            ),
        },
        "outer_fold_used_for_early_stopping": False,
        "cross_validation_unit": "patient_id",
        "identifiers_used_as_predictors": False,
        "mimic_test_used": False,
        "eicu_used": False,
        "target_trials_per_study": int(
            args.trials
        ),
        "studies": results,
    }
    (
        modeling_root
        / "hyperparameters"
        / "optuna_manifest.json"
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (
        modeling_root
        / "hyperparameters"
        / "optuna_manifest.json"
    ).write_text(
        json.dumps(
            master_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Optuna tuning complete. Frozen parameter files are ready for "
        "02_train_xgboost.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
