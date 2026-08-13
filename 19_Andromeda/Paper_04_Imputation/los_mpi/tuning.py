from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, ParameterSampler
from xgboost import XGBRegressor

from .config import Config
from .data import DataBundle
from .preprocessing import LeakageSafePreprocessor


DEFAULT_PARAMETERS = {
    "n_estimators": 100,
    "learning_rate": 0.3,
    "max_depth": 6,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}


@dataclass
class FoldMatrix:
    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    train_groups: set[str]
    validation_groups: set[str]


@dataclass
class SearchResult:
    mode: str
    best_parameters: dict[str, int | float]
    best_score: float | None
    trials: pd.DataFrame


def build_model(parameters: dict, config: Config, seed: int) -> XGBRegressor:
    clean = {
        "n_estimators": int(parameters["n_estimators"]),
        "learning_rate": float(parameters["learning_rate"]),
        "max_depth": int(parameters["max_depth"]),
        "reg_alpha": float(parameters["reg_alpha"]),
        "reg_lambda": float(parameters["reg_lambda"]),
    }
    return XGBRegressor(
        **clean,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        device=config.experiment.device if config.experiment.device.startswith("cuda") else "cpu",
        n_jobs=config.experiment.threads_per_rank,
        random_state=seed,
        verbosity=0,
    )


def score_parameters(
    parameters: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: Config,
    seed: int,
) -> float:
    model = build_model(parameters, config, seed)
    model.fit(X_train, y_train)
    return float(mean_squared_error(y_validation, model.predict(X_validation)))


def random_space() -> dict[str, list[int | float]]:
    return {
        "n_estimators": [100, 200, 300],
        "learning_rate": [round(0.01 + 0.20 * index, 2) for index in range(6)],
        "max_depth": list(range(1, 10)),
        "reg_alpha": [round(0.1 + index, 1) for index in range(15)],
        "reg_lambda": [round(0.1 + index, 1) for index in range(15)],
    }


def randomized_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: Config,
    seed: int,
) -> SearchResult:
    candidates = list(
        ParameterSampler(
            random_space(),
            n_iter=config.experiment.search_iterations,
            random_state=seed,
        )
    )
    records = []
    for index, parameters in enumerate(candidates):
        score = score_parameters(
            parameters, X_train, y_train, X_validation, y_validation, config, seed + index
        )
        records.append({"trial": index, "validation_mse": score, **parameters})
    trials = pd.DataFrame(records).sort_values("validation_mse", kind="stable")
    best = trials.iloc[0]
    parameters = {key: best[key] for key in random_space()}
    return SearchResult("randomized", _clean_parameters(parameters), float(best.validation_mse), trials)


def grouped_fold_matrices(
    bundle: DataBundle,
    method: str,
    scale: bool,
    config: Config,
    seed: int,
) -> list[FoldMatrix]:
    train = bundle.train
    splitter = GroupKFold(n_splits=config.experiment.bayesian_folds)
    folds: list[FoldMatrix] = []
    # Shuffle patient groups deterministically before GroupKFold assignment.
    unique_groups = np.unique(train.split_groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)
    group_rank = {group: index for index, group in enumerate(shuffled)}
    ordered_groups = np.asarray([group_rank[group] for group in train.split_groups])
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(train.X, train.y, groups=ordered_groups)
    ):
        fold_train = train.take(train_index, f"mimic_train_fold{fold_index}_fit")
        fold_validation = train.take(validation_index, f"mimic_train_fold{fold_index}_validation")
        train_groups = set(fold_train.split_groups)
        validation_groups = set(fold_validation.split_groups)
        if train_groups & validation_groups:
            raise RuntimeError(f"Grouped CV leakage in fold {fold_index}")
        preprocessor = LeakageSafePreprocessor(
            method, scale, config, seed + 10_000 + fold_index, config.experiment.threads_per_rank
        )
        X_fold_train = preprocessor.fit_transform(fold_train)
        X_fold_validation = preprocessor.transform(fold_validation)
        if preprocessor.fit_scope_["row_id_sha256"] == "":
            raise RuntimeError("Invalid preprocessing scope audit")
        folds.append(
            FoldMatrix(
                X_fold_train,
                fold_train.y,
                X_fold_validation,
                fold_validation.y,
                train_groups,
                validation_groups,
            )
        )
    return folds


def _grouped_cv_score(parameters: dict, folds: list[FoldMatrix], config: Config, seed: int) -> float:
    scores = []
    for fold_index, fold in enumerate(folds):
        if fold.train_groups & fold.validation_groups:
            raise RuntimeError("Patient overlap detected while scoring grouped CV")
        scores.append(
            score_parameters(
                parameters,
                fold.X_train,
                fold.y_train,
                fold.X_validation,
                fold.y_validation,
                config,
                seed + fold_index,
            )
        )
    return float(np.mean(scores))


def bayesian_search(folds: list[FoldMatrix], config: Config, seed: int) -> SearchResult:
    from skopt import Optimizer
    from skopt.space import Integer, Real

    dimensions = [
        Integer(100, 300, name="n_estimators"),
        Real(0.01, 1.0, prior="log-uniform", name="learning_rate"),
        Integer(1, 10, name="max_depth"),
        Real(0.1, 15.0, name="reg_alpha"),
        Real(0.1, 15.0, name="reg_lambda"),
    ]
    optimizer = Optimizer(dimensions, random_state=seed, base_estimator="GP", acq_func="gp_hedge")
    names = [dimension.name for dimension in dimensions]
    records = []
    for trial in range(config.experiment.search_iterations):
        point = optimizer.ask()
        parameters = dict(zip(names, point))
        score = _grouped_cv_score(parameters, folds, config, seed + trial * 100)
        optimizer.tell(point, score)
        records.append({"trial": trial, "grouped_cv_mse": score, **parameters})
    trials = pd.DataFrame(records).sort_values("grouped_cv_mse", kind="stable")
    best = trials.iloc[0]
    parameters = {name: best[name] for name in names}
    return SearchResult("bayesian", _clean_parameters(parameters), float(best.grouped_cv_mse), trials)


def hyperopt_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    config: Config,
    seed: int,
) -> SearchResult:
    from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 300, 50),
        "learning_rate": hp.loguniform("learning_rate", math.log(0.01), math.log(1.0)),
        "max_depth": hp.quniform("max_depth", 1, 10, 1),
        "reg_alpha": hp.uniform("reg_alpha", 0.1, 15.0),
        "reg_lambda": hp.uniform("reg_lambda", 0.1, 15.0),
    }
    records: list[dict] = []

    def objective(sample: dict) -> dict:
        parameters = _clean_parameters(sample)
        trial = len(records)
        score = score_parameters(
            parameters, X_train, y_train, X_validation, y_validation, config, seed + trial
        )
        records.append({"trial": trial, "validation_mse": score, **parameters})
        return {"loss": score, "status": STATUS_OK}

    trials_object = Trials()
    best_raw = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=config.experiment.search_iterations,
        trials=trials_object,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )
    best_parameters = _clean_parameters(best_raw)
    trials = pd.DataFrame(records).sort_values("validation_mse", kind="stable")
    return SearchResult(
        "hyperopt", best_parameters, float(trials.iloc[0].validation_mse), trials
    )


def no_search() -> SearchResult:
    return SearchResult("none", dict(DEFAULT_PARAMETERS), None, pd.DataFrame([{**DEFAULT_PARAMETERS}]))


def _clean_parameters(parameters: dict) -> dict[str, int | float]:
    return {
        "n_estimators": int(round(float(parameters["n_estimators"]))),
        "learning_rate": float(parameters["learning_rate"]),
        "max_depth": int(round(float(parameters["max_depth"]))),
        "reg_alpha": float(parameters["reg_alpha"]),
        "reg_lambda": float(parameters["reg_lambda"]),
    }

