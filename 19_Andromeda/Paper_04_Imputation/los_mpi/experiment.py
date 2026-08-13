from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import Config
from .data import DataBundle, FrameSet
from .metrics import (
    admission_aggregations,
    cluster_bootstrap_mae,
    prediction_frame,
    regression_metrics,
)
from .preprocessing import LeakageSafePreprocessor
from .tuning import (
    SearchResult,
    bayesian_search,
    build_model,
    grouped_fold_matrices,
    hyperopt_search,
    no_search,
    randomized_search,
)


IMPUTATION_METHODS = ("native", "mean", "interpolation", "mlp", "rnn", "regression")
SCALING_OPTIONS = (False, True)
TUNING_MODES = ("none", "randomized", "bayesian", "hyperopt")


@dataclass(frozen=True)
class BundleTask:
    index: int
    imputation: str
    scale: bool

    @property
    def bundle_id(self) -> str:
        return f"preprocess__{self.imputation}__scale-{'yes' if self.scale else 'no'}"

    def config_id(self, tuning: str) -> str:
        return f"{self.imputation}__scale-{'yes' if self.scale else 'no'}__{tuning}"


def all_bundle_tasks() -> list[BundleTask]:
    return [
        BundleTask(index=index, imputation=method, scale=scale)
        for index, (method, scale) in enumerate(
            (pair for method in IMPUTATION_METHODS for pair in ((method, False), (method, True)))
        )
    ]


def _write_json_atomic(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _hash_rows(frame: FrameSet) -> str:
    digest = hashlib.sha256()
    for row_id in frame.row_ids:
        digest.update(str(row_id).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _search(
    mode: str,
    matrices: dict[str, np.ndarray],
    bundle: DataBundle,
    folds,
    config: Config,
    seed: int,
) -> SearchResult:
    if mode == "none":
        return no_search()
    if mode == "randomized":
        return randomized_search(
            matrices["train"], bundle.train.y,
            matrices["validation"], bundle.validation.y,
            config, seed,
        )
    if mode == "bayesian":
        return bayesian_search(folds, config, seed)
    if mode == "hyperopt":
        return hyperopt_search(
            matrices["train"], bundle.train.y,
            matrices["validation"], bundle.validation.y,
            config, seed,
        )
    raise ValueError(mode)


def _evaluate_set(
    config_id: str,
    frame: FrameSet,
    predictions: np.ndarray,
    output_dir: Path,
    bootstrap_iterations: int,
    seed: int,
) -> dict:
    table = prediction_frame(
        frame.name,
        frame.y,
        predictions,
        frame.sequence_groups,
        frame.split_groups,
        frame.window_order,
        frame.row_ids,
    )
    table.to_parquet(output_dir / f"predictions_{frame.name}.parquet", index=False)
    metrics = regression_metrics(frame.y, predictions)
    metrics.update(
        cluster_bootstrap_mae(
            frame.y,
            predictions,
            frame.sequence_groups,
            bootstrap_iterations,
            seed,
        )
    )
    metrics["n_admissions"] = int(pd.Series(frame.sequence_groups).nunique())
    metrics["aggregation_sensitivity"] = admission_aggregations(table)
    return metrics


def run_bundle_task(task: BundleTask, bundle: DataBundle, config: Config, rank: int) -> dict:
    started = time.time()
    root = Path(config.experiment.output_dir).expanduser().resolve()
    bundle_dir = root / "bundles" / task.bundle_id
    configuration_root = root / "configurations"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pending_modes = [
        mode
        for mode in TUNING_MODES
        if not (
            config.experiment.resume
            and (configuration_root / task.config_id(mode) / "DONE.json").exists()
        )
    ]
    if not pending_modes:
        return {"task": task.bundle_id, "status": "skipped", "rank": rank, "seconds": 0.0}

    seed = config.split.seed + task.index * 100_000
    preprocessor = LeakageSafePreprocessor(
        task.imputation,
        task.scale,
        config,
        seed,
        config.experiment.threads_per_rank,
    )
    matrices = {
        "train": preprocessor.fit_transform(bundle.train),
        "validation": preprocessor.transform(bundle.validation),
        "test": preprocessor.transform(bundle.test),
        "external": preprocessor.transform(bundle.external),
    }
    if preprocessor.fit_scope_["dataset"] != "mimic_train":
        raise RuntimeError(f"Preprocessor fit outside MIMIC train: {preprocessor.fit_scope_}")
    joblib.dump(preprocessor, bundle_dir / "preprocessor.joblib", compress=3)
    _write_json_atomic(
        bundle_dir / "preprocessing_audit.json",
        {
            "bundle_id": task.bundle_id,
            "fit_scope": preprocessor.fit_scope_,
            "fit_only_on_mimic_train": True,
            "validation_transform_only": True,
            "test_transform_only": True,
            "external_transform_only": True,
            "output_features": preprocessor.feature_names_out_,
            "scaling_order": ["StandardScaler", "MinMaxScaler"] if task.scale else [],
            "interpolation_mode": (
                "causal within-admission interpolation/extrapolation"
                if task.imputation == "interpolation" else None
            ),
        },
    )

    folds = None
    if "bayesian" in pending_modes:
        folds = grouped_fold_matrices(bundle, task.imputation, task.scale, config, seed + 50_000)

    completed = []
    for mode_index, mode in enumerate(pending_modes):
        config_id = task.config_id(mode)
        output_dir = configuration_root / config_id
        output_dir.mkdir(parents=True, exist_ok=True)
        mode_seed = seed + 1_000 * (mode_index + 1)
        search = _search(mode, matrices, bundle, folds, config, mode_seed)
        search.trials.to_csv(output_dir / "search_trials.csv", index=False)
        model = build_model(search.best_parameters, config, mode_seed)
        model.fit(matrices["train"], bundle.train.y)
        model.save_model(output_dir / "model.json")

        metrics = {}
        for offset, (name, frame) in enumerate(
            (("validation", bundle.validation), ("test", bundle.test), ("external", bundle.external))
        ):
            prediction = model.predict(matrices[name])
            metrics[frame.name] = _evaluate_set(
                config_id,
                frame,
                prediction,
                output_dir,
                config.experiment.bootstrap_iterations,
                mode_seed + 100 + offset,
            )
        train_mean = float(np.mean(bundle.train.y))
        metrics["training_mean_baseline"] = {
            frame.name: regression_metrics(frame.y, np.full_like(frame.y, train_mean))
            for frame in (bundle.validation, bundle.test, bundle.external)
        }
        metadata = {
            "config_id": config_id,
            "imputation": task.imputation,
            "scaling": task.scale,
            "tuning": mode,
            "best_parameters": search.best_parameters,
            "selection_score_mse": search.best_score,
            "selection_data": (
                "none"
                if mode == "none"
                else "MIMIC validation"
                if mode in {"randomized", "hyperopt"}
                else f"MIMIC train {config.experiment.bayesian_folds}-fold patient-grouped CV"
            ),
            "model_fit_scope": {
                "dataset": "mimic_train",
                "rows": len(bundle.train.y),
                "row_id_sha256": _hash_rows(bundle.train),
            },
            "external_outcomes_used_for_training_or_selection": False,
            "metrics": metrics,
        }
        _write_json_atomic(output_dir / "result.json", metadata)
        _write_json_atomic(
            output_dir / "DONE.json",
            {"config_id": config_id, "rank": rank, "completed_at_unix": time.time()},
        )
        completed.append(config_id)
    return {
        "task": task.bundle_id,
        "status": "completed",
        "rank": rank,
        "configurations": completed,
        "seconds": time.time() - started,
    }


def collect_results(config: Config) -> pd.DataFrame:
    root = Path(config.experiment.output_dir).expanduser().resolve()
    records = []
    for result_path in sorted((root / "configurations").glob("*/result.json")):
        value = json.loads(result_path.read_text(encoding="utf-8"))
        base = {
            "config_id": value["config_id"],
            "imputation": value["imputation"],
            "scaling": value["scaling"],
            "tuning": value["tuning"],
            **{f"param_{key}": val for key, val in value["best_parameters"].items()},
        }
        for dataset, metrics in value["metrics"].items():
            if dataset == "training_mean_baseline":
                continue
            record = dict(base)
            record["dataset"] = dataset
            for key, val in metrics.items():
                if key != "aggregation_sensitivity":
                    record[key] = val
            records.append(record)
    table = pd.DataFrame(records)
    if not table.empty:
        table.to_csv(root / "all_metrics_long.csv", index=False)
        test_external = table[table.dataset.isin(["mimic_test", "eicu_external"])]
        wide = test_external.pivot(index=["config_id", "imputation", "scaling", "tuning"], columns="dataset")
        wide.columns = [f"{metric}__{dataset}" for metric, dataset in wide.columns]
        wide.reset_index().to_csv(root / "all_metrics_internal_external.csv", index=False)
    return table


def environment_provenance(config: Config, mpi_size: int) -> dict:
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except OSError:
        freeze = []
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "mpi_size": mpi_size,
        "threads_per_rank": config.experiment.threads_per_rank,
        "config": config.as_dict(),
        "packages": freeze,
    }


def safe_task(task: BundleTask, bundle: DataBundle, config: Config, rank: int) -> dict:
    try:
        return run_bundle_task(task, bundle, config, rank)
    except Exception as exc:  # task-level containment so the scheduler can report every failure
        return {
            "task": task.bundle_id,
            "status": "failed",
            "rank": rank,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

