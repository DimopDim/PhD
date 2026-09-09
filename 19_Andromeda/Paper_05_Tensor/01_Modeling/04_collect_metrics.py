#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_collect_metrics.py

Collect per-task XGBoost metrics into master CSV tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List
import json

import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)


def main() -> int:
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
    args = p.parse_args()

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

    report_dir = modeling_root / "reports"
    training_summary_path = report_dir / "training_task_summary.csv"
    if not training_summary_path.is_file():
        raise FileNotFoundError(
            f"{training_summary_path}. Run 03_train_xgboost.py first."
        )
    training = pd.read_csv(training_summary_path)
    required = {"landmark_hour", "outcome", "variant", "status"}
    missing = required - set(training.columns)
    if missing:
        raise RuntimeError(
            f"Training summary missing columns: {sorted(missing)}"
        )
    if training.empty or not training["status"].eq("PASS").all():
        raise RuntimeError("Training summary is empty or contains non-PASS tasks.")
    keys = ["landmark_hour", "outcome", "variant"]
    if training.duplicated(keys).any():
        raise RuntimeError("Duplicate tasks in training_task_summary.csv.")

    metric_files = sorted(
        (
            modeling_root
            / "results"
        ).glob(
            "landmark_*h/*/*/metrics.csv"
        )
    )

    if not metric_files:
        raise FileNotFoundError(
            "No task metrics.csv files found."
        )

    if len(metric_files) != len(training):
        raise RuntimeError(
            f"Expected {len(training)} metrics.csv files from Stage 03, "
            f"found {len(metric_files)}."
        )

    frames: List[pd.DataFrame] = []
    for path in metric_files:
        frame = pd.read_csv(path)
        required_metric = {
            "landmark_hour", "outcome", "variant", "cohort", "n"
        }
        missing = required_metric - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{path} missing required columns: {sorted(missing)}"
            )
        if len(frame) != 3:
            raise RuntimeError(
                f"{path} must contain exactly 3 evaluated cohort rows; "
                f"found {len(frame)}."
            )
        expected_cohorts = {
            "mimic_train_oof", "mimic_test", "eicu_external"
        }
        observed = set(frame["cohort"].astype(str))
        if observed != expected_cohorts:
            raise RuntimeError(
                f"{path} cohort rows mismatch: {sorted(observed)}"
            )
        if frame.duplicated(["landmark_hour","outcome","variant","cohort"]).any():
            raise RuntimeError(f"Duplicate metric rows in {path}.")
        frames.append(frame)
    all_metrics = pd.concat(
        frames,
        ignore_index=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = all_metrics.sort_values(
        [
            "outcome",
            "landmark_hour",
            "variant",
            "cohort",
        ]
    )
    all_metrics.to_csv(
        report_dir
        / "all_metrics.csv",
        index=False,
    )

    all_metrics.loc[
        all_metrics[
            "outcome"
        ].eq("los")
    ].to_csv(
        report_dir
        / "los_metrics.csv",
        index=False,
    )

    all_metrics.loc[
        all_metrics[
            "outcome"
        ].eq(
            "mortality"
        )
    ].to_csv(
        report_dir
        / "mortality_metrics.csv",
        index=False,
    )

    observed_tasks = (
        all_metrics[["landmark_hour", "outcome", "variant"]]
        .drop_duplicates()
        .sort_values(["landmark_hour", "outcome", "variant"])
        .reset_index(drop=True)
    )
    expected_tasks = (
        training[["landmark_hour", "outcome", "variant"]]
        .sort_values(["landmark_hour", "outcome", "variant"])
        .reset_index(drop=True)
    )
    if not observed_tasks.equals(expected_tasks):
        raise RuntimeError(
            "Collected metric task coverage does not match Stage-03 training summary."
        )

    audit = {
        "training_tasks": int(len(training)),
        "metric_files": int(len(metric_files)),
        "aggregate_metric_rows": int(len(all_metrics)),
        "expected_metric_rows": int(3 * len(training)),
        "all_training_tasks_pass": True,
        "task_coverage_matches_training_summary": True,
        "status": "PASS",
    }
    (report_dir / "metrics_collection_manifest.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print(
        f"PASS: collected {len(metric_files)} task metric files "
        f"into {len(all_metrics)} rows."
    )
    print(
        report_dir
        / "all_metrics.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
