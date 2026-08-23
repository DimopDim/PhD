#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_collect_metrics.py

Collect per-task XGBoost metrics into master CSV tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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

    frames: List[pd.DataFrame] = [
        pd.read_csv(path)
        for path in metric_files
    ]
    all_metrics = pd.concat(
        frames,
        ignore_index=True,
    )

    report_dir = (
        modeling_root
        / "reports"
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

    print(
        f"Collected {len(metric_files)} task metric files."
    )
    print(
        report_dir
        / "all_metrics.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
