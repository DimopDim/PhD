#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_plot_results.py

Generate publication-oriented XGBoost evaluation figures.

Figures
-------
Global:
- LOS MAE vs landmark
- LOS RMSE vs landmark
- LOS R2 vs landmark
- mortality ROC-AUC vs landmark
- mortality Average Precision vs landmark
- mortality Brier score vs landmark

Per task/cohort:
- LOS observed vs predicted
- LOS residuals vs predicted
- mortality ROC curve
- mortality Precision-Recall curve
- mortality calibration curve

Train curves are intentionally not plotted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)


def save_figure(
    fig,
    path: Path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_metric_vs_landmark(
    metrics: pd.DataFrame,
    *,
    outcome: str,
    metric: str,
    ylabel: str,
    output_path: Path,
):
    data = metrics.loc[
        metrics[
            "outcome"
        ].eq(outcome)
        & metrics[
            "cohort"
        ].isin(
            [
                "mimic_test",
                "eicu_external",
            ]
        )
    ].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    for (
        cohort,
        variant,
    ), group in data.groupby(
        [
            "cohort",
            "variant",
        ]
    ):
        group = group.sort_values(
            "landmark_hour"
        )
        ax.plot(
            group[
                "landmark_hour"
            ],
            group[
                metric
            ],
            marker="o",
            label=(
                f"{variant} | "
                f"{cohort.replace('_', ' ')}"
            ),
        )

    ax.set_xlabel(
        "Landmark hour"
    )
    ax.set_ylabel(
        ylabel
    )
    ax.set_xticks(
        sorted(
            data[
                "landmark_hour"
            ].unique()
        )
    )
    ax.grid(
        alpha=0.25
    )
    ax.legend(
        fontsize=8,
        ncol=2,
    )

    save_figure(
        fig,
        output_path,
    )


def load_predictions(
    task_dir: Path,
    cohort: str,
) -> pd.DataFrame:
    path = (
        task_dir
        / f"predictions_{cohort}.parquet"
    )
    if not path.is_file():
        raise FileNotFoundError(
            path
        )
    return pd.read_parquet(
        path
    )


def plot_los_task(
    pred: pd.DataFrame,
    *,
    landmark: int,
    variant: str,
    cohort: str,
    output_dir: Path,
):
    y = pred[
        "y_true"
    ].to_numpy()
    p = pred[
        "prediction"
    ].to_numpy()

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )
    ax.scatter(
        y,
        p,
        s=12,
        alpha=0.45,
    )

    minimum = float(
        min(
            np.min(y),
            np.min(p),
        )
    )
    maximum = float(
        max(
            np.max(y),
            np.max(p),
        )
    )
    ax.plot(
        [
            minimum,
            maximum,
        ],
        [
            minimum,
            maximum,
        ],
        linestyle="--",
    )
    ax.set_xlabel(
        "Observed remaining ICU LOS (days)"
    )
    ax.set_ylabel(
        "Predicted remaining ICU LOS (days)"
    )
    ax.set_title(
        f"{landmark}h | {variant} | {cohort.replace('_', ' ')}"
    )
    ax.grid(
        alpha=0.25
    )
    save_figure(
        fig,
        output_dir
        / "observed_vs_predicted.png",
    )

    residual = y - p
    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
    ax.scatter(
        p,
        residual,
        s=12,
        alpha=0.45,
    )
    ax.axhline(
        0.0,
        linestyle="--",
    )
    ax.set_xlabel(
        "Predicted remaining ICU LOS (days)"
    )
    ax.set_ylabel(
        "Residual: observed - predicted (days)"
    )
    ax.set_title(
        f"{landmark}h | {variant} | {cohort.replace('_', ' ')}"
    )
    ax.grid(
        alpha=0.25
    )
    save_figure(
        fig,
        output_dir
        / "residuals.png",
    )


def plot_mortality_task(
    pred: pd.DataFrame,
    *,
    landmark: int,
    variant: str,
    cohort: str,
    output_dir: Path,
):
    y = pred[
        "y_true"
    ].to_numpy(
        dtype=int
    )
    p = pred[
        "prediction_calibrated"
    ].to_numpy(
        dtype=float
    )

    if len(
        np.unique(
            y
        )
    ) < 2:
        return

    fpr, tpr, _ = roc_curve(
        y,
        p,
    )
    auc = roc_auc_score(
        y,
        p,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
    ax.plot(
        fpr,
        tpr,
        label=f"AUROC={auc:.3f}",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    ax.set_xlabel(
        "False positive rate"
    )
    ax.set_ylabel(
        "True positive rate"
    )
    ax.set_title(
        f"{landmark}h | {variant} | {cohort.replace('_', ' ')}"
    )
    ax.legend()
    ax.grid(
        alpha=0.25
    )
    save_figure(
        fig,
        output_dir
        / "roc_curve.png",
    )

    precision, recall, _ = (
        precision_recall_curve(
            y,
            p,
        )
    )
    ap = average_precision_score(
        y,
        p,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
    ax.plot(
        recall,
        precision,
        label=f"AP={ap:.3f}",
    )
    ax.axhline(
        np.mean(y),
        linestyle="--",
        label=(
            f"Prevalence={np.mean(y):.3f}"
        ),
    )
    ax.set_xlabel(
        "Recall"
    )
    ax.set_ylabel(
        "Precision"
    )
    ax.set_title(
        f"{landmark}h | {variant} | {cohort.replace('_', ' ')}"
    )
    ax.legend()
    ax.grid(
        alpha=0.25
    )
    save_figure(
        fig,
        output_dir
        / "pr_curve.png",
    )

    prob_true, prob_pred = (
        calibration_curve(
            y,
            p,
            n_bins=10,
            strategy="quantile",
        )
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )
    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )
    ax.set_xlabel(
        "Mean predicted probability"
    )
    ax.set_ylabel(
        "Observed event frequency"
    )
    ax.set_title(
        f"{landmark}h | {variant} | {cohort.replace('_', ' ')}"
    )
    ax.grid(
        alpha=0.25
    )
    save_figure(
        fig,
        output_dir
        / "calibration_curve.png",
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
    p.add_argument(
        "--task-plots",
        action="store_true",
        help=(
            "Also generate per-task ROC/PR/calibration and LOS scatter plots."
        ),
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

    metrics_path = (
        modeling_root
        / "reports"
        / "all_metrics.csv"
    )
    if not metrics_path.is_file():
        raise FileNotFoundError(
            f"{metrics_path}. Run 03_collect_metrics.py first."
        )

    metrics = pd.read_csv(
        metrics_path
    )

    figure_root = (
        modeling_root
        / "figures"
    )

    global_specs = [
        (
            "los",
            "mae",
            "MAE (days)",
            "los_mae_vs_landmark.png",
        ),
        (
            "los",
            "rmse",
            "RMSE (days)",
            "los_rmse_vs_landmark.png",
        ),
        (
            "los",
            "r2",
            "R²",
            "los_r2_vs_landmark.png",
        ),
        (
            "mortality",
            "roc_auc_calibrated",
            "AUROC",
            "mortality_auroc_vs_landmark.png",
        ),
        (
            "mortality",
            "average_precision_calibrated",
            "Average Precision",
            "mortality_ap_vs_landmark.png",
        ),
        (
            "mortality",
            "brier_calibrated",
            "Brier score",
            "mortality_brier_vs_landmark.png",
        ),
    ]

    for outcome, metric, ylabel, filename in global_specs:
        if metric in metrics.columns:
            plot_metric_vs_landmark(
                metrics,
                outcome=outcome,
                metric=metric,
                ylabel=ylabel,
                output_path=(
                    figure_root
                    / "summary"
                    / filename
                ),
            )

    if args.task_plots:
        task_metrics = metrics.loc[
            metrics[
                "cohort"
            ].isin(
                [
                    "mimic_test",
                    "eicu_external",
                ]
            )
        ]

        for row in task_metrics.itertuples(
            index=False
        ):
            task = (
                modeling_root
                / "results"
                / f"landmark_{int(row.landmark_hour):03d}h"
                / row.outcome
                / row.variant
            )

            pred = load_predictions(
                task,
                row.cohort,
            )
            output = (
                figure_root
                / f"landmark_{int(row.landmark_hour):03d}h"
                / row.outcome
                / row.variant
                / row.cohort
            )

            if row.outcome == "los":
                plot_los_task(
                    pred,
                    landmark=int(
                        row.landmark_hour
                    ),
                    variant=row.variant,
                    cohort=row.cohort,
                    output_dir=output,
                )
            else:
                plot_mortality_task(
                    pred,
                    landmark=int(
                        row.landmark_hour
                    ),
                    variant=row.variant,
                    cohort=row.cohort,
                    output_dir=output,
                )

    print(
        f"Figures written under {figure_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
