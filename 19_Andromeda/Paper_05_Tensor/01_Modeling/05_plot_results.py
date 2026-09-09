#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_plot_results.py

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

Summary-figure visual encoding:
- colour = temporal representation
- solid line = MIMIC-IV internal test
- dashed line = eICU-CRD external evaluation
- marker shape = temporal representation

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
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    """Save publication-quality raster plus vector companion."""
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
    if path.suffix.lower() == ".png":
        fig.savefig(
            path.with_suffix(".pdf"),
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
    """
    Plot landmark trajectories with a consistent visual grammar.

    Encoding:
    - colour = representation
    - line style = evaluation cohort/database
      * MIMIC internal test: solid
      * eICU external: dashed
    - marker shape = representation

    This keeps the same representation visually identifiable across databases
    and makes internal versus external evaluation immediately distinguishable.
    """
    data = metrics.loc[
        metrics["outcome"].eq(outcome)
        & metrics["cohort"].isin(
            ["mimic_test", "eicu_external"]
        )
    ].copy()

    if data.empty:
        raise RuntimeError(
            f"No internal/external rows for outcome={outcome}, metric={metric}."
        )
    if metric not in data.columns:
        raise RuntimeError(f"Missing metric column: {metric}")
    if data[metric].isna().any():
        bad = data.loc[
            data[metric].isna(),
            ["landmark_hour", "variant", "cohort"],
        ]
        raise RuntimeError(
            f"NaN values in {metric}:\n{bad.to_string(index=False)}"
        )

    preferred_variant_order = [
        "static",
        "o1",
        "o2",
        "o3",
        "o4",
        "full",
    ]
    variants = [
        v
        for v in preferred_variant_order
        if v in set(data["variant"].astype(str))
    ]
    extras = sorted(
        set(data["variant"].astype(str)) - set(variants)
    )
    variants.extend(extras)

    # Use Matplotlib's active default colour cycle, but bind colours
    # deterministically to representations so that the same representation
    # has the same colour in MIMIC and eICU.
    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key().get("color", [])
    if len(default_colors) < len(variants):
        raise RuntimeError(
            "Active Matplotlib colour cycle has too few colours "
            f"for {len(variants)} representations."
        )
    variant_color = {
        variant: default_colors[i]
        for i, variant in enumerate(variants)
    }

    marker_cycle = ["o", "s", "^", "D", "P", "X"]
    variant_marker = {
        variant: marker_cycle[i % len(marker_cycle)]
        for i, variant in enumerate(variants)
    }

    cohort_style = {
        "mimic_test": "-",
        "eicu_external": "--",
    }
    cohort_label = {
        "mimic_test": "MIMIC-IV internal test",
        "eicu_external": "eICU-CRD external",
    }

    fig, ax = plt.subplots(
        figsize=(8.6, 5.4)
    )

    for cohort in ["mimic_test", "eicu_external"]:
        for variant in variants:
            group = data.loc[
                data["cohort"].eq(cohort)
                & data["variant"].eq(variant)
            ].sort_values("landmark_hour")

            if group.empty:
                continue

            ax.plot(
                group["landmark_hour"],
                group[metric],
                color=variant_color[variant],
                linestyle=cohort_style[cohort],
                marker=variant_marker[variant],
                markersize=5.5,
                markeredgewidth=0.8,
                linewidth=1.8,
                alpha=0.95,
            )

    ax.set_xlabel(
        "Landmark time (h)",
        fontsize=11,
    )
    ax.set_ylabel(
        ylabel,
        fontsize=11,
    )
    ax.set_xticks(
        sorted(data["landmark_hour"].unique())
    )
    ax.tick_params(
        axis="both",
        labelsize=9.5,
    )

    # Light horizontal guide only; avoid visually competing with trajectories.
    ax.grid(
        axis="y",
        alpha=0.20,
        linewidth=0.8,
    )
    ax.grid(
        axis="x",
        visible=False,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    representation_handles = [
        Line2D(
            [0],
            [0],
            color=variant_color[v],
            linestyle="-",
            marker=variant_marker[v],
            linewidth=1.8,
            markersize=5.5,
            label=v,
        )
        for v in variants
    ]
    cohort_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=cohort_style[c],
            linewidth=1.8,
            label=cohort_label[c],
        )
        for c in ["mimic_test", "eicu_external"]
    ]

    representation_legend = ax.legend(
        handles=representation_handles,
        title="Representation",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        borderaxespad=0.0,
        frameon=False,
        fontsize=8.8,
        title_fontsize=9.2,
    )
    ax.add_artist(
        representation_legend
    )

    ax.legend(
        handles=cohort_handles,
        title="Evaluation cohort",
        loc="lower left",
        bbox_to_anchor=(1.01, 0.00),
        borderaxespad=0.0,
        frameon=False,
        fontsize=8.8,
        title_fontsize=9.2,
    )

    fig.subplots_adjust(
        right=0.74
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
            f"{metrics_path}. Run 04_collect_metrics.py first."
        )

    metrics = pd.read_csv(
        metrics_path
    )

    required_columns = {
        "outcome", "landmark_hour", "variant", "cohort", "n"
    }
    missing = required_columns - set(metrics.columns)
    if missing:
        raise RuntimeError(
            f"all_metrics.csv missing required columns: {sorted(missing)}"
        )
    if metrics.empty:
        raise RuntimeError("all_metrics.csv is empty.")
    if metrics.duplicated(
        ["outcome", "landmark_hour", "variant", "cohort"]
    ).any():
        raise RuntimeError("Duplicate task/cohort rows in all_metrics.csv.")

    collection_manifest_path = (
        modeling_root / "reports" / "metrics_collection_manifest.json"
    )
    if collection_manifest_path.is_file():
        collection_manifest = json.loads(
            collection_manifest_path.read_text(encoding="utf-8")
        )
        if collection_manifest.get("status") != "PASS":
            raise RuntimeError(
                "Metric collection manifest is not PASS."
            )
        expected_rows = int(
            collection_manifest.get("aggregate_metric_rows", -1)
        )
        if expected_rows != len(metrics):
            raise RuntimeError(
                f"Metric row-count mismatch: manifest={expected_rows}, "
                f"all_metrics.csv={len(metrics)}."
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

    generated_summary = []
    for outcome, metric, ylabel, filename in global_specs:
        if metric not in metrics.columns:
            raise RuntimeError(
                f"Required plotting metric missing from all_metrics.csv: {metric}"
            )
        output_path = figure_root / "summary" / filename
        plot_metric_vs_landmark(
            metrics,
            outcome=outcome,
            metric=metric,
            ylabel=ylabel,
            output_path=output_path,
        )
        generated_summary.append(str(output_path))

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

    plot_manifest = {
        "source_metrics": str(metrics_path),
        "metric_rows": int(len(metrics)),
        "summary_figures": generated_summary,
        "task_plots_requested": bool(args.task_plots),
        "status": "PASS",
    }
    figure_root.mkdir(parents=True, exist_ok=True)
    (figure_root / "05_plot_manifest.json").write_text(
        json.dumps(plot_manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"PASS: figures written under {figure_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
