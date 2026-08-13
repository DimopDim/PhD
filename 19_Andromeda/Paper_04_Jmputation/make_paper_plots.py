#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Configuration
# =========================================================

IMPUTATION_ORDER = [
    "native",
    "mean",
    "interpolation",
    "mlp",
    "rnn",
    "regression",
]

IMPUTATION_LABEL = {
    "native": "No",
    "mean": "Mean",
    "interpolation": "Inter",
    "mlp": "MLP",
    "rnn": "RNN",
    "regression": "Reg",
}

TUNING_ORDER = ["none", "randomized", "bayesian", "hyperopt"]

TUNING_LABEL = {
    "none": "No",
    "randomized": "Rscv",
    "bayesian": "Bayes",
    "hyperopt": "Hopt",
}

SCALING_LABEL = {
    False: "No",
    True: "Yes",
}

PLOT_COLORS = {
    "native": "#5470C6",
    "mean": "#91B7E8",
    "interpolation": "#C7D9ED",
    "mlp": "#E6D5C9",
    "rnn": "#E5A07A",
    "regression": "#CC6F5A",
}


# =========================================================
# Helpers
# =========================================================

def normalise_scaling_column(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
    }
    return s.map(mapping)


def prepare_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv).copy()

    # Normalise column types
    df["scaling"] = normalise_scaling_column(df["scaling"])

    # Sorting keys
    df["imputation"] = pd.Categorical(
        df["imputation"], categories=IMPUTATION_ORDER, ordered=True
    )
    df["tuning"] = pd.Categorical(
        df["tuning"], categories=TUNING_ORDER, ordered=True
    )
    df["scaling"] = pd.Categorical(
        df["scaling"], categories=[False, True], ordered=True
    )

    # Display label for 48 full configurations
    df["config_label"] = (
        df["imputation"].map(IMPUTATION_LABEL).astype(str)
        + " | Scale/Norm "
        + df["scaling"].map(SCALING_LABEL).astype(str)
        + " | HP "
        + df["tuning"].map(TUNING_LABEL).astype(str)
    )

    df = df.sort_values(["imputation", "scaling", "tuning"]).reset_index(drop=True)
    return df


def remove_stat_suffix(name: str) -> str:
    """
    Remove trailing statistic suffixes:
    Albumin_(Mean)   -> Albumin
    pH_(Max)         -> pH
    X*(Median)       -> X
    """
    name = str(name)
    return re.sub(r'[\*_]?\((Min|Max|Mean|Median)\)$', '', name, flags=re.IGNORECASE)


def collapse_statistical_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group columns ending in Min/Max/Mean/Median into a single concept
    by row-wise mean, for the correlation matrix.
    """
    # Drop non-feature columns if present
    non_feature_cols = {
        "row_count",
        "subject_id",
        "hadm_id",
        "stay_id",
        "patientunitstayid",
        "Time_Zone",
        "los",
        "hospital_expire_flag",
        "mortality_30d",
        "mortality_180d",
        "mortality_360d",
        "window_index",
        "window_start_hour",
        "window_end_hour",
        "window_label",
    }

    df = feature_df.copy()
    existing_drop = [c for c in df.columns if c in non_feature_cols]
    df = df.drop(columns=existing_drop, errors="ignore")

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number]).copy()

    grouped = {}
    for col in df.columns:
        base = remove_stat_suffix(col)
        grouped.setdefault(base, []).append(col)

    collapsed = {}
    for base, cols in grouped.items():
        if len(cols) == 1:
            collapsed[base] = df[cols[0]]
        else:
            collapsed[base] = df[cols].mean(axis=1, skipna=True)

    collapsed_df = pd.DataFrame(collapsed)
    return collapsed_df


def annotate_bars(ax, bars, fontsize=8, rotation=0):
    for bar in bars:
        height = bar.get_height()
        if pd.isna(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


# =========================================================
# Plot 1: Comparison plot like example 1
# =========================================================

def plot_comparison_like(df: pd.DataFrame, outpath: str | Path):
    fig, ax_mse = plt.subplots(figsize=(11, 14))
    ax_mae = ax_mse.twiny()

    y = np.arange(len(df))

    # Bottom axis = MSE
    l3, = ax_mse.plot(
        df["mse__mimic_test"], y,
        marker="o", linestyle="-", color="green", label="Internal MSE"
    )
    l4, = ax_mse.plot(
        df["mse__eicu_external"], y,
        marker="x", linestyle="--", color="orange", label="External MSE"
    )

    # Top axis = MAE
    l1, = ax_mae.plot(
        df["mae__mimic_test"], y,
        marker="o", linestyle="-", color="blue", label="Internal MAE"
    )
    l2, = ax_mae.plot(
        df["mae__eicu_external"], y,
        marker="x", linestyle="--", color="red", label="External MAE"
    )

    ax_mse.set_yticks(y)
    ax_mse.set_yticklabels(df["config_label"], fontsize=9)
    ax_mse.invert_yaxis()

    ax_mse.set_xlabel("MSE", color="green", fontsize=12)
    ax_mae.set_xlabel("MAE", color="blue", fontsize=12)
    ax_mse.set_ylabel("Impute | Scale/Norm | HP", fontsize=12)

    ax_mse.tick_params(axis="x", colors="green")
    ax_mae.tick_params(axis="x", colors="blue")

    ax_mse.grid(True, axis="both", alpha=0.3)

    lines = [l1, l2, l3, l4]
    labels = [line.get_label() for line in lines]
    ax_mse.legend(lines, labels, loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Plot 2: 6-panel metrics grid like example 3
# =========================================================

def aggregate_over_scaling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over scaling (No/Yes) so we get:
    6 imputations x 4 tuning methods = 24 bars per metric
    """
    agg_cols = [
        "mse__mimic_test",
        "mse__eicu_external",
        "mae__mimic_test",
        "mae__eicu_external",
        "rmse__mimic_test",
        "rmse__eicu_external",
    ]
    out = (
        df.groupby(["imputation", "tuning"], observed=False)[agg_cols]
        .mean()
        .reset_index()
    )
    out["imputation"] = pd.Categorical(
        out["imputation"], categories=IMPUTATION_ORDER, ordered=True
    )
    out["tuning"] = pd.Categorical(
        out["tuning"], categories=TUNING_ORDER, ordered=True
    )
    return out.sort_values(["tuning", "imputation"]).reset_index(drop=True)


def plot_metrics_grid_like(df: pd.DataFrame, outpath: str | Path):
    agg = aggregate_over_scaling(df)

    fig, axes = plt.subplots(3, 2, figsize=(14, 17))
    axes = axes.flatten()

    panels = [
        ("mse__mimic_test", "Mimic MSE", "Internal MSE"),
        ("mse__eicu_external", "eICU MSE", "External MSE"),
        ("mae__mimic_test", "Mimic MAE", "Internal MAE"),
        ("mae__eicu_external", "eICU MAE", "External MAE"),
        ("rmse__mimic_test", "Mimic RMSE", "Internal RMSE"),
        ("rmse__eicu_external", "eICU RMSE", "External RMSE"),
    ]

    x = np.arange(len(TUNING_ORDER))
    n_imp = len(IMPUTATION_ORDER)
    bar_width = 0.12

    for ax, (metric_col, title, ylabel) in zip(axes, panels):
        for i, imp in enumerate(IMPUTATION_ORDER):
            imp_df = agg[agg["imputation"] == imp].set_index("tuning")
            vals = [imp_df.loc[t, metric_col] if t in imp_df.index else np.nan for t in TUNING_ORDER]

            positions = x + (i - (n_imp - 1) / 2) * bar_width
            bars = ax.bar(
                positions,
                vals,
                width=bar_width,
                label=IMPUTATION_LABEL[imp],
                color=PLOT_COLORS[imp],
                edgecolor="white",
                linewidth=0.5,
            )
            annotate_bars(ax, bars, fontsize=8)

        mean_value = np.nanmean(agg[metric_col])
        ax.axhline(
            mean_value,
            color="red",
            linestyle="--",
            linewidth=1.3,
            label=f"Mean {ylabel}: {mean_value:.2f}",
        )

        ax.set_title(title, fontsize=15, weight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Hyperparameter Tuning Method", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([TUNING_LABEL[t] for t in TUNING_ORDER], rotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Plot 3: Correlation matrix like example 2
# =========================================================

def plot_correlation_matrix_like(
    feature_csv: str | Path,
    outpath: str | Path,
    threshold: float = 0.5,
    max_features: int = 60,
):
    raw = pd.read_csv(feature_csv, low_memory=False)
    feat = collapse_statistical_features(raw)

    # Remove columns with too many missing or zero variance
    feat = feat.loc[:, feat.notna().sum() > 0]
    feat = feat.loc[:, feat.nunique(dropna=True) > 1]

    corr = feat.corr(method="pearson")
    corr_abs = corr.abs()

    # Keep only features that participate in at least one strong correlation
    strong_mask = corr_abs >= threshold
    np.fill_diagonal(strong_mask.values, False)

    keep = strong_mask.any(axis=0)
    corr_sub = corr.loc[keep, keep].copy()
    strong_counts = strong_mask.loc[keep, keep].sum(axis=0).sort_values(ascending=False)

    # Limit number of features so the plot stays readable
    if len(strong_counts) > max_features:
        selected = strong_counts.index[:max_features]
        corr_sub = corr_sub.loc[selected, selected]

    # Mask weak correlations
    plot_values = corr_sub.copy()
    plot_values[np.abs(plot_values) < threshold] = np.nan

    fig, ax = plt.subplots(figsize=(13, 11))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#f2f2f2")

    im = ax.imshow(plot_values.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    labels = list(plot_values.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(f"Feature Correlation Matrix (Threshold {threshold})", fontsize=14)

    # Grid to mimic matrix cells
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=11)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # CHANGE THESE PATHS
    # -----------------------------------------------------
    metrics_csv = "/home/ddimopoulos/01_simple_imputation/outputs/leakage_safe_mpi_seed42/all_metrics_internal_external.csv"

    # for correlation matrix: choose one feature table
    feature_csv = "/home/ddimopoulos/01_simple_imputation/CSV/exports/final/mimic_mean_median_min_max_final.csv"

    outdir = Path("/home/ddimopoulos/01_simple_imputation/outputs/paper_like_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load metrics
    # -----------------------------------------------------
    df = prepare_metrics(metrics_csv)

    # 1) comparison-like plot
    plot_comparison_like(df, outdir / "01_comparison_like.png")

    # 2) multi-panel metrics plot
    plot_metrics_grid_like(df, outdir / "02_metrics_grid_like.png")

    # 3) correlation matrix
    plot_correlation_matrix_like(
        feature_csv=feature_csv,
        outpath=outdir / "03_correlation_matrix_threshold_0.5.png",
        threshold=0.5,
        max_features=60,
    )

    print(f"Plots written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()