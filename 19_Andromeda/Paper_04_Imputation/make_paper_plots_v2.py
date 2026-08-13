#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Global style
# =========================================================

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
    "axes.grid": False,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
})


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

# Better colors, close to publication style
PLOT_COLORS = {
    "native": "#4C72B0",         # blue
    "mean": "#8FB7E8",           # light blue
    "interpolation": "#C9D7E8",  # pale blue-gray
    "mlp": "#E7D7CC",            # warm beige
    "rnn": "#E4A07A",            # salmon
    "regression": "#C96B5C",     # deeper red-brown
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


def save_figure(fig, outpath_without_suffix: Path):
    png_path = outpath_without_suffix.with_suffix(".png")
    pdf_path = outpath_without_suffix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def prepare_metrics(metrics_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv).copy()

    df["scaling"] = normalise_scaling_column(df["scaling"])

    df["imputation"] = pd.Categorical(
        df["imputation"], categories=IMPUTATION_ORDER, ordered=True
    )
    df["tuning"] = pd.Categorical(
        df["tuning"], categories=TUNING_ORDER, ordered=True
    )
    df["scaling"] = pd.Categorical(
        df["scaling"], categories=[False, True], ordered=True
    )

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
    name = str(name)
    return re.sub(r'[\*_]?\((Min|Max|Mean|Median)\)$', '', name, flags=re.IGNORECASE)


def collapse_statistical_features(feature_df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.drop(columns=[c for c in df.columns if c in non_feature_cols], errors="ignore")
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

    return pd.DataFrame(collapsed)


def annotate_bars(ax, bars, fontsize=9):
    for bar in bars:
        h = bar.get_height()
        if pd.isna(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


# =========================================================
# Plot 1: comparison plot (all 48)
# =========================================================

def plot_comparison(df: pd.DataFrame, outbase: Path, title: str):
    fig, ax_mse = plt.subplots(figsize=(14, 18))
    ax_mae = ax_mse.twiny()

    y = np.arange(len(df))

    line_internal_mae, = ax_mae.plot(
        df["mae__mimic_test"], y,
        marker="o", markersize=6,
        linestyle="-", linewidth=1.8,
        color="blue", label="Internal MAE"
    )

    line_external_mae, = ax_mae.plot(
        df["mae__eicu_external"], y,
        marker="x", markersize=6,
        linestyle="--", linewidth=1.5,
        color="red", label="External MAE"
    )

    line_internal_mse, = ax_mse.plot(
        df["mse__mimic_test"], y,
        marker="o", markersize=6,
        linestyle="-", linewidth=1.8,
        color="green", label="Internal MSE"
    )

    line_external_mse, = ax_mse.plot(
        df["mse__eicu_external"], y,
        marker="x", markersize=6,
        linestyle="--", linewidth=1.5,
        color="orange", label="External MSE"
    )

    ax_mse.set_yticks(y)
    ax_mse.set_yticklabels(df["config_label"])
    ax_mse.invert_yaxis()

    ax_mse.set_xlabel("MSE", color="green", fontsize=15)
    ax_mae.set_xlabel("MAE", color="blue", fontsize=15)
    ax_mse.set_ylabel("Impute | Scale/Norm | HP", fontsize=15)
    ax_mse.set_title(title, fontsize=18, pad=16)

    ax_mse.tick_params(axis="x", colors="green")
    ax_mae.tick_params(axis="x", colors="blue")

    ax_mse.grid(True, axis="both", alpha=0.25)

    lines = [line_internal_mae, line_external_mae, line_internal_mse, line_external_mse]
    labels = [l.get_label() for l in lines]
    ax_mse.legend(lines, labels, loc="upper right", frameon=True)

    fig.tight_layout()
    save_figure(fig, outbase)
    plt.close(fig)


# =========================================================
# Plot 2: separate comparison plots for scaling No / Yes
# =========================================================

def plot_comparison_by_scaling(df: pd.DataFrame, outdir: Path):
    for scaling_value in [False, True]:
        subset = df[df["scaling"] == scaling_value].copy()
        scale_label = SCALING_LABEL[scaling_value]
        outbase = outdir / f"comparison_scale_{scale_label.lower()}"
        title = f"Comparison Plot | Scale/Norm {scale_label}"
        plot_comparison(subset, outbase, title)


# =========================================================
# Plot 3: 6-panel metrics grid per scaling
# =========================================================

def plot_metrics_grid_for_scaling(df: pd.DataFrame, scaling_value: bool, outbase: Path):
    subset = df[df["scaling"] == scaling_value].copy()

    subset["imputation"] = pd.Categorical(
        subset["imputation"], categories=IMPUTATION_ORDER, ordered=True
    )
    subset["tuning"] = pd.Categorical(
        subset["tuning"], categories=TUNING_ORDER, ordered=True
    )

    subset = subset.sort_values(["tuning", "imputation"]).reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
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
            imp_df = subset[subset["imputation"] == imp].set_index("tuning")
            vals = [imp_df.loc[t, metric_col] if t in imp_df.index else np.nan for t in TUNING_ORDER]
            pos = x + (i - (n_imp - 1) / 2) * bar_width

            bars = ax.bar(
                pos,
                vals,
                width=bar_width,
                label=IMPUTATION_LABEL[imp],
                color=PLOT_COLORS[imp],
                edgecolor="white",
                linewidth=0.6,
            )
            annotate_bars(ax, bars, fontsize=8)

        mean_value = np.nanmean(subset[metric_col])

        ax.axhline(
            mean_value,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean {ylabel}: {mean_value:.2f}",
        )

        ax.set_title(title, weight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Hyperparameter Tuning Method")
        ax.set_xticks(x)
        ax.set_xticklabels([TUNING_LABEL[t] for t in TUNING_ORDER], rotation=35)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="upper right", frameon=True, fontsize=10)

    scale_label = SCALING_LABEL[scaling_value]
    fig.suptitle(f"Performance Metrics | Scale/Norm {scale_label}", y=0.995, fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    save_figure(fig, outbase)
    plt.close(fig)


# =========================================================
# Plot 4: correlation matrix
# =========================================================

def plot_correlation_matrix_like(
    feature_csv: str | Path,
    outbase: Path,
    threshold: float = 0.5,
    max_features: int = 60,
):
    raw = pd.read_csv(feature_csv, low_memory=False)
    feat = collapse_statistical_features(raw)

    feat = feat.loc[:, feat.notna().sum() > 0]
    feat = feat.loc[:, feat.nunique(dropna=True) > 1]

    corr = feat.corr(method="pearson")
    corr_abs = corr.abs()

    strong_mask = corr_abs >= threshold
    np.fill_diagonal(strong_mask.values, False)

    keep = strong_mask.any(axis=0)
    corr_sub = corr.loc[keep, keep].copy()
    strong_counts = strong_mask.loc[keep, keep].sum(axis=0).sort_values(ascending=False)

    if len(strong_counts) > max_features:
        selected = strong_counts.index[:max_features]
        corr_sub = corr_sub.loc[selected, selected]

    plot_values = corr_sub.copy()
    plot_values[np.abs(plot_values) < threshold] = np.nan

    fig, ax = plt.subplots(figsize=(14, 12))

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#f2f2f2")

    im = ax.imshow(plot_values.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    labels = list(plot_values.columns)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    ax.set_title(f"Feature Correlation Matrix (Threshold {threshold})", fontsize=18, pad=12)

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=13)

    fig.tight_layout()
    save_figure(fig, outbase)
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    # -----------------------------------------------------
    # Paths
    # -----------------------------------------------------
    metrics_csv = "/home/ddimopoulos/01_simple_imputation/outputs/leakage_safe_mpi_seed42/all_metrics_internal_external.csv"
    feature_csv = "/home/ddimopoulos/01_simple_imputation/CSV/exports/final/mimic_mean_median_min_max_final.csv"

    outdir = Path("/home/ddimopoulos/01_simple_imputation/outputs/paper_like_plots_v2")
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_metrics(metrics_csv)

    # 1. Full 48-config comparison plot
    plot_comparison(
        df=df,
        outbase=outdir / "01_comparison_all_48",
        title="Comparison Plot | All 48 Configurations"
    )

    # 2. Separate comparison plots for Scale/Norm No and Yes
    plot_comparison_by_scaling(df, outdir)

    # 3. Separate 6-panel metrics plots for Scale/Norm No and Yes
    plot_metrics_grid_for_scaling(
        df=df,
        scaling_value=False,
        outbase=outdir / "02_metrics_grid_scale_no"
    )

    plot_metrics_grid_for_scaling(
        df=df,
        scaling_value=True,
        outbase=outdir / "03_metrics_grid_scale_yes"
    )

    # 4. Correlation matrix
    plot_correlation_matrix_like(
        feature_csv=feature_csv,
        outbase=outdir / "04_correlation_matrix_threshold_0_5",
        threshold=0.5,
        max_features=60,
    )

    print(f"\nAll plots written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()