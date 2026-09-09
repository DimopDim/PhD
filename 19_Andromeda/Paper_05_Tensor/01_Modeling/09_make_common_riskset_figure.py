#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_make_common_riskset_figure.py

Create a publication-ready common-risk-set paired landmark forest figure
for the revised cross-database landmark analysis.

The figure uses existing paired patient-level bootstrap outputs only.
No model fitting, tuning, prediction, or bootstrap resampling is performed.

Scientific orientation
----------------------
For every metric the plotted effect is re-oriented so that:

    positive value  = later landmark performed better
    negative value  = earlier landmark performed better
    zero            = no paired difference

Thus:
  * higher-is-better metrics (R2, AUROC, AP):
        plotted_delta = later - earlier
  * lower-is-better metrics (MAE, RMSE, Brier):
        plotted_delta = earlier - later
      i.e. the sign of the stored later-minus-earlier delta is reversed.

Panels
------
(a) LOS MAE
(b) LOS RMSE
(c) LOS R2
(d) Mortality AUROC
(e) Mortality average precision
(f) Mortality Brier score

Each panel compares:
  * MIMIC-IV internal test, Full
  * MIMIC-IV internal test, Static
  * eICU-CRD external, Full
  * eICU-CRD external, Static

for:
  * 4 -> 24 h common-risk-set comparison
  * 24 -> 48 h common-risk-set comparison

Outputs
-------
PDF, SVG, 600-dpi PNG, exact plotting data CSV, and a provenance manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


DEFAULT_MODELING_ROOT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/01_Modeling"
)

VARIANTS = ["full", "static"]
COHORTS = ["mimic_test", "eicu_external"]

VARIANT_LABEL = {
    "full": "Full",
    "static": "Static",
}

COHORT_LABEL = {
    "mimic_test": "MIMIC-IV",
    "eicu_external": "eICU-CRD",
}

COLORS = {
    "full": "#1f77b4",
    "static": "#ff7f0e",
    "reference": "#666666",
}

INTERVAL_MARKERS = {
    "4→24 h": "o",
    "24→48 h": "s",
}

# The six metrics intentionally cover:
# - absolute-error and squared-error LOS behavior,
# - LOS explained variance,
# - mortality discrimination,
# - class-imbalance-sensitive discrimination,
# - probability quality / calibration-sensitive error.
PANEL_SPECS = [
    {
        "outcome": "los",
        "metric": "mae",
        "title": "LOS MAE",
        "xlabel": "Oriented paired difference in MAE (days)",
        "higher_better": False,
    },
    {
        "outcome": "los",
        "metric": "rmse",
        "title": "LOS RMSE",
        "xlabel": "Oriented paired difference in RMSE (days)",
        "higher_better": False,
    },
    {
        "outcome": "los",
        "metric": "r2",
        "title": r"LOS $R^2$",
        "xlabel": r"Oriented paired difference in $R^2$",
        "higher_better": True,
    },
    {
        "outcome": "mortality",
        "metric": "roc_auc",
        "title": "Mortality AUROC",
        "xlabel": "Oriented paired difference in AUROC",
        "higher_better": True,
    },
    {
        "outcome": "mortality",
        "metric": "average_precision",
        "title": "Mortality average precision",
        "xlabel": "Oriented paired difference in average precision",
        "higher_better": True,
    },
    {
        "outcome": "mortality",
        "metric": "brier",
        "title": "Mortality Brier score",
        "xlabel": "Oriented paired difference in Brier score",
        "higher_better": False,
    },
]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )


def _read_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("bootstrap_replicates", -1)) != 2000:
        raise ValueError(
            f"{path}: expected 2000 bootstrap replicates, "
            f"found {payload.get('bootstrap_replicates')!r}."
        )
    if payload.get("mortality_probability") != "calibrated":
        raise ValueError(
            f"{path}: mortality_probability must be 'calibrated'."
        )
    return payload


def load_commonrisk_source(
    path: Path,
    *,
    expected_variant: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {
        "metric",
        "ci_low",
        "ci_high",
        "delta_later_minus_earlier",
        "earlier_landmark_hour",
        "later_landmark_hour",
        "outcome",
        "variant",
        "cohort",
        "n_common_riskset",
        "valid_bootstrap_replicates",
        "ci_excludes_zero",
        "conclusion",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    variants = set(df["variant"].astype(str).unique().tolist())
    if variants != {expected_variant}:
        raise ValueError(
            f"{path}: expected only variant={expected_variant!r}, "
            f"found {sorted(variants)}."
        )

    if not np.isfinite(
        df[
            [
                "ci_low",
                "ci_high",
                "delta_later_minus_earlier",
                "n_common_riskset",
                "valid_bootstrap_replicates",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{path}: non-finite common-risk values detected.")

    if (df["n_common_riskset"] <= 0).any():
        raise ValueError(f"{path}: non-positive common-risk-set size.")
    if (df["valid_bootstrap_replicates"] <= 0).any():
        raise ValueError(f"{path}: no valid bootstrap replicates for some rows.")

    df = df.copy()
    df["source_file"] = str(path)
    return df


def load_all_commonrisk(
    modeling_root: Path,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the two canonical Stage-07 common-risk outputs:
      * primary bootstrap_results -> Full
      * bootstrap_results/commonrisk_static -> Static

    Only the prespecified widest-window contrasts 4->24 h and 24->48 h
    are retained for the figure. The underlying Stage-07 CSVs may contain
    additional pairwise contrasts within each common-risk block.
    """
    bootstrap_root = modeling_root / "bootstrap_results"

    full_csv = (
        bootstrap_root
        / "paired_landmark_common_riskset.csv"
    )
    static_csv = (
        bootstrap_root
        / "commonrisk_static"
        / "paired_landmark_common_riskset.csv"
    )

    full_manifest_path = bootstrap_root / "bootstrap_run_manifest.json"
    static_manifest_path = (
        bootstrap_root
        / "commonrisk_static"
        / "bootstrap_run_manifest.json"
    )

    full_manifest = _read_manifest(full_manifest_path)
    static_manifest = _read_manifest(static_manifest_path)

    full = load_commonrisk_source(
        full_csv,
        expected_variant="full",
    )
    static = load_commonrisk_source(
        static_csv,
        expected_variant="static",
    )

    out = pd.concat([full, static], ignore_index=True)

    comparisons = [
        (4, 24, "4→24 h"),
        (24, 48, "24→48 h"),
    ]
    frames = []
    for earlier, later, label in comparisons:
        part = out.loc[
            out["earlier_landmark_hour"].eq(earlier)
            & out["later_landmark_hour"].eq(later)
        ].copy()
        if part.empty:
            raise ValueError(
                f"No {earlier}->{later} h rows found in canonical outputs."
            )
        part["comparison"] = label
        frames.append(part)

    out = pd.concat(frames, ignore_index=True)

    key = [
        "comparison",
        "outcome",
        "metric",
        "variant",
        "cohort",
        "earlier_landmark_hour",
        "later_landmark_hour",
    ]
    duplicates = out.duplicated(key, keep=False)
    if duplicates.any():
        dup = out.loc[duplicates, key]
        raise ValueError(
            "Duplicate common-risk-set rows detected:\n"
            + dup.to_string(index=False)
        )

    provenance = {
        "full_csv": str(full_csv),
        "static_csv": str(static_csv),
        "full_manifest": str(full_manifest_path),
        "static_manifest": str(static_manifest_path),
        "bootstrap_replicates_full": int(
            full_manifest.get("bootstrap_replicates", -1)
        ),
        "bootstrap_replicates_static": int(
            static_manifest.get("bootstrap_replicates", -1)
        ),
        "mortality_probability_full": full_manifest.get(
            "mortality_probability"
        ),
        "mortality_probability_static": static_manifest.get(
            "mortality_probability"
        ),
    }
    return out, provenance


def orient_effects(
    df: pd.DataFrame,
    higher_better: bool,
) -> pd.DataFrame:
    """
    Re-orient stored later-minus-earlier deltas so that positive means
    the later landmark is better for every metric.
    """
    out = df.copy()

    if higher_better:
        out["effect"] = out["delta_later_minus_earlier"]
        out["effect_ci_low"] = out["ci_low"]
        out["effect_ci_high"] = out["ci_high"]
    else:
        # Stored value = later - earlier.
        # For error/loss metrics, lower is better.
        # Therefore improvement = earlier - later = -(later - earlier).
        out["effect"] = -out["delta_later_minus_earlier"]
        out["effect_ci_low"] = -out["ci_high"]
        out["effect_ci_high"] = -out["ci_low"]

    return out


def panel_rows(
    data: pd.DataFrame,
    *,
    outcome: str,
    metric: str,
    higher_better: bool,
) -> pd.DataFrame:
    df = data.loc[
        data["outcome"].eq(outcome)
        & data["metric"].eq(metric)
        & data["variant"].isin(VARIANTS)
        & data["cohort"].isin(COHORTS)
    ].copy()

    df = orient_effects(df, higher_better)

    expected = (
        len(VARIANTS)
        * len(COHORTS)
        * len(INTERVAL_MARKERS)
    )
    if len(df) != expected:
        raise ValueError(
            f"Expected {expected} rows for {outcome}/{metric}, "
            f"found {len(df)}."
        )

    return df


def add_panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def symmetric_limits(
    lows: np.ndarray,
    highs: np.ndarray,
    pad_fraction: float = 0.12,
) -> tuple[float, float]:
    extent = float(
        np.nanmax(
            np.abs(
                np.concatenate([lows, highs, np.array([0.0])])
            )
        )
    )
    if not np.isfinite(extent) or extent == 0:
        extent = 1.0
    extent *= 1.0 + pad_fraction
    return -extent, extent


def plot_metric_panel(
    ax,
    data: pd.DataFrame,
    spec: dict,
    *,
    show_ylabels: bool,
) -> None:
    df = panel_rows(
        data,
        outcome=spec["outcome"],
        metric=spec["metric"],
        higher_better=spec["higher_better"],
    )

    # Four rows. Within each row, 4→24 and 24→48 are vertically offset.
    row_order = [
        ("mimic_test", "full"),
        ("mimic_test", "static"),
        ("eicu_external", "full"),
        ("eicu_external", "static"),
    ]

    y_base = {
        pair: 3 - i
        for i, pair in enumerate(row_order)
    }

    interval_offset = {
        "4→24 h": +0.12,
        "24→48 h": -0.12,
    }

    for (cohort, variant), base in y_base.items():
        subset = df.loc[
            df["cohort"].eq(cohort)
            & df["variant"].eq(variant)
        ]

        for comparison in INTERVAL_MARKERS:
            row = subset.loc[
                subset["comparison"].eq(comparison)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one row for "
                    f"{spec['outcome']}/{spec['metric']}/"
                    f"{cohort}/{variant}/{comparison}; "
                    f"found {len(row)}."
                )

            r = row.iloc[0]
            y = base + interval_offset[comparison]

            x = float(r["effect"])
            lo = float(r["effect_ci_low"])
            hi = float(r["effect_ci_high"])

            ax.errorbar(
                x,
                y,
                xerr=np.array([[x - lo], [hi - x]]),
                fmt=INTERVAL_MARKERS[comparison],
                markersize=5.2,
                markerfacecolor=COLORS[variant],
                markeredgecolor=COLORS[variant],
                color=COLORS[variant],
                ecolor=COLORS[variant],
                elinewidth=1.35,
                capsize=2.5,
                capthick=1.0,
                zorder=3,
            )

    # Neutral reference line.
    ax.axvline(
        0.0,
        color=COLORS["reference"],
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )

    # Light row separators.
    for y in [0.5, 1.5, 2.5]:
        ax.axhline(
            y,
            color="#DDDDDD",
            linewidth=0.6,
            zorder=0,
        )

    labels = [
        "MIMIC-IV — Full",
        "MIMIC-IV — Static",
        "eICU-CRD — Full",
        "eICU-CRD — Static",
    ]

    ax.set_yticks([3, 2, 1, 0])
    if show_ylabels:
        ax.set_yticklabels(labels)
    else:
        ax.set_yticklabels([])

    ax.set_ylim(-0.55, 3.55)
    ax.set_title(spec["title"])
    ax.set_xlabel(spec["xlabel"])
    ax.grid(
        axis="x",
        alpha=0.20,
        linewidth=0.7,
    )

    xmin, xmax = symmetric_limits(
        df["effect_ci_low"].to_numpy(dtype=float),
        df["effect_ci_high"].to_numpy(dtype=float),
    )
    ax.set_xlim(xmin, xmax)



def build_plot_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    for spec in PANEL_SPECS:
        df = panel_rows(
            data,
            outcome=spec["outcome"],
            metric=spec["metric"],
            higher_better=spec["higher_better"],
        )
        df["panel_title"] = spec["title"]
        df["positive_means"] = "later_landmark_better"
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    columns = [
        "panel_title",
        "comparison",
        "outcome",
        "metric",
        "variant",
        "cohort",
        "earlier_landmark_hour",
        "later_landmark_hour",
        "n_common_riskset",
        "effect",
        "effect_ci_low",
        "effect_ci_high",
        "delta_later_minus_earlier",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
        "conclusion",
        "valid_bootstrap_replicates",
        "positive_means",
        "source_file",
    ]
    return out[columns].sort_values(
        [
            "outcome",
            "metric",
            "cohort",
            "variant",
            "earlier_landmark_hour",
        ]
    )


def make_figure(
    data: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.0, 6.7),
    )

    for idx, (ax, spec) in enumerate(
        zip(axes.flat, PANEL_SPECS)
    ):
        # Show row labels only on the leftmost panels.
        show_ylabels = (idx % 3 == 0)

        plot_metric_panel(
            ax,
            data,
            spec,
            show_ylabels=show_ylabels,
        )
        add_panel_label(
            ax,
            f"({chr(ord('a') + idx)})",
        )

    # -----------------------------------------------------------------
    # Figure-level legend OUTSIDE the plotting panels.
    # It cannot overlap any curves, estimates, or confidence intervals.
    # -----------------------------------------------------------------
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["full"],
            marker="o",
            linestyle="none",
            markersize=6,
            label="Full",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["static"],
            marker="o",
            linestyle="none",
            markersize=6,
            label="Static",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            marker=INTERVAL_MARKERS["4→24 h"],
            linestyle="none",
            markersize=6,
            label="4→24 h",
        ),
        Line2D(
            [0],
            [0],
            color="#222222",
            marker=INTERVAL_MARKERS["24→48 h"],
            linestyle="none",
            markersize=6,
            label="24→48 h",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["reference"],
            linestyle="--",
            linewidth=1.0,
            label="No paired difference",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        columnspacing=1.5,
        handletextpad=0.5,
    )

    # Leave dedicated space for the external legend.
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.95,
        bottom=0.13,
        wspace=0.28,
        hspace=0.36,
    )

    stem = "common_riskset_paired_landmark_changes"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / f"{stem}.svg",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the publication-ready common-risk-set paired landmark "
            "forest figure from canonical Stage-07 outputs."
        )
    )
    parser.add_argument(
        "--modeling-root",
        type=Path,
        default=DEFAULT_MODELING_ROOT,
        help=(
            "Path to Paper_05_Tensor/01_Modeling. "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Default: "
            "<modeling-root>/figures/main"
        ),
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()

    modeling_root = (
        args.modeling_root
        .expanduser()
        .resolve()
    )

    if not modeling_root.is_dir():
        raise FileNotFoundError(modeling_root)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else modeling_root
        / "figures"
        / "main"
    )

    configure_matplotlib()

    data, provenance = load_all_commonrisk(modeling_root)

    # Write the exact data displayed in the figure.
    plot_data = build_plot_data(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_data.to_csv(
        output_dir
        / "common_riskset_paired_landmark_changes_data.csv",
        index=False,
    )

    make_figure(
        data,
        output_dir,
    )

    manifest = {
        "script": Path(__file__).name,
        "modeling_root": str(modeling_root),
        "output_dir": str(output_dir),
        "displayed_comparisons": ["4→24 h", "24→48 h"],
        "displayed_variants": VARIANTS,
        "displayed_cohorts": COHORTS,
        "displayed_metrics": [
            f"{s['outcome']}:{s['metric']}" for s in PANEL_SPECS
        ],
        "effect_orientation": (
            "positive means later landmark performed better; "
            "lower-is-better metrics are sign-reversed"
        ),
        "provenance": provenance,
        "status": "PASS",
    }
    manifest_path = (
        output_dir / "09_common_riskset_figure_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("Created common-risk-set figure:")
    for suffix in ("png", "pdf", "svg"):
        print(
            output_dir
            / f"common_riskset_paired_landmark_changes.{suffix}"
        )
    print(
        output_dir
        / "common_riskset_paired_landmark_changes_data.csv"
    )
    print(manifest_path)
    print("PASS: Stage 09 common-risk-set figure complete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
