# run_dca_from_oof.py
# ------------------------------------------------------------
# DCA from saved OOF predictions produced by your scenario script
# (no need to load models).
# ------------------------------------------------------------

from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
OUT_DIR = Path("ML_Scenarios_OutOfHospitalMortality")
OOF_DIR = OUT_DIR / "oof"

DCA_DIR = OUT_DIR / "dca"
DCA_PLOTS_DIR = DCA_DIR / "plots"
DCA_CSV_DIR = DCA_DIR / "csv"
DCA_DIR.mkdir(parents=True, exist_ok=True)
DCA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DCA_CSV_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.linspace(0.01, 0.50, 100)

# =========================
# Plot style
# =========================
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7.6, 5.2),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 2.2,
    })

def save_fig_png_pdf(path_base: Path):
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

# =========================
# DCA core
# =========================
def decision_curve(y_true: np.ndarray, p_pred: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Net Benefit:
      NB(pt) = TP/n - FP/n * (pt/(1-pt))
    Treat-none: 0
    Treat-all : prev - (1-prev)*pt/(1-pt)
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    # safety: drop nan
    m = np.isfinite(p_pred) & np.isfinite(y_true)
    y_true = y_true[m]
    p_pred = p_pred[m]

    n = len(y_true)
    prev = float(y_true.mean()) if n else 0.0

    rows = []
    for pt in thresholds:
        w = pt / (1.0 - pt)

        treat = (p_pred >= pt).astype(int)
        tp = int(np.sum((treat == 1) & (y_true == 1)))
        fp = int(np.sum((treat == 1) & (y_true == 0)))

        nb_model = (tp / n) - (fp / n) * w if n else np.nan
        nb_all   = prev - (1.0 - prev) * w
        nb_none  = 0.0

        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": float(nb_none),
            "prevalence": float(prev),
            "n": int(n),
            "tp": tp,
            "fp": fp,
        })

    return pd.DataFrame(rows)

def useful_threshold_range(dca_df: pd.DataFrame) -> tuple[float | None, float | None]:
    """
    Useful range where model NB > max(treat-all, treat-none=0).
    """
    ref = np.maximum(dca_df["net_benefit_treat_all"].values, 0.0)
    ok = dca_df["net_benefit_model"].values > ref
    if not np.any(ok):
        return None, None
    th = dca_df["threshold"].values
    return float(th[ok][0]), float(th[ok][-1])

# =========================
# Parsing helpers
# =========================
def parse_target_scenario(filename: str):
    # expects: oof__mort_30d__SOFA.csv
    m = re.match(r"oof__([^_]+_\d+d)__([^\.]+)\.csv$", filename)
    if not m:
        return None, None
    return m.group(1), m.group(2)

# =========================
# Plotting
# =========================
def plot_dca_single(dca_df: pd.DataFrame, title: str, out_base: Path):
    plt.figure()
    plt.plot(dca_df["threshold"], dca_df["net_benefit_model"], label="Model")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_treat_all"], label="Treat all")
    plt.plot(dca_df["threshold"], dca_df["net_benefit_treat_none"], label="Treat none")
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(title)
    plt.legend(loc="best", frameon=False)
    save_fig_png_pdf(out_base)

def plot_dca_target_overlay(target: str, dca_map: dict, out_base: Path):
    """
    dca_map: scenario -> dca_df
    """
    # pick one df to get treat-all/none lines
    any_df = next(iter(dca_map.values()))
    prev = any_df["prevalence"].iloc[0]
    n = int(any_df["n"].iloc[0])

    plt.figure()
    # scenarios
    for scenario, df in sorted(dca_map.items()):
        plt.plot(df["threshold"], df["net_benefit_model"], label=f"{scenario}")

    # reference lines
    plt.plot(any_df["threshold"], any_df["net_benefit_treat_all"], linestyle="--", label="Treat all")
    plt.plot(any_df["threshold"], any_df["net_benefit_treat_none"], linestyle=":", label="Treat none")

    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"DCA — {target} (n={n}, event rate={prev:.3f}) — all scenarios")
    plt.legend(loc="best", frameon=False)
    save_fig_png_pdf(out_base)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    set_plot_style()

    if not OOF_DIR.exists():
        raise FileNotFoundError(f"Missing OOF dir: {OOF_DIR}")

    files = sorted([p for p in OOF_DIR.glob("oof__*.csv")])
    if not files:
        raise FileNotFoundError(f"No OOF files found in: {OOF_DIR}")

    print(f"Found {len(files)} OOF files.")

    # store for overlay plots
    per_target = {}  # target -> {scenario -> dca_df}
    summary_rows = []

    for fp in files:
        target, scenario = parse_target_scenario(fp.name)
        if target is None:
            print("[SKIP] Unrecognized filename:", fp.name)
            continue

        oof = pd.read_csv(fp)
        if not {"y_true", "p_oof"}.issubset(oof.columns):
            print("[SKIP] Missing columns in:", fp.name)
            continue

        y = oof["y_true"].astype(int).values
        p = oof["p_oof"].astype(float).values

        dca = decision_curve(y, p, THRESHOLDS)

        # save dca csv
        out_csv = DCA_CSV_DIR / f"dca__{target}__{scenario}.csv"
        dca.to_csv(out_csv, index=False)

        # single plot
        prev = dca["prevalence"].iloc[0]
        n = int(dca["n"].iloc[0])
        title = f"DCA — {target} | {scenario} (n={n}, event rate={prev:.3f})"
        out_plot = DCA_PLOTS_DIR / f"dca__{target}__{scenario}"
        plot_dca_single(dca, title, out_plot)

        # useful range
        lo, hi = useful_threshold_range(dca)
        summary_rows.append({
            "target": target,
            "scenario": scenario,
            "n": n,
            "event_rate": float(prev),
            "useful_threshold_low": lo,
            "useful_threshold_high": hi,
            "dca_csv": str(out_csv),
        })

        per_target.setdefault(target, {})[scenario] = dca

        print(f"[OK] {target} | {scenario} -> saved DCA")

    # overlay plots per target
    for target, dmap in per_target.items():
        out_plot = DCA_PLOTS_DIR / f"dca__{target}__ALL_SCENARIOS"
        plot_dca_target_overlay(target, dmap, out_plot)
        print(f"[OK] Overlay saved for target: {target}")

    # summary table
    summary = pd.DataFrame(summary_rows).sort_values(["target", "scenario"]).reset_index(drop=True)
    summary_path = DCA_DIR / "dca_summary_ranges.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved summary:", summary_path)
    print(summary.to_string(index=False))

    print("\nDone. Outputs in:", DCA_DIR)
