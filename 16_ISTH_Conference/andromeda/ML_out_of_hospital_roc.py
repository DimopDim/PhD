# plot_combined_rocs_from_oof.py
# ------------------------------------------------------------
# Creates combined ROC plots from saved OOF predictions:
#   - 1 plot per target (mort_30d / mort_180d / mort_360d)
#   - Each plot overlays all scenarios
# Saves PNG + PDF into: ML_Scenarios_OutOfHospitalMortality/plots/
# ------------------------------------------------------------

from pathlib import Path
import re
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------
# CONFIG
# -------------------------
OUT_DIR = Path("ML_Scenarios_OutOfHospitalMortality")
OOF_DIR = OUT_DIR / "oof"
PLOTS_DIR = OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["mort_30d", "mort_180d", "mort_360d"]
SCENARIOS = ["SOFA", "SOFA+DEMO", "RAR", "RAR+DEMO", "ALL_NO_SOFA", "ALL_PLUS_SOFA"]

# -------------------------
# Helpers
# -------------------------
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.figsize": (7.6, 5.2),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 2.2,
    })

def save_png_pdf(base: Path):
    plt.tight_layout()
    plt.savefig(base.with_suffix(".png"), bbox_inches="tight")
    plt.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()

def load_oof(target: str, scenario: str) -> pd.DataFrame:
    fp = OOF_DIR / f"oof__{target}__{scenario}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp)
    if not {"y_true", "p_oof"}.issubset(df.columns):
        return None
    df = df.copy()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["p_oof"] = pd.to_numeric(df["p_oof"], errors="coerce")
    df = df.dropna(subset=["y_true", "p_oof"])
    df["y_true"] = df["y_true"].astype(int)
    return df

def plot_combined_roc_for_target(target: str):
    plt.figure()

    plotted_any = False
    n_for_title = None
    prev_for_title = None

    for scenario in SCENARIOS:
        oof = load_oof(target, scenario)
        if oof is None or oof.empty:
            continue

        y = oof["y_true"].values
        p = oof["p_oof"].values

        # need both classes
        if len(np.unique(y)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        if n_for_title is None:
            n_for_title = len(y)
            prev_for_title = float(np.mean(y))

        plt.plot(fpr, tpr, label=f"{scenario} (AUC={auc:.3f})")
        plotted_any = True

    # reference diagonal
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3)

    if not plotted_any:
        plt.close()
        print(f"[SKIP] No valid ROC curves found for {target}")
        return

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    title = f"Combined ROC (OOF) — {target}"
    if n_for_title is not None:
        title += f" (n={n_for_title}, event rate={prev_for_title:.3f})"
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)

    out_base = PLOTS_DIR / f"roc__{target}__ALL_SCENARIOS"
    save_png_pdf(out_base)
    print("[OK] Saved:", out_base.with_suffix(".png"))

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    set_plot_style()

    if not OOF_DIR.exists():
        raise FileNotFoundError(f"Missing OOF_DIR: {OOF_DIR}")

    for t in TARGETS:
        plot_combined_roc_for_target(t)

    print("Done.")
