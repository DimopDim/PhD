# decision_curve_inhospital.py
# ============================================================
# Decision Curve Analysis (DCA) for saved XGBoost model
# - Loads saved model + saved feature list (one-hot columns)
# - Recreates the same MIMIC DEV/TEST holdout split (SGKF pick-best)
# - Predicts on MIMIC TEST + eICU EXTERNAL
# - Plots Net Benefit vs threshold (treat-all / treat-none baselines)
# - Saves PNG/PDF + CSV tables
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb


# -------------------------
# CONFIG (same as training)
# -------------------------
MIMIC_PATH = "CSV/o01_mimic_for_ext_val.csv"
EICU_PATH  = "CSV/o01_eicu_for_ext_val.csv"

RANDOM_STATE = 42
TARGET_COL   = "hospital_expire_flag"
GROUP_COL    = "subject_id"

EXCLUDE_FEATURES = [
    "row_count",
    "subject_id",
    "hadm_id",
    "Time_Zone",
    "los",
    TARGET_COL,
]

# Where the model was saved by your training script
OUT_DIR = Path("CSV/Exports/Temp/13_ML_inhospital_mortality_optuna")

MODEL_BASENAME = "xgb_inhospital_mortality_optuna"
MODEL_PATH     = OUT_DIR / f"{MODEL_BASENAME}.json"
FEATS_PATH     = OUT_DIR / f"{MODEL_BASENAME}_features.txt"
META_PATH      = OUT_DIR / f"{MODEL_BASENAME}_meta.json"

# Decision thresholds (edit if you want)
THRESH_START = 0.01
THRESH_STOP  = 0.50
THRESH_STEP  = 0.01


# -------------------------
# Plot style
# -------------------------
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

set_plot_style()


# -------------------------
# IO + label (same logic)
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_binary_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.copy()

    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        return df

    if "unitdischargestatus" in df.columns:
        m = {"Alive": 0, "alive": 0, "Expired": 1, "expired": 1, "Dead": 1, "dead": 1}
        df[target_col] = df["unitdischargestatus"].astype(str).map(m)
        return df

    for c in ["hospitaldischargestatus", "dischargestatus", "death", "mortality"]:
        if c in df.columns:
            df[target_col] = pd.to_numeric(df[c], errors="coerce")
            return df

    raise KeyError(f"Δεν βρέθηκε label '{target_col}' ούτε fallback.")

def make_X_y_groups(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    exclude: list[str] = EXCLUDE_FEATURES,
    require_group: bool = True,
):
    """
    - Drop only rows with missing/invalid LABEL (not features)
    - Do NOT fill missing feature values
    - Keep NaN as NaN
    - dummy_na=True in one-hot
    """
    df = df.copy()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    y_full = pd.to_numeric(df[target_col], errors="coerce")
    keep = y_full.notna()
    df = df.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y  = y.loc[ok].copy()

    if group_col in df.columns:
        groups = df[group_col].values
    else:
        if require_group:
            raise KeyError(f"Group column '{group_col}' not found.")
        groups = np.arange(len(df), dtype=int)

    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    for c in bool_cols:
        X[c] = X[c].astype("int8")

    leaked = [c for c in exclude if c in X.columns]
    if leaked:
        raise RuntimeError(f"Excluded columns leaked into X: {leaked}")

    return X, y, groups

def one_hot_transform_to_saved_features(X_raw: pd.DataFrame, saved_columns: list[str]) -> pd.DataFrame:
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=True)
    # IMPORTANT: keep exact training feature space
    return X_oh.reindex(columns=saved_columns, fill_value=0)


# -------------------------
# Same holdout splitter
# -------------------------
def _pick_best_sgkf_holdout(X, y, groups, holdout_size: float, random_state: int):
    n_splits = max(int(round(1.0 / holdout_size)), 2)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_arr = np.asarray(y)
    target_n = int(round(len(y_arr) * holdout_size))
    target_pos = float(np.mean(y_arr))

    best = None
    best_score = float("inf")

    for tr_idx, ho_idx in sgkf.split(X, y, groups=groups):
        n_ho = len(ho_idx)
        pos_ho = float(np.mean(y_arr[ho_idx])) if n_ho > 0 else 0.0

        size_term = abs(n_ho - target_n) / max(target_n, 1)
        pos_term  = abs(pos_ho - target_pos) / max(target_pos, 1e-6)
        score = size_term + pos_term

        if score < best_score:
            best_score = score
            best = (tr_idx, ho_idx)

    return np.asarray(best[0]), np.asarray(best[1])

def split_mimic_dev_test(X, y, groups, test_size=0.10, random_state=RANDOM_STATE):
    return _pick_best_sgkf_holdout(X, y, groups, holdout_size=test_size, random_state=random_state)


# -------------------------
# Model load + predict
# -------------------------
def load_saved_feature_list(path: Path) -> list[str]:
    cols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip()
            if c:
                cols.append(c)
    if not cols:
        raise RuntimeError(f"Empty feature list: {path}")
    return cols

def load_booster(path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster

def predict_proba(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    return booster.predict(d)


# -------------------------
# Decision Curve Analysis
# -------------------------
def dca_net_benefit_curve(y_true: np.ndarray, p_pred: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Net Benefit (Vickers & Elkin):
      NB(pt) = TP/n - FP/n * (pt/(1-pt))
    Baselines:
      Treat-none => 0
      Treat-all  => prevalence - (1-prevalence)*(pt/(1-pt))
    """
    y_true = np.asarray(y_true).astype(int)
    p_pred = np.asarray(p_pred).astype(float)

    n = len(y_true)
    prev = float(np.mean(y_true))

    rows = []
    for pt in thresholds:
        if pt <= 0.0 or pt >= 1.0:
            continue
        w = pt / (1.0 - pt)

        treat = (p_pred >= pt).astype(int)
        tp = int(np.sum((treat == 1) & (y_true == 1)))
        fp = int(np.sum((treat == 1) & (y_true == 0)))

        nb_model = (tp / n) - (fp / n) * w
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

def plot_dca(df_curve: pd.DataFrame, title: str, out_base: Path):
    plt.figure()
    plt.plot(df_curve["threshold"], df_curve["net_benefit_model"], label="Model", linewidth=2.4)
    plt.plot(df_curve["threshold"], df_curve["net_benefit_treat_all"], label="Treat all", linewidth=2.0)
    plt.plot(df_curve["threshold"], df_curve["net_benefit_treat_none"], label="Treat none", linewidth=2.0)

    prev = float(df_curve["prevalence"].iloc[0]) if len(df_curve) else float("nan")
    n    = int(df_curve["n"].iloc[0]) if len(df_curve) else 0

    plt.axhline(0.0, linestyle="--", linewidth=1.3)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net Benefit")
    plt.title(f"{title}  (n={n}, event rate={prev:.3f})")
    plt.legend(frameon=False)
    save_fig_png_pdf(out_base)

def plot_dca_compare(curves: dict, title: str, out_base: Path):
    """
    curves: dict[name] = df_curve (already contains treat-all based on its prevalence)
    """
    plt.figure()
    # Model curves
    for name, dfc in curves.items():
        plt.plot(dfc["threshold"], dfc["net_benefit_model"], label=f"{name}: Model")

    # Treat-all for each dataset
    for name, dfc in curves.items():
        plt.plot(dfc["threshold"], dfc["net_benefit_treat_all"], linestyle="--", linewidth=1.8, label=f"{name}: Treat all")

    # Treat-none (common)
    any_df = next(iter(curves.values()))
    plt.plot(any_df["threshold"], any_df["net_benefit_treat_none"], linestyle="-.", linewidth=1.8, label="Treat none")

    plt.axhline(0.0, linestyle="--", linewidth=1.3)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net Benefit")
    plt.title(title)
    plt.legend(frameon=False)
    save_fig_png_pdf(out_base)


# -------------------------
# MAIN
# -------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not FEATS_PATH.exists():
        raise FileNotFoundError(f"Missing features file: {FEATS_PATH}")

    # Optional: read meta (just to log)
    if META_PATH.exists():
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("Loaded metadata:", META_PATH)
        print("Model file in meta:", meta.get("model_file"))
        print("Num features in meta:", meta.get("num_features"))

    saved_cols = load_saved_feature_list(FEATS_PATH)
    print(f"Loaded saved feature space: {len(saved_cols)} columns")

    booster = load_booster(MODEL_PATH)
    print("Loaded XGBoost model:", MODEL_PATH)

    print("\nLoad datasets...")
    mimic_df = ensure_binary_label(load_csv(MIMIC_PATH), TARGET_COL)
    eicu_df  = ensure_binary_label(load_csv(EICU_PATH),  TARGET_COL)

    print("Build X/y/groups (NO IMPUTATION)...")
    X_m_raw, y_m, g_m = make_X_y_groups(mimic_df, TARGET_COL, GROUP_COL, EXCLUDE_FEATURES, require_group=True)
    X_e_raw, y_e, _   = make_X_y_groups(eicu_df,  TARGET_COL, GROUP_COL, EXCLUDE_FEATURES, require_group=False)

    print("Recreate MIMIC DEV/TEST holdout split...")
    dev_idx, test_idx = split_mimic_dev_test(X_m_raw, y_m, g_m, test_size=0.10, random_state=RANDOM_STATE)
    X_test_raw = X_m_raw.iloc[test_idx].copy()
    y_test     = y_m.iloc[test_idx].values

    print(f"MIMIC TEST: n={len(y_test)} | pos_rate={np.mean(y_test):.4f}")

    # One-hot -> align to saved training feature space
    X_test = one_hot_transform_to_saved_features(X_test_raw, saved_cols)
    X_ext  = one_hot_transform_to_saved_features(X_e_raw,    saved_cols)

    # Predict
    p_test = predict_proba(booster, X_test)
    p_ext  = predict_proba(booster, X_ext)

    # Threshold grid
    thresholds = np.arange(THRESH_START, THRESH_STOP + 1e-12, THRESH_STEP)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]

    # DCA curves
    print("\nCompute DCA curves...")
    dca_test = dca_net_benefit_curve(y_test, p_test, thresholds)
    dca_ext  = dca_net_benefit_curve(y_e.values, p_ext, thresholds)

    # Save CSV
    dca_test.to_csv(OUT_DIR / "decision_curve_mimic_test.csv", index=False)
    dca_ext.to_csv(OUT_DIR / "decision_curve_eicu_external.csv", index=False)

    # Plots (single)
    plot_dca(dca_test, "Decision Curve (MIMIC TEST)", OUT_DIR / "dca_mimic_test")
    plot_dca(dca_ext,  "Decision Curve (eICU EXTERNAL)", OUT_DIR / "dca_eicu_external")

    # Plot (compare)
    plot_dca_compare(
        {"MIMIC TEST": dca_test, "eICU EXTERNAL": dca_ext},
        title="Decision Curve Analysis: MIMIC TEST vs eICU EXTERNAL",
        out_base=OUT_DIR / "dca_test_vs_external"
    )

    print("\nSaved DCA outputs in:", OUT_DIR)
    print(" - decision_curve_mimic_test.csv")
    print(" - decision_curve_eicu_external.csv")
    print(" - dca_mimic_test.(png/pdf)")
    print(" - dca_eicu_external.(png/pdf)")
    print(" - dca_test_vs_external.(png/pdf)")


if __name__ == "__main__":
    main()
