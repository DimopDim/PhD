# roc_by_timezones_from_saved_xgb.py
# ------------------------------------------------------------
# Load saved XGBoost model + features and compute ROC/AUC
# + PR curve / PR-AUC (Average Precision)
# per Time_Zone (e.g., 1..16) for:
# - MIMIC test (recreated split)
# - eICU external (full external)
#
# Produces:
# - ROC plots with multiple curves (one per Time_Zone)
# - PR plots with multiple curves (one per Time_Zone)
# - Metrics table CSVs (AUC + PR_AUC)
# ------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)


# -------------------------
# CONFIG
# -------------------------
MIMIC_PATH = "CSV/o01_mimic_for_ext_val.csv"
EICU_PATH  = "CSV/o01_eicu_for_ext_val.csv"

TARGET_COL = "hospital_expire_flag"
GROUP_COL  = "subject_id"
TZ_COL     = "Time_Zone"

# Must match training script
RANDOM_STATE = 42
TEST_SIZE = 0.99
VAL_SIZE  = 0.99

MODEL_DIR = Path("CSV/Exports/Temp/13_ML_inhospital_mortality_optuna")
MODEL_BASENAME = "xgb_inhospital_mortality_optuna"

# I chooce between 1 and 16.
TZ_LIST = [1, 2, 4, 8, 16]   # 1=3h, 2=6h, 4=12h, 8=24h, 16=48h

# Exclude columns (IDs/targets/leakage)
EXCLUDE_FEATURES = [
    "row_count", "subject_id", "hadm_id", "stay_id", "los",
    "icu_intime", "icu_outtime", "hosp_dischtime", "dod", "deathtime",
    TZ_COL, TARGET_COL
]

OUT_DIR = MODEL_DIR / "ROC_by_TimeZone"
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


# -------------------------
# IO + label
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_binary_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Ensures df[target_col] exists and is {0,1}.
    Fallback for eICU: unitdischargestatus -> Alive/Expired/Dead.
    """
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

    raise KeyError(f"Label '{target_col}' not found (and no fallback found).")


# -------------------------
# Build X/y/groups + Time_Zone (NO IMPUTATION)
# -------------------------
def make_X_y_groups_tz(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    group_col: str = GROUP_COL,
    tz_col: str = TZ_COL,
    exclude: list[str] = EXCLUDE_FEATURES
):
    """
    - Drops only rows with missing/invalid LABEL (not features)
    - Keeps feature NaNs as NaN (XGBoost handles them)
    - One-hot later with dummy_na=True so categorical missing is explicit
    """
    df = df.copy()

    if tz_col not in df.columns:
        raise KeyError(f"Time_Zone column '{tz_col}' not found.")

    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not found.")

    # clean label
    y_full = pd.to_numeric(df[target_col], errors="coerce")
    keep = y_full.notna()
    df = df.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y  = y.loc[ok].copy()

    groups = df[group_col].values
    tz = pd.to_numeric(df[tz_col], errors="coerce").values  # may contain NaN

    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    # bool -> int (not imputation)
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    for c in bool_cols:
        X[c] = X[c].astype("int8")

    return X, y, groups, tz


# -------------------------
# Recreate grouped split (same logic as training)
# -------------------------
def grouped_train_val_test_split(X, y, groups, test_size=0.10, val_size=0.10, random_state=RANDOM_STATE):
    n_splits_test = max(int(round(1.0 / test_size)), 2)
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=random_state)
    trval_idx, test_idx = list(sgkf_test.split(X, y, groups=groups))[0]

    X_trval = X.iloc[trval_idx]
    y_trval = y.iloc[trval_idx]
    g_trval = groups[trval_idx]

    val_frac = val_size / (1.0 - test_size)
    n_splits_val = max(int(round(1.0 / val_frac)), 2)
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=random_state)
    train_rel, val_rel = list(sgkf_val.split(X_trval, y_trval, groups=g_trval))[0]

    train_idx = trval_idx[train_rel]
    val_idx   = trval_idx[val_rel]
    return train_idx, val_idx, test_idx


# -------------------------
# Load model + feature list
# -------------------------
def load_booster_and_features(model_dir: Path, basename: str):
    model_path = model_dir / f"{basename}.json"
    feat_path  = model_dir / f"{basename}_features.txt"
    meta_path  = model_dir / f"{basename}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features: {feat_path}")

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with open(feat_path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]

    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    return booster, features, meta


# -------------------------
# Feature mapping to saved space
# -------------------------
def to_saved_feature_space(X_raw: pd.DataFrame, saved_features: list[str]) -> pd.DataFrame:
    # categorical missing gets its own dummy column; numeric NaN stays NaN
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=True)
    X_oh = X_oh.reindex(columns=saved_features, fill_value=0)
    return X_oh

def predict_proba(booster: xgb.Booster, X: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=feature_names, missing=np.nan)
    return booster.predict(d)


# -------------------------
# ROC / PR plots with multiple curves per Time_Zone
# -------------------------
def tz_label(tz: int) -> str:
    hours = int(tz) * 3
    return f"{hours}h"

def plot_multi_roc(curves, title: str, out_base: Path):
    """
    curves: list of dicts with keys: label, fpr, tpr, auc
    """
    plt.figure()
    for c in curves:
        plt.plot(c["fpr"], c["tpr"], label=f'{c["label"]} AUC={c["auc"]:.3f}')
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=False, ncol=1)
    save_fig_png_pdf(out_base)

def plot_multi_pr(curves, title: str, out_base: Path):
    """
    curves: list of dicts with keys: label, recall, precision, pr_auc
    """
    plt.figure()
    for c in curves:
        plt.plot(c["recall"], c["precision"], label=f'{c["label"]} PR-AUC={c["pr_auc"]:.3f}')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", frameon=False, ncol=1)
    save_fig_png_pdf(out_base)


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    set_plot_style()

    print("Loading saved model + features...")
    booster, saved_features, meta = load_booster_and_features(MODEL_DIR, MODEL_BASENAME)
    print(f"Loaded model. #features={len(saved_features)}")

    print("\nLoading datasets...")
    mimic_df = ensure_binary_label(load_csv(MIMIC_PATH), TARGET_COL)
    eicu_df  = ensure_binary_label(load_csv(EICU_PATH),  TARGET_COL)

    print("Building X/y/groups/tz...")
    X_m_raw, y_m, g_m, tz_m = make_X_y_groups_tz(mimic_df, TARGET_COL, GROUP_COL, TZ_COL, EXCLUDE_FEATURES)
    X_e_raw, y_e, g_e, tz_e = make_X_y_groups_tz(eicu_df,  TARGET_COL, GROUP_COL, TZ_COL, EXCLUDE_FEATURES)

    # Recreate exact split to locate MIMIC test set (same seed + logic)
    print("\nRecreating MIMIC grouped split to locate TEST set...")
    _, _, te_idx = grouped_train_val_test_split(
        X_m_raw, y_m, g_m, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_STATE
    )

    X_test_all_raw = X_m_raw.iloc[te_idx].copy()
    y_test_all     = y_m.iloc[te_idx].values
    tz_test_all    = tz_m[te_idx]

    # Evaluate per Time_Zone
    rows_test = []
    rows_ext  = []

    curves_test_roc = []
    curves_ext_roc  = []
    curves_test_pr  = []
    curves_ext_pr   = []

    print("\nComputing ROC/AUC + PR-AUC per Time_Zone...")
    for tz in TZ_LIST:
        # -----------------
        # MIMIC TEST subset
        # -----------------
        mask_t = (tz_test_all == tz)
        yt = y_test_all[mask_t]
        if len(yt) == 0 or len(np.unique(yt)) < 2:
            print(f"[MIMIC test] {tz_label(tz)} -> skipped (n=0 or single class)")
        else:
            Xt_raw = X_test_all_raw.loc[mask_t].copy()
            Xt = to_saved_feature_space(Xt_raw, saved_features)
            pt = predict_proba(booster, Xt, saved_features)

            auc_t = float(roc_auc_score(yt, pt))
            fpr, tpr, _ = roc_curve(yt, pt)

            pr_auc_t = float(average_precision_score(yt, pt))
            prec, rec, _ = precision_recall_curve(yt, pt)

            curves_test_roc.append({"label": tz_label(tz), "fpr": fpr, "tpr": tpr, "auc": auc_t})
            curves_test_pr.append({"label": tz_label(tz), "recall": rec, "precision": prec, "pr_auc": pr_auc_t})

            rows_test.append({
                "dataset": "MIMIC_test",
                "time_zone": int(tz),
                "hours": int(tz) * 3,
                "n": int(len(yt)),
                "event_rate": float(np.mean(yt)),
                "AUC": auc_t,
                "PR_AUC": pr_auc_t
            })
            print(f"[MIMIC test] {tz_label(tz)} -> n={len(yt)} AUC={auc_t:.4f} PR-AUC={pr_auc_t:.4f}")

        # -----------------
        # eICU EXTERNAL subset
        # -----------------
        mask_e = (tz_e == tz)
        ye = y_e.values[mask_e]
        if len(ye) == 0 or len(np.unique(ye)) < 2:
            print(f"[eICU ext]   {tz_label(tz)} -> skipped (n=0 or single class)")
        else:
            Xe_raw = X_e_raw.loc[mask_e].copy()
            Xe = to_saved_feature_space(Xe_raw, saved_features)
            pe = predict_proba(booster, Xe, saved_features)

            auc_e = float(roc_auc_score(ye, pe))
            fpr, tpr, _ = roc_curve(ye, pe)

            pr_auc_e = float(average_precision_score(ye, pe))
            prec, rec, _ = precision_recall_curve(ye, pe)

            curves_ext_roc.append({"label": tz_label(tz), "fpr": fpr, "tpr": tpr, "auc": auc_e})
            curves_ext_pr.append({"label": tz_label(tz), "recall": rec, "precision": prec, "pr_auc": pr_auc_e})

            rows_ext.append({
                "dataset": "eICU_external",
                "time_zone": int(tz),
                "hours": int(tz) * 3,
                "n": int(len(ye)),
                "event_rate": float(np.mean(ye)),
                "AUC": auc_e,
                "PR_AUC": pr_auc_e
            })
            print(f"[eICU ext]   {tz_label(tz)} -> n={len(ye)} AUC={auc_e:.4f} PR-AUC={pr_auc_e:.4f}")

    # Save tables (κρατάω τα ίδια filenames, αλλά πλέον έχουν και PR_AUC στήλη)
    df_test = pd.DataFrame(rows_test).sort_values(["time_zone"])
    df_ext  = pd.DataFrame(rows_ext).sort_values(["time_zone"])

    test_csv = OUT_DIR / "auc_by_time_zone_mimic_test.csv"
    ext_csv  = OUT_DIR / "auc_by_time_zone_eicu_external.csv"
    df_test.to_csv(test_csv, index=False)
    df_ext.to_csv(ext_csv, index=False)
    print("\nSaved:", test_csv)
    print("Saved:", ext_csv)

    # ROC plots
    if len(curves_test_roc) > 0:
        plot_multi_roc(
            curves_test_roc,
            title=f"ROC by Time_Zone — MIMIC test (TZ in {TZ_LIST})",
            out_base=OUT_DIR / "roc_by_time_zone_mimic_test"
        )
        print("Saved ROC:", OUT_DIR / "roc_by_time_zone_mimic_test.(png/pdf)")

    if len(curves_ext_roc) > 0:
        plot_multi_roc(
            curves_ext_roc,
            title=f"ROC by Time_Zone — eICU external (TZ in {TZ_LIST})",
            out_base=OUT_DIR / "roc_by_time_zone_eicu_external"
        )
        print("Saved ROC:", OUT_DIR / "roc_by_time_zone_eicu_external.(png/pdf)")

    # PR plots
    if len(curves_test_pr) > 0:
        plot_multi_pr(
            curves_test_pr,
            title=f"PR by Time_Zone — MIMIC test (TZ in {TZ_LIST})",
            out_base=OUT_DIR / "pr_by_time_zone_mimic_test"
        )
        print("Saved PR:", OUT_DIR / "pr_by_time_zone_mimic_test.(png/pdf)")

    if len(curves_ext_pr) > 0:
        plot_multi_pr(
            curves_ext_pr,
            title=f"PR by Time_Zone — eICU external (TZ in {TZ_LIST})",
            out_base=OUT_DIR / "pr_by_time_zone_eicu_external"
        )
        print("Saved PR:", OUT_DIR / "pr_by_time_zone_eicu_external.(png/pdf)")

    # Optional: combined ROC plot (both datasets)
    if len(curves_test_roc) > 0 and len(curves_ext_roc) > 0:
        plt.figure()
        for c in curves_test_roc:
            plt.plot(c["fpr"], c["tpr"], label=f'MIMIC {c["label"]} AUC={c["auc"]:.3f}')
        for c in curves_ext_roc:
            plt.plot(c["fpr"], c["tpr"], linestyle="--", label=f'eICU {c["label"]} AUC={c["auc"]:.3f}')
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC — MIMIC test vs eICU external")
        plt.legend(loc="lower right", frameon=False, ncol=1)
        save_fig_png_pdf(OUT_DIR / "roc_by_time_zone_combined")
        print("Saved ROC:", OUT_DIR / "roc_by_time_zone_combined.(png/pdf)")

    # Optional: combined PR plot (both datasets)
    if len(curves_test_pr) > 0 and len(curves_ext_pr) > 0:
        plt.figure()
        for c in curves_test_pr:
            plt.plot(c["recall"], c["precision"], label=f'MIMIC {c["label"]} PR-AUC={c["pr_auc"]:.3f}')
        for c in curves_ext_pr:
            plt.plot(c["recall"], c["precision"], linestyle="--", label=f'eICU {c["label"]} PR-AUC={c["pr_auc"]:.3f}')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR — MIMIC test vs eICU external")
        plt.legend(loc="lower left", frameon=False, ncol=1)
        save_fig_png_pdf(OUT_DIR / "pr_by_time_zone_combined")
        print("Saved PR:", OUT_DIR / "pr_by_time_zone_combined.(png/pdf)")

    print("\nDone. Outputs in:", OUT_DIR)
