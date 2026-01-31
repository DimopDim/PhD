# make_grouped_shap_from_saved_xgb.py
# ============================================================
# Load a saved XGBoost model (+ feature list) and compute SHAP.
# Then aggregate (sum) SHAP contributions for features that share
# the same clinical "base name" by collapsing the final suffix:
#   _(Max), _(Mean), _(Median), _(Min)
#
# This is analogous to "sum of SHAP values per feature group"
# (group-level SHAP aggregation).
#
# Outputs (in OUT_DIR):
# - shap_grouped_values.npy              : (n_samples, n_groups)
# - shap_grouped_feature_values.csv      : feature values used for coloring (n_samples x n_groups)
# - shap_grouped_importance_mean_abs.csv : mean(|SHAP|) per group (ranked)
# - shap_grouped_groups.json             : mapping group -> original one-hot columns
# - shap_grouped_summary.(png/pdf)       : beeswarm plot (grouped)
# - shap_grouped_bar.(png/pdf)           : bar importance plot (grouped)
# ============================================================

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb
import shap


# -------------------------
# CONFIG (EDIT THESE)
# -------------------------
MODEL_DIR = Path("CSV/Exports/Temp/13_ML_inhospital_mortality_optuna")
MODEL_BASENAME = "xgb_inhospital_mortality_optuna"

# Data to explain (can be MIMIC or eICU)
DATA_PATH = "CSV/o01_mimic_for_ext_val.csv"   # or "CSV/o01_eicu_for_ext_val.csv"

TARGET_COL = "hospital_expire_flag"

EXCLUDE_FEATURES = [
    "row_count",
    "subject_id",
    "hadm_id",
    "Time_Zone",
    "los",
    TARGET_COL,
]

# SHAP sampling
RANDOM_STATE = 42
SHAP_SAMPLE_SIZE = 90000   # rows sampled from DATA_PATH for SHAP
SHAP_MAX_DISPLAY = 25

# How to compute grouped "feature values" for beeswarm coloring:
# - "mean": average of the member columns (this is my default)
# - "max" : max of the member columns
# - "min" : min of the member columns
GROUP_VALUE_AGG = "mean"

OUT_DIR = MODEL_DIR / "shap_grouped"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------
_SUFFIX_RE = re.compile(r'_\((Max|Mean|Median|Min)\)$')

def base_feature_name(col: str) -> str:
    """
    Collapse the *final* suffix _(Max|Mean|Median|Min).
    Examples:
      Albumin_(Max) -> Albumin
      Alanine_Aminotransferase_(ALT)_(Max) -> Alanine_Aminotransferase_(ALT)
      gender_Male -> gender_Male (unchanged)
    """
    return _SUFFIX_RE.sub("", col)

def ensure_binary_label(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.copy()
    if target_col in df.columns:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        return df

    # fallback (useful for eICU)
    if "unitdischargestatus" in df.columns:
        m = {"Alive": 0, "alive": 0, "Expired": 1, "expired": 1, "Dead": 1, "dead": 1}
        df[target_col] = df["unitdischargestatus"].astype(str).map(m)
        return df

    raise KeyError(f"Δεν βρέθηκε label '{target_col}' ούτε fallback.")

def make_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    - Drop rows with missing/invalid label only
    - Do NOT impute feature NaNs
    - Remove excluded columns
    """
    df = df.copy()
    y_full = pd.to_numeric(df[TARGET_COL], errors="coerce")
    keep = y_full.notna()
    df = df.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y  = y.loc[ok].copy()

    X = df.drop(columns=[c for c in EXCLUDE_FEATURES if c in df.columns], errors="ignore")

    # bool -> int
    for c in X.columns:
        if X[c].dtype == "bool":
            X[c] = X[c].astype("int8")

    return X, y

def one_hot_align(X_raw: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    """
    Use dummy_na=True, then align to the training feature space from *_features.txt
    """
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=True)
    return X_oh.reindex(columns=train_columns, fill_value=0)

def group_shap_by_base_name(
    X_oh: pd.DataFrame,
    shap_values: np.ndarray,
    agg_values: str = "mean",
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Returns:
      grouped_X   : DataFrame (n_samples, n_groups) for coloring
      grouped_sv  : np.ndarray (n_samples, n_groups) summed SHAP
      groups      : dict group -> list of member columns
    """
    cols = list(X_oh.columns)
    base_names = [base_feature_name(c) for c in cols]

    groups: dict[str, list[int]] = {}
    for j, bn in enumerate(base_names):
        groups.setdefault(bn, []).append(j)

    # sum SHAP across member columns
    grouped_sv = np.column_stack([shap_values[:, idxs].sum(axis=1) for idxs in groups.values()])

    # build grouped_X for coloring
    if agg_values == "mean":
        grouped_X = pd.DataFrame({g: X_oh.iloc[:, idxs].mean(axis=1) for g, idxs in groups.items()})
    elif agg_values == "max":
        grouped_X = pd.DataFrame({g: X_oh.iloc[:, idxs].max(axis=1) for g, idxs in groups.items()})
    elif agg_values == "min":
        grouped_X = pd.DataFrame({g: X_oh.iloc[:, idxs].min(axis=1) for g, idxs in groups.items()})
    else:
        raise ValueError("agg_values must be one of: 'mean', 'max', 'min'")

    # mapping group -> original member column names
    groups_named = {g: [cols[i] for i in idxs] for g, idxs in groups.items()}

    return grouped_X, grouped_sv, groups_named

def save_fig_png_pdf(path_base: Path):
    plt.tight_layout()
    plt.savefig(path_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    model_path = MODEL_DIR / f"{MODEL_BASENAME}.json"
    feat_path  = MODEL_DIR / f"{MODEL_BASENAME}_features.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing features file: {feat_path}")

    # Load feature space (the one-hot columns used for training)
    with open(feat_path, "r", encoding="utf-8") as f:
        train_cols = [line.strip() for line in f if line.strip()]

    # Load model
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    # Load data
    df = pd.read_csv(DATA_PATH)
    df = ensure_binary_label(df, TARGET_COL)

    X_raw, y = make_X_y(df)
    X_oh = one_hot_align(X_raw, train_cols)

    # Sample rows for SHAP
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SHAP_SAMPLE_SIZE, len(X_oh))
    idx = rng.choice(len(X_oh), size=n, replace=False)
    X_shap = X_oh.iloc[idx].copy()

    print(f"Loaded model: {model_path.name}")
    print(f"Data: {DATA_PATH} | rows={len(X_oh)} | SHAP sample n={n}")
    print(f"One-hot features: {X_oh.shape[1]}")

    # Compute SHAP
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_shap)

    # Binary classification sometimes returns list [class0, class1]
    if isinstance(sv, list) and len(sv) >= 2:
        sv = sv[1]
    sv = np.asarray(sv)

    # Group by base feature name
    grouped_X, grouped_sv, groups = group_shap_by_base_name(
        X_shap, sv, agg_values=GROUP_VALUE_AGG
    )

    # Save mapping
    with open(OUT_DIR / "shap_grouped_groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

    # Save arrays / values
    np.save(OUT_DIR / "shap_grouped_values.npy", grouped_sv)
    grouped_X.to_csv(OUT_DIR / "shap_grouped_feature_values.csv", index=False)

    # Importance table
    imp = np.mean(np.abs(grouped_sv), axis=0)
    imp_df = pd.DataFrame({
        "group_feature": grouped_X.columns,
        "mean_abs_shap": imp
    }).sort_values("mean_abs_shap", ascending=False)
    imp_df.to_csv(OUT_DIR / "shap_grouped_importance_mean_abs.csv", index=False)
    
    
    TOP_N = 25  # ό,τι θες να φαίνεται
    imp = np.mean(np.abs(grouped_sv), axis=0)
    top_idx = np.argsort(imp)[::-1][:TOP_N]
    
    grouped_sv = grouped_sv[:, top_idx]
    grouped_X  = grouped_X.iloc[:, top_idx]
    

    # Build SHAP Explanation for plotting
    grouped_exp = shap.Explanation(
        values=grouped_sv,
        data=grouped_X.values,
        feature_names=list(grouped_X.columns),
    )

    # Beeswarm
    shap.plots.beeswarm(grouped_exp, max_display=SHAP_MAX_DISPLAY, show=False)
    save_fig_png_pdf(OUT_DIR / "shap_grouped_summary")

    # Bar
    shap.plots.bar(grouped_exp, max_display=SHAP_MAX_DISPLAY, show=False)
    save_fig_png_pdf(OUT_DIR / "shap_grouped_bar")

    print("\nSaved grouped SHAP outputs to:", OUT_DIR.resolve())
    print("Top groups:")
    print(imp_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
