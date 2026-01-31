# xgb_optuna_no_imputation_holdout_test.py
# ============================================================
# HOLDOUT TEST independent from 5-fold CV (NO leakage)
#
# 1) Outer split (MIMIC): DEV / TEST using StratifiedGroupKFold (groups=subject_id)
#    - TEST is never used in Optuna CV nor in training/early-stopping.
# 2) Optuna tuning: 5-fold StratifiedGroupKFold CV on DEV only
# 3) Final model: train on DEV-TRAIN, early stop on DEV-VAL (both inside DEV)
# 4) Evaluate: MIMIC TEST + eICU EXTERNAL
#
# IMPORTANT: NO IMPUTATION
# - numeric NaNs remain NaN (XGBoost handles them)
# - categorical NaNs encoded with dummy_na=True
# - no object->string casting (avoids NaN -> "nan")
#
# Excluded from training features:
# - row_count, subject_id, hadm_id, Time_Zone, los, hospital_expire_flag (label)
# ============================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

import xgboost as xgb
import optuna


# ============================================================
# CONFIG
# ============================================================
MIMIC_PATH = "CSV/o01_mimic_for_ext_val.csv"
EICU_PATH  = "CSV/o01_eicu_for_ext_val.csv"

RANDOM_STATE = 42
TARGET_COL   = "hospital_expire_flag"
GROUP_COL    = "subject_id"

# MUST NOT be used as features
EXCLUDE_FEATURES = [
    "row_count",
    "subject_id",
    "hadm_id",
    "Time_Zone",
    "los",
    TARGET_COL,
]

OUT_DIR = Path("CSV/Exports/Temp/13_ML_inhospital_mortality_optuna")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Outer holdout (MIMIC)
TEST_SIZE = 0.10  # holdout test (independent)

# Inside DEV: early-stopping validation (still independent from TEST)
DEV_VAL_SIZE = 0.10  # fraction of DEV

# Optuna (on DEV only)
N_TRIALS = 100
CV_FOLDS = 5
OPTUNA_TIMEOUT = None  # for example 7000 but I'm on server

# XGB training
NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 200
CHUNK_SIZE = 50

# Bootstrap (final metrics on TEST/EXTERNAL)
N_BOOT = 2000
ALPHA  = 0.05

# SHAP (I have put a 2000 sample. I create separate code for all rows)
RUN_SHAP = True
SHAP_SAMPLE_SIZE = 2000
SHAP_MAX_DISPLAY = 25

MODEL_BASENAME = "xgb_inhospital_mortality_optuna"


# ============================================================
# Plot style
# ============================================================
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


# ============================================================
# IO + label
# ============================================================
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


# ============================================================
# Build X/y/groups (NO IMPUTATION)
# ============================================================
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
    - dummy_na=True in one-hot later
    """
    df = df.copy()

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found.")

    # Label cleaning only
    y_full = pd.to_numeric(df[target_col], errors="coerce")
    keep = y_full.notna()
    df = df.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y  = y.loc[ok].copy()

    # Groups
    if group_col in df.columns:
        groups = df[group_col].values
    else:
        if require_group:
            raise KeyError(f"Group column '{group_col}' not found.")
        groups = np.arange(len(df), dtype=int)

    # Features
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    # bool -> int (not imputation)
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    for c in bool_cols:
        X[c] = X[c].astype("int8")

    # Guard: excluded columns must NOT appear
    leaked = [c for c in exclude if c in X.columns]
    if leaked:
        raise RuntimeError(f"Excluded columns leaked into X: {leaked}")

    return X, y, groups

def one_hot_fit(X_raw: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X_raw, drop_first=False, dummy_na=True)

def one_hot_transform(X_raw: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=True)
    return X_oh.reindex(columns=train_columns, fill_value=0)

def assert_numeric_nans_preserved(X_raw: pd.DataFrame, X_oh: pd.DataFrame, where: str):
    numeric_cols = [c for c in X_raw.columns if pd.api.types.is_numeric_dtype(X_raw[c])]
    common = [c for c in numeric_cols if c in X_oh.columns]
    for c in common:
        raw_mask = X_raw[c].isna().to_numpy()
        oh_mask  = X_oh[c].isna().to_numpy()
        if raw_mask.shape != oh_mask.shape or not np.array_equal(raw_mask, oh_mask):
            raise RuntimeError(f"[NO-IMPUTATION GUARD FAIL] Numeric NaNs changed in '{c}' at {where}.")


# ============================================================
# Splits
# ============================================================
def _pick_best_sgkf_holdout(
    X, y, groups, holdout_size: float, random_state: int
):
    """
    Creates a stratified+grouped holdout by trying all SGKF folds
    and picking the one closest to target size + target pos_rate.
    """
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

        # size term + pos-rate term
        size_term = abs(n_ho - target_n) / max(target_n, 1)
        pos_term  = abs(pos_ho - target_pos) / max(target_pos, 1e-6)
        score = size_term + pos_term

        if score < best_score:
            best_score = score
            best = (tr_idx, ho_idx)

    return best[0], best[1]

def split_mimic_dev_test(X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    dev_idx, test_idx = _pick_best_sgkf_holdout(X, y, groups, holdout_size=test_size, random_state=random_state)
    return np.asarray(dev_idx), np.asarray(test_idx)

def split_dev_train_val(X_dev, y_dev, g_dev, val_size=DEV_VAL_SIZE, random_state=RANDOM_STATE):
    tr_idx, va_idx = _pick_best_sgkf_holdout(X_dev, y_dev, g_dev, holdout_size=val_size, random_state=random_state)
    return np.asarray(tr_idx), np.asarray(va_idx)

def assert_no_group_overlap(*sets_of_groups):
    """
    Each argument is an array-like of group ids (subject_id) for a set.
    Ensures pairwise disjointness.
    """
    sets = [set(map(int, np.asarray(g).tolist())) for g in sets_of_groups]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if sets[i] & sets[j]:
                raise RuntimeError("Group leakage detected: same subject_id appears in multiple sets.")


# ============================================================
# XGB helpers
# ============================================================
def base_scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return (neg / max(pos, 1.0))

def train_booster_manual_es(
    params: dict,
    X_train, y_train,
    X_val, y_val,
    num_boost_round=NUM_BOOST_ROUND,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    chunk_size=CHUNK_SIZE
):
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns), missing=np.nan)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=list(X_val.columns),   missing=np.nan)

    booster = None
    best_auc = -np.inf
    best_iter = -1
    current_round = 0

    pbar = tqdm(total=num_boost_round, desc="Final XGBoost training", unit="round")
    while current_round < num_boost_round:
        step = min(chunk_size, num_boost_round - current_round)
        evals_result = {}

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=step,
            evals=[(dval, "val")],
            xgb_model=booster,
            evals_result=evals_result,
            verbose_eval=False
        )

        try:
            auc_list = evals_result["val"]["auc"]
        except Exception:
            auc_list = None

        if auc_list:
            auc_vals = np.array([float(v) for v in auc_list], dtype=float)
            chunk_best_idx = int(np.argmax(auc_vals))
            chunk_best_auc = float(auc_vals[chunk_best_idx])
            if chunk_best_auc > best_auc + 1e-12:
                best_auc = chunk_best_auc
                best_iter = current_round + chunk_best_idx
            last_auc = float(auc_vals[-1])
        else:
            last_auc = float("nan")

        current_round += step
        pbar.update(step)

        since_best = current_round - (best_iter + 1)
        pbar.set_postfix({
            "val_auc_last": f"{last_auc:.4f}" if np.isfinite(last_auc) else "nan",
            "best_auc": f"{best_auc:.4f}" if np.isfinite(best_auc) else "nan",
            "best_iter": best_iter,
            "since_best": since_best
        })

        if best_iter >= 0 and since_best >= early_stopping_rounds:
            break

    pbar.close()

    try:
        booster = booster[: best_iter + 1]
    except Exception:
        pass

    return booster, best_iter, best_auc

def predict_proba(booster: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=list(X.columns), missing=np.nan)
    return booster.predict(d)


# ============================================================
# Metrics + CI (I try to group the bootstrap by subject but it crash)
# ============================================================
def bootstrap_ci(y_true, p_pred, metric_fn, n_boot=N_BOOT, alpha=ALPHA, seed=RANDOM_STATE, desc="bootstrap"):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p_pred = np.asarray(p_pred)

    point = metric_fn(y_true, p_pred)
    vals = []
    n = len(y_true)

    for _ in tqdm(range(n_boot), desc=desc, leave=False):
        idx = rng.integers(0, n, size=n)
        yb = y_true[idx]
        pb = p_pred[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(metric_fn(yb, pb))

    if len(vals) < 50:
        return float(point), np.nan, np.nan

    vals = np.sort(vals)
    lo = np.quantile(vals, alpha / 2)
    hi = np.quantile(vals, 1 - alpha / 2)
    return float(point), float(lo), float(hi)


# ============================================================
# Plots
# ============================================================
def plot_roc(y_true, p_pred, title, out_base: Path, label: str):
    fpr, tpr, _ = roc_curve(y_true, p_pred)
    auc = roc_auc_score(y_true, p_pred)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{label} AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)
    save_fig_png_pdf(out_base)

def plot_pr(y_true, p_pred, title, out_base: Path, label: str):
    prec, rec, _ = precision_recall_curve(y_true, p_pred)
    ap = average_precision_score(y_true, p_pred)

    plt.figure()
    plt.plot(rec, prec, label=f"{label} PR-AUC={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", frameon=False)
    save_fig_png_pdf(out_base)

def plot_roc_both(y_test, p_test, y_ext, p_ext, out_base: Path):
    fpr_t, tpr_t, _ = roc_curve(y_test, p_test)
    fpr_e, tpr_e, _ = roc_curve(y_ext,  p_ext)
    auc_t = roc_auc_score(y_test, p_test)
    auc_e = roc_auc_score(y_ext,  p_ext)

    plt.figure()
    plt.plot(fpr_t, tpr_t, label=f"MIMIC TEST AUC={auc_t:.3f}")
    plt.plot(fpr_e, tpr_e, label=f"eICU EXT AUC={auc_e:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC: MIMIC TEST vs eICU EXTERNAL")
    plt.legend(loc="lower right", frameon=False)
    save_fig_png_pdf(out_base)

def plot_pr_both(y_test, p_test, y_ext, p_ext, out_base: Path):
    prec_t, rec_t, _ = precision_recall_curve(y_test, p_test)
    prec_e, rec_e, _ = precision_recall_curve(y_ext,  p_ext)
    ap_t = average_precision_score(y_test, p_test)
    ap_e = average_precision_score(y_ext,  p_ext)

    plt.figure()
    plt.plot(rec_t, prec_t, label=f"MIMIC TEST PR-AUC={ap_t:.3f}")
    plt.plot(rec_e, prec_e, label=f"eICU EXT PR-AUC={ap_e:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR: MIMIC TEST vs eICU EXTERNAL")
    plt.legend(loc="lower left", frameon=False)
    save_fig_png_pdf(out_base)


# ============================================================
# Save artifacts
# ============================================================
def save_model_and_metadata(booster: xgb.Booster,
                            feature_names: list[str],
                            best_params: dict,
                            out_dir: Path,
                            basename: str):
    model_path = out_dir / f"{basename}.json"
    booster.save_model(str(model_path))

    feat_path = out_dir / f"{basename}_features.txt"
    with open(feat_path, "w", encoding="utf-8") as f:
        for c in feature_names:
            f.write(f"{c}\n")

    meta_path = out_dir / f"{basename}_meta.json"
    meta = {
        "xgboost_version": xgb.__version__,
        "optuna_best_params": best_params,
        "random_state": int(RANDOM_STATE),
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "excluded_from_training": EXCLUDE_FEATURES,
        "missing_handling": {
            "numeric": "kept as NaN (no imputation)",
            "categorical": "dummy_na=True (explicit missing dummy)"
        },
        "model_file": model_path.name,
        "features_file": feat_path.name,
        "num_features": int(len(feature_names)),
        "splitting": {
            "outer": "DEV/TEST with StratifiedGroupKFold (TEST independent of CV)",
            "inner_optuna_cv": f"{CV_FOLDS}-fold StratifiedGroupKFold on DEV only",
            "final_es_split": "DEV split into DEV-TRAIN/DEV-VAL (groups disjoint)"
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Saved model   :", model_path)
    print("Saved features:", feat_path)
    print("Saved metadata:", meta_path)


# ============================================================
# SHAP
# ============================================================
def run_shap_and_save(booster: xgb.Booster, X_train_oh: pd.DataFrame, out_dir: Path):
    import shap

    rng = np.random.default_rng(RANDOM_STATE)
    n = min(SHAP_SAMPLE_SIZE, len(X_train_oh))
    idx = rng.choice(len(X_train_oh), size=n, replace=False)
    X_shap = X_train_oh.iloc[idx].copy()

    print(f"\nComputing SHAP on sample n={n} ...")
    explainer = shap.TreeExplainer(booster)

    try:
        shap_values = explainer.shap_values(X_shap)
        imp = np.mean(np.abs(shap_values), axis=0)
        imp_df = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
        imp_df.to_csv(out_dir / "shap_importance_mean_abs.csv", index=False)

        shap.summary_plot(shap_values, X_shap, show=False, max_display=SHAP_MAX_DISPLAY)
        plt.savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / "shap_summary.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
        plt.savefig(out_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / "shap_bar.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    except Exception:
        shap_exp = explainer(X_shap)
        try:
            imp = np.mean(np.abs(shap_exp.values), axis=0)
            imp_df = pd.DataFrame({"feature": X_shap.columns, "mean_abs_shap": imp}).sort_values("mean_abs_shap", ascending=False)
            imp_df.to_csv(out_dir / "shap_importance_mean_abs.csv", index=False)
        except Exception:
            pass

        shap.summary_plot(shap_exp, show=False, max_display=SHAP_MAX_DISPLAY)
        plt.savefig(out_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / "shap_summary.pdf", dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_exp, plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
        plt.savefig(out_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_dir / "shap_bar.pdf", dpi=300, bbox_inches="tight")
        plt.close()

    print("Saved SHAP plots in:", out_dir)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("xgboost version:", xgb.__version__)

    print("\n[1/6] Load datasets...")
    mimic_df = ensure_binary_label(load_csv(MIMIC_PATH), TARGET_COL)
    eicu_df  = ensure_binary_label(load_csv(EICU_PATH),  TARGET_COL)

    print("[2/6] Build X/y/groups (NO IMPUTATION, excluded cols removed)...")
    X_m, y_m, g_m = make_X_y_groups(mimic_df, TARGET_COL, GROUP_COL, EXCLUDE_FEATURES, require_group=True)
    X_e_raw, y_e, _ = make_X_y_groups(eicu_df, TARGET_COL, GROUP_COL, EXCLUDE_FEATURES, require_group=False)

    print("[3/6] OUTER split: DEV / TEST (TEST independent of CV)...")
    dev_idx, test_idx = split_mimic_dev_test(X_m, y_m, g_m, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_dev_raw = X_m.iloc[dev_idx].copy()
    y_dev     = y_m.iloc[dev_idx].copy()
    g_dev     = g_m[dev_idx]

    X_test_raw = X_m.iloc[test_idx].copy()
    y_test     = y_m.iloc[test_idx].copy()
    g_test     = g_m[test_idx]

    # Ensure no subject overlap DEV vs TEST
    assert_no_group_overlap(g_dev, g_test)

    print("\nMIMIC OUTER SPLIT:")
    print(f"  DEV : n={len(y_dev)} | pos_rate={y_dev.mean():.4f} | unique_subjects={len(set(g_dev))}")
    print(f"  TEST: n={len(y_test)} | pos_rate={y_test.mean():.4f} | unique_subjects={len(set(g_test))}")

    print("\n[4/6] Optuna tuning: 5-fold StratifiedGroupKFold CV on DEV only...")
    base_spw = base_scale_pos_weight(y_dev.values)
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "seed": RANDOM_STATE,
            "verbosity": 0,

            "eta": trial.suggest_float("eta", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "lambda": trial.suggest_float("lambda", 1e-3, 20.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-3, 20.0, log=True),

            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.6 * base_spw, 1.6 * base_spw),
        }

        aucs = []
        for fold, (i_tr, i_va) in enumerate(cv.split(X_dev_raw, y_dev, groups=g_dev)):
            X_tr_raw = X_dev_raw.iloc[i_tr].copy()
            y_tr = y_dev.iloc[i_tr].values
            X_va_raw = X_dev_raw.iloc[i_va].copy()
            y_va = y_dev.iloc[i_va].values

            # One-hot per fold (NO IMPUTATION)
            X_tr = pd.get_dummies(X_tr_raw, drop_first=False, dummy_na=True)
            X_va = pd.get_dummies(X_va_raw, drop_first=False, dummy_na=True)
            X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

            # Guards: numeric NaNs preserved
            assert_numeric_nans_preserved(X_tr_raw, X_tr, where=f"Optuna fold {fold} train")
            assert_numeric_nans_preserved(X_va_raw, X_va, where=f"Optuna fold {fold} val")

            dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns), missing=np.nan)
            dva = xgb.DMatrix(X_va, label=y_va, feature_names=list(X_va.columns), missing=np.nan)

            booster = xgb.train(
                params=params,
                dtrain=dtr,
                num_boost_round=2500,
                evals=[(dva, "val")],
                early_stopping_rounds=100,
                verbose_eval=False
            )

            preds = booster.predict(dva)
            auc = roc_auc_score(y_va, preds)
            aucs.append(float(auc))

            trial.report(float(np.mean(aucs)), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(aucs))

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
    )

    with tqdm(total=N_TRIALS, desc="Optuna trials") as pbar:
        def _cb(_study, _trial):
            pbar.update(1)
        study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, callbacks=[_cb], gc_after_trial=True)

    # Save trials
    try:
        study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)
    except Exception:
        pass

    best_params = study.best_trial.params
    print("\nBest CV AUC (DEV CV):", study.best_value)
    print("Best params:", best_params)

    final_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr"],
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "verbosity": 0,

        "eta": float(best_params["eta"]),
        "max_depth": int(best_params["max_depth"]),
        "min_child_weight": float(best_params["min_child_weight"]),
        "subsample": float(best_params["subsample"]),
        "colsample_bytree": float(best_params["colsample_bytree"]),
        "gamma": float(best_params["gamma"]),
        "lambda": float(best_params["lambda"]),
        "alpha": float(best_params["alpha"]),
        "scale_pos_weight": float(best_params["scale_pos_weight"]),
    }

    print("\n[5/6] Final training inside DEV (DEV-TRAIN / DEV-VAL), TEST untouched...")
    dev_tr_idx, dev_va_idx = split_dev_train_val(X_dev_raw, y_dev, g_dev, val_size=DEV_VAL_SIZE, random_state=RANDOM_STATE)

    X_dev_tr_raw = X_dev_raw.iloc[dev_tr_idx].copy()
    y_dev_tr = y_dev.iloc[dev_tr_idx].copy()
    g_dev_tr = g_dev[dev_tr_idx]

    X_dev_va_raw = X_dev_raw.iloc[dev_va_idx].copy()
    y_dev_va = y_dev.iloc[dev_va_idx].copy()
    g_dev_va = g_dev[dev_va_idx]

    # Ensure no subject overlap among DEV-TRAIN / DEV-VAL / TEST
    assert_no_group_overlap(g_dev_tr, g_dev_va, g_test)

    print("\nDEV INNER SPLIT (for early stopping only):")
    print(f"  DEV-TRAIN: n={len(y_dev_tr)} | pos_rate={y_dev_tr.mean():.4f} | unique_subjects={len(set(g_dev_tr))}")
    print(f"  DEV-VAL  : n={len(y_dev_va)} | pos_rate={y_dev_va.mean():.4f} | unique_subjects={len(set(g_dev_va))}")
    print(f"  TEST     : n={len(y_test)}   | pos_rate={y_test.mean():.4f} | unique_subjects={len(set(g_test))}")

    # One-hot fitted on DEV-TRAIN (final model feature space)
    X_train = one_hot_fit(X_dev_tr_raw)
    train_cols = list(X_train.columns)

    X_val   = one_hot_transform(X_dev_va_raw, train_cols)
    X_test  = one_hot_transform(X_test_raw,   train_cols)
    X_ext   = one_hot_transform(X_e_raw,      train_cols)

    # Guards: numeric NaNs preserved
    assert_numeric_nans_preserved(X_dev_tr_raw, X_train, where="final DEV-TRAIN one-hot")
    assert_numeric_nans_preserved(X_dev_va_raw, X_val,   where="final DEV-VAL one-hot")
    assert_numeric_nans_preserved(X_test_raw,   X_test,  where="final TEST one-hot")
    assert_numeric_nans_preserved(X_e_raw,      X_ext,   where="final EXTERNAL one-hot")

    booster, best_iter, best_val_auc = train_booster_manual_es(
        params=final_params,
        X_train=X_train, y_train=y_dev_tr.values,
        X_val=X_val,     y_val=y_dev_va.values,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        chunk_size=CHUNK_SIZE
    )
    print(f"Final model: best_iter={best_iter}, best_dev_val_auc={best_val_auc:.4f}")

    # Save model + metadata (includes split strategy)
    save_model_and_metadata(
        booster=booster,
        feature_names=train_cols,
        best_params={"optuna_best_dev_cv_auc": float(study.best_value), **best_params},
        out_dir=OUT_DIR,
        basename=MODEL_BASENAME
    )

    print("\n[6/6] Evaluation ONLY on TEST + eICU external...")
    p_test = predict_proba(booster, X_test)
    p_ext  = predict_proba(booster, X_ext)

    def _auc(y, p):   return roc_auc_score(y, p)
    def _ap(y, p):    return average_precision_score(y, p)
    def _brier(y, p): return brier_score_loss(y, p)

    rows = []
    for name, yy, pp in [("MIMIC_test", y_test.values, p_test),
                         ("eICU_ext",   y_e.values,    p_ext)]:
        auc, auc_lo, auc_hi = bootstrap_ci(yy, pp, _auc,   desc=f"{name} | AUC boot")
        ap,  ap_lo,  ap_hi  = bootstrap_ci(yy, pp, _ap,    desc=f"{name} | PR-AUC boot")
        br,  br_lo,  br_hi  = bootstrap_ci(yy, pp, _brier, desc=f"{name} | Brier boot")
        rows.append({
            "dataset": name,
            "n": int(len(yy)),
            "pos_rate": float(np.mean(yy)),
            "AUC": auc, "AUC_CI_low": auc_lo, "AUC_CI_high": auc_hi,
            "PR_AUC": ap, "PR_AUC_CI_low": ap_lo, "PR_AUC_CI_high": ap_hi,
            "Brier": br, "Brier_CI_low": br_lo, "Brier_CI_high": br_hi,
        })

    metrics_df = pd.DataFrame(rows)
    metrics_path = OUT_DIR / "metrics_inhospital_mortality.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print("\nSaved:", metrics_path)
    print(metrics_df.to_string(index=False))

    # Plots
    plot_roc(y_test.values, p_test, "ROC (MIMIC TEST)", OUT_DIR / "roc_mimic_test", "MIMIC TEST")
    plot_pr (y_test.values, p_test, "PR (MIMIC TEST)",  OUT_DIR / "pr_mimic_test",  "MIMIC TEST")

    plot_roc(y_e.values,    p_ext,  "ROC (eICU EXTERNAL)", OUT_DIR / "roc_eicu_external", "eICU EXT")
    plot_pr (y_e.values,    p_ext,  "PR (eICU EXTERNAL)",  OUT_DIR / "pr_eicu_external",  "eICU EXT")

    plot_roc_both(y_test.values, p_test, y_e.values, p_ext, OUT_DIR / "roc_test_vs_external")
    plot_pr_both (y_test.values, p_test, y_e.values, p_ext, OUT_DIR / "pr_test_vs_external")

    # SHAP (optional but important)
    if RUN_SHAP:
        try:
            run_shap_and_save(booster, X_train, OUT_DIR)
        except Exception as e:
            print("SHAP failed:", repr(e))

    print("\nDone. Outputs in:", OUT_DIR)
