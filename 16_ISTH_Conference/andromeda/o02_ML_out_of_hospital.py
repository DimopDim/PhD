# xgb_optuna_no_imputation_holdout_test_out_of_hospital.py
# ============================================================
# Out-of-hospital Mortality @30/180/360 days (Single MIMIC dataset)
#
# SAME STRUCTURE as your in-hospital script:
# 1) Outer split (MIMIC): DEV / TEST using StratifiedGroupKFold (groups=subject_id)
# 2) Optuna tuning: 5-fold StratifiedGroupKFold CV on DEV only
# 3) Final model: train on DEV-TRAIN, early stop on DEV-VAL
# 4) Evaluate: MIMIC TEST only (no external)
#
# IMPORTANT: NO IMPUTATION
# - numeric NaNs remain NaN
# - categorical one-hot WITHOUT dummy_na (NO missing-indicator columns)
#
# Excluded from training features:
# - row_count, subject_id, hadm_id, Time_Zone, los, hospital_expire_flag
# - PLUS all event_* and duration_* columns (to avoid leakage)
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
MIMIC_PATH = "CSV/o01_mimic_out_of_hospital.csv"  # <-- άλλαξέ το

RANDOM_STATE = 42
GROUP_COL    = "subject_id"

HORIZONS = [30, 180, 360]  # days

# Outer holdout (MIMIC)
TEST_SIZE = 0.10

# Inside DEV: early-stopping validation
DEV_VAL_SIZE = 0.10

# Optuna (on DEV only)
N_TRIALS = 100
CV_FOLDS = 5
OPTUNA_TIMEOUT = None

# XGB training
NUM_BOOST_ROUND = 4000
EARLY_STOPPING_ROUNDS = 200
CHUNK_SIZE = 50

# Bootstrap (final metrics on TEST)
N_BOOT = 2000
ALPHA  = 0.05

# SHAP
RUN_SHAP = True
SHAP_SAMPLE_SIZE = 2000
SHAP_MAX_DISPLAY = 25

# NO missing-indicator dummies
DUMMY_NA = False  # <-- αυτό είναι που κόβει τα "_nan" columns

# Important for correct labeling with censoring
USE_KNOWN_OUTCOME_FILTER = True  # keep event=1 OR (event=0 & duration>=H)

OUT_DIR = Path("CSV/Exports/Temp/13_ML_outofhospital_mortality_optuna")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Exclusions (exactly what you asked + labels)
# ============================================================
EXCLUDE_FEATURES_BASE = [
    "row_count",
    "subject_id",
    "hadm_id",
    "Time_Zone",
    "los",
    "hospital_expire_flag",
    "time_to_death_days",
]

def build_exclude_features(df: pd.DataFrame) -> list[str]:
    # exclude base + ANY event_*/duration_* columns found (avoid leakage)
    extra = [c for c in df.columns if c.startswith("event_") or c.startswith("duration_")]
    return list(dict.fromkeys(EXCLUDE_FEATURES_BASE + extra))


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
# IO
# ============================================================
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ============================================================
# Build X/y/groups (NO IMPUTATION) + optional known outcome filter
# ============================================================
def make_X_y_groups_out(
    df: pd.DataFrame,
    target_event_col: str,
    duration_col: str | None,
    horizon_days: int,
    group_col: str,
    exclude: list[str],
    require_group: bool = True,
):
    df = df.copy()

    if target_event_col not in df.columns:
        raise KeyError(f"Target event column '{target_event_col}' not found.")

    # label cleaning only
    y_full = pd.to_numeric(df[target_event_col], errors="coerce")
    keep = y_full.notna()
    df = df.loc[keep].copy()
    y = y_full.loc[keep].astype(int)

    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    y  = y.loc[ok].copy()

    # Known outcome filter (prevents censored-before-H from being counted as 0)
    if USE_KNOWN_OUTCOME_FILTER and duration_col and duration_col in df.columns:
        dur = pd.to_numeric(df[duration_col], errors="coerce")
        known = (y == 1) | (dur >= float(horizon_days))
        df = df.loc[known].copy()
        y  = y.loc[known].copy()

    # groups
    if group_col in df.columns:
        groups = df[group_col].values
    else:
        if require_group:
            raise KeyError(f"Group column '{group_col}' not found.")
        groups = np.arange(len(df), dtype=int)

    # features
    X = df.drop(columns=[c for c in exclude if c in df.columns], errors="ignore")

    # bool -> int
    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    for c in bool_cols:
        X[c] = X[c].astype("int8")

    # guard: excluded columns must NOT appear
    leaked = [c for c in exclude if c in X.columns]
    if leaked:
        raise RuntimeError(f"Excluded columns leaked into X: {leaked}")

    return X, y, groups


def one_hot_fit(X_raw: pd.DataFrame) -> pd.DataFrame:
    # NO dummy_na => no explicit missing-indicator dummy columns
    return pd.get_dummies(X_raw, drop_first=False, dummy_na=DUMMY_NA)

def one_hot_transform(X_raw: pd.DataFrame, train_columns: list[str]) -> pd.DataFrame:
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=DUMMY_NA)
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

def split_mimic_dev_test(X, y, groups, test_size: float, random_state: int):
    return _pick_best_sgkf_holdout(X, y, groups, holdout_size=test_size, random_state=random_state)

def split_dev_train_val(X_dev, y_dev, g_dev, val_size: float, random_state: int):
    return _pick_best_sgkf_holdout(X_dev, y_dev, g_dev, holdout_size=val_size, random_state=random_state)

def assert_no_group_overlap(*sets_of_groups):
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
    num_boost_round: int,
    early_stopping_rounds: int,
    chunk_size: int
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
# Metrics + CI
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


# ============================================================
# Save artifacts
# ============================================================
def save_model_and_metadata(booster: xgb.Booster,
                            feature_names: list[str],
                            best_params: dict,
                            out_dir: Path,
                            basename: str,
                            target_event_col: str,
                            duration_col: str | None,
                            horizon_days: int,
                            exclude_features: list[str]):
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
        "group_col": GROUP_COL,
        "horizon_days": int(horizon_days),
        "target_event_col": target_event_col,
        "duration_col": duration_col,
        "excluded_from_training": exclude_features,
        "dummy_na": bool(DUMMY_NA),
        "known_outcome_filter_enabled": bool(USE_KNOWN_OUTCOME_FILTER),
        "model_file": model_path.name,
        "features_file": feat_path.name,
        "num_features": int(len(feature_names)),
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

    print("\n[Load] MIMIC dataset...")
    df_all = load_csv(MIMIC_PATH)

    EXCLUDE_FEATURES = build_exclude_features(df_all)

    summary_rows = []

    for H in HORIZONS:
        TARGET_EVENT_COL = f"event_{H}d"
        DURATION_COL = f"duration_{H}d" if f"duration_{H}d" in df_all.columns else None

        horizon_dir = OUT_DIR / f"mort_{H}d"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        model_basename = f"xgb_outofhospital_mort_{H}d_optuna"

        print("\n" + "="*70)
        print(f"[H={H}] Target={TARGET_EVENT_COL} | Duration={DURATION_COL}")
        print("="*70)

        print("[1/6] Build X/y/groups (NO IMPUTATION, NO masks)...")
        X_m, y_m, g_m = make_X_y_groups_out(
            df=df_all,
            target_event_col=TARGET_EVENT_COL,
            duration_col=DURATION_COL,
            horizon_days=H,
            group_col=GROUP_COL,
            exclude=EXCLUDE_FEATURES,
            require_group=True
        )

        print(f"  Cohort: n={len(y_m)} | pos_rate={y_m.mean():.4f} | unique_subjects={len(set(g_m))}")

        print("[2/6] OUTER split: DEV / TEST ...")
        dev_idx, test_idx = split_mimic_dev_test(X_m, y_m, g_m, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        X_dev_raw = X_m.iloc[dev_idx].copy()
        y_dev     = y_m.iloc[dev_idx].copy()
        g_dev     = g_m[dev_idx]

        X_test_raw = X_m.iloc[test_idx].copy()
        y_test     = y_m.iloc[test_idx].copy()
        g_test     = g_m[test_idx]

        assert_no_group_overlap(g_dev, g_test)

        print(f"  DEV : n={len(y_dev)} | pos_rate={y_dev.mean():.4f} | unique_subjects={len(set(g_dev))}")
        print(f"  TEST: n={len(y_test)} | pos_rate={y_test.mean():.4f} | unique_subjects={len(set(g_test))}")

        print("[3/6] Optuna tuning: 5-fold StratifiedGroupKFold CV on DEV only...")
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

                # One-hot per fold (NO masks: dummy_na=False)
                X_tr = pd.get_dummies(X_tr_raw, drop_first=False, dummy_na=DUMMY_NA)
                X_va = pd.get_dummies(X_va_raw, drop_first=False, dummy_na=DUMMY_NA)
                X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)

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
                aucs.append(float(roc_auc_score(y_va, preds)))

                trial.report(float(np.mean(aucs)), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(aucs))

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )

        with tqdm(total=N_TRIALS, desc=f"Optuna trials (H={H})") as pbar:
            def _cb(_study, _trial):
                pbar.update(1)
            study.optimize(objective, n_trials=N_TRIALS, timeout=OPTUNA_TIMEOUT, callbacks=[_cb], gc_after_trial=True)

        try:
            study.trials_dataframe().to_csv(horizon_dir / "optuna_trials.csv", index=False)
        except Exception:
            pass

        best_params = study.best_trial.params
        print("  Best DEV-CV AUC:", float(study.best_value))

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

        print("[4/6] Final training inside DEV (DEV-TRAIN / DEV-VAL)...")
        dev_tr_idx, dev_va_idx = split_dev_train_val(X_dev_raw, y_dev, g_dev, val_size=DEV_VAL_SIZE, random_state=RANDOM_STATE)

        X_dev_tr_raw = X_dev_raw.iloc[dev_tr_idx].copy()
        y_dev_tr = y_dev.iloc[dev_tr_idx].copy()
        g_dev_tr = g_dev[dev_tr_idx]

        X_dev_va_raw = X_dev_raw.iloc[dev_va_idx].copy()
        y_dev_va = y_dev.iloc[dev_va_idx].copy()
        g_dev_va = g_dev[dev_va_idx]

        assert_no_group_overlap(g_dev_tr, g_dev_va, g_test)

        X_train = one_hot_fit(X_dev_tr_raw)
        train_cols = list(X_train.columns)

        X_val  = one_hot_transform(X_dev_va_raw, train_cols)
        X_test = one_hot_transform(X_test_raw,   train_cols)

        assert_numeric_nans_preserved(X_dev_tr_raw, X_train, where="final DEV-TRAIN one-hot")
        assert_numeric_nans_preserved(X_dev_va_raw, X_val,   where="final DEV-VAL one-hot")
        assert_numeric_nans_preserved(X_test_raw,   X_test,  where="final TEST one-hot")

        booster, best_iter, best_val_auc = train_booster_manual_es(
            params=final_params,
            X_train=X_train, y_train=y_dev_tr.values,
            X_val=X_val,     y_val=y_dev_va.values,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            chunk_size=CHUNK_SIZE
        )

        print(f"  Final model: best_iter={best_iter}, best_dev_val_auc={best_val_auc:.4f}")

        print("[5/6] Save model + metadata...")
        save_model_and_metadata(
            booster=booster,
            feature_names=train_cols,
            best_params={"optuna_best_dev_cv_auc": float(study.best_value), **best_params},
            out_dir=horizon_dir,
            basename=model_basename,
            target_event_col=TARGET_EVENT_COL,
            duration_col=DURATION_COL,
            horizon_days=H,
            exclude_features=EXCLUDE_FEATURES
        )

        print("[6/6] Evaluation ONLY on MIMIC TEST...")
        p_test = predict_proba(booster, X_test)

        def _auc(y, p):   return roc_auc_score(y, p)
        def _ap(y, p):    return average_precision_score(y, p)
        def _brier(y, p): return brier_score_loss(y, p)

        auc, auc_lo, auc_hi = bootstrap_ci(y_test.values, p_test, _auc,   desc=f"H={H} | TEST AUC boot")
        ap,  ap_lo,  ap_hi  = bootstrap_ci(y_test.values, p_test, _ap,    desc=f"H={H} | TEST PR-AUC boot")
        br,  br_lo,  br_hi  = bootstrap_ci(y_test.values, p_test, _brier, desc=f"H={H} | TEST Brier boot")

        row = {
            "horizon_days": int(H),
            "dataset": "MIMIC_test",
            "n": int(len(y_test)),
            "pos_rate": float(np.mean(y_test.values)),
            "AUC": auc, "AUC_CI_low": auc_lo, "AUC_CI_high": auc_hi,
            "PR_AUC": ap, "PR_AUC_CI_low": ap_lo, "PR_AUC_CI_high": ap_hi,
            "Brier": br, "Brier_CI_low": br_lo, "Brier_CI_high": br_hi,
        }
        summary_rows.append(row)

        pd.DataFrame([row]).to_csv(horizon_dir / f"metrics_mort_{H}d.csv", index=False)

        plot_roc(y_test.values, p_test, f"ROC (MIMIC TEST) | Mortality {H}d", horizon_dir / "roc_mimic_test", "MIMIC TEST")
        plot_pr (y_test.values, p_test, f"PR (MIMIC TEST) | Mortality {H}d",  horizon_dir / "pr_mimic_test",  "MIMIC TEST")

        if RUN_SHAP:
            try:
                run_shap_and_save(booster, X_train, horizon_dir)
            except Exception as e:
                print("  SHAP failed:", repr(e))

        print("  Done for H=", H, "| outputs:", horizon_dir)

    summary_df = pd.DataFrame(summary_rows).sort_values("horizon_days")
    summary_df.to_csv(OUT_DIR / "metrics_summary_outofhospital.csv", index=False)
    print("\nSaved overall summary:", OUT_DIR / "metrics_summary_outofhospital.csv")
    print(summary_df.to_string(index=False))
    print("\nDone. Outputs in:", OUT_DIR)
