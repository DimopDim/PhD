# decision_curve_outofhospital.py
# ============================================================
# Decision Curve Analysis (DCA) for saved OUT-OF-HOSPITAL models
# - For H in {30,180,360} days:
#   * rebuild cohort with known-outcome filter (same as training)
#   * recreate DEV/TEST holdout split (SGKF pick-best)
#   * load saved model + saved one-hot feature list from OUT_DIR/mort_{H}d/
#   * predict on MIMIC TEST only
#   * compute Net Benefit curves + save PNG/PDF + CSV
# - Also saves a combined plot with the 3 model curves.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import xgboost as xgb


# -------------------------
# CONFIG (match training)
# -------------------------
MIMIC_PATH = "CSV/o01_mimic_out_of_hospital.csv"

RANDOM_STATE = 42
GROUP_COL = "subject_id"

HORIZONS = [30, 180, 360]
TEST_SIZE = 0.10

# IMPORTANT: training used dummy_na=False
DUMMY_NA = False

# training used this to avoid counting censored-before-H as negatives
USE_KNOWN_OUTCOME_FILTER = True

OUT_DIR = Path("CSV/Exports/Temp/13_ML_outofhospital_mortality_optuna")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DCA thresholds (edit if you want)
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
# Exclusions (same idea as training)
# -------------------------
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
    extra = [c for c in df.columns if c.startswith("event_") or c.startswith("duration_")]
    # unique preserve order
    return list(dict.fromkeys(EXCLUDE_FEATURES_BASE + extra))


# -------------------------
# Dataset -> X/y/groups (NO IMPUTATION)
# -------------------------
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

    # Known outcome filter: keep event=1 OR (event=0 & duration>=H)
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


def one_hot_transform_to_saved_features(X_raw: pd.DataFrame, saved_columns: list[str]) -> pd.DataFrame:
    X_oh = pd.get_dummies(X_raw, drop_first=False, dummy_na=DUMMY_NA)
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
    Net Benefit:
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

        rows.append({
            "threshold": float(pt),
            "net_benefit_model": float(nb_model),
            "net_benefit_treat_all": float(nb_all),
            "net_benefit_treat_none": 0.0,
            "prevalence": float(prev),
            "n": int(n),
            "tp": tp,
            "fp": fp,
        })

    return pd.DataFrame(rows)

def plot_dca(df_curve: pd.DataFrame, title: str, out_base: Path):
    plt.figure()
    plt.plot(df_curve["threshold"], df_curve["net_benefit_model"], label="Model")
    plt.plot(df_curve["threshold"], df_curve["net_benefit_treat_all"], label="Treat all")
    plt.plot(df_curve["threshold"], df_curve["net_benefit_treat_none"], label="Treat none")
    plt.axhline(0.0, linestyle="--", linewidth=1.3)

    prev = float(df_curve["prevalence"].iloc[0]) if len(df_curve) else float("nan")
    n    = int(df_curve["n"].iloc[0]) if len(df_curve) else 0

    plt.xlabel("Threshold probability")
    plt.ylabel("Net Benefit")
    plt.title(f"{title}  (n={n}, event rate={prev:.3f})")
    plt.legend(frameon=False)
    save_fig_png_pdf(out_base)

def plot_dca_models_across_horizons(curves_by_h: dict[int, pd.DataFrame], out_base: Path):
    plt.figure()
    for H, dfc in curves_by_h.items():
        prev = float(dfc["prevalence"].iloc[0])
        plt.plot(dfc["threshold"], dfc["net_benefit_model"], label=f"Model {H}d (event={prev:.3f})")

    # Treat-none (common)
    any_df = next(iter(curves_by_h.values()))
    plt.plot(any_df["threshold"], any_df["net_benefit_treat_none"], linestyle="-.", linewidth=1.8, label="Treat none")

    plt.axhline(0.0, linestyle="--", linewidth=1.3)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve (Model) across horizons")
    plt.legend(frameon=False)
    save_fig_png_pdf(out_base)


# -------------------------
# MAIN
# -------------------------
def main():
    df_all = pd.read_csv(MIMIC_PATH)
    exclude_features = build_exclude_features(df_all)

    thresholds = np.arange(THRESH_START, THRESH_STOP + 1e-12, THRESH_STEP)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]

    curves_by_h = {}
    summary_rows = []

    for H in HORIZONS:
        target_event_col = f"event_{H}d"
        duration_col = f"duration_{H}d" if f"duration_{H}d" in df_all.columns else None

        horizon_dir = OUT_DIR / f"mort_{H}d"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        model_basename = f"xgb_outofhospital_mort_{H}d_optuna"
        model_path = horizon_dir / f"{model_basename}.json"
        feats_path = horizon_dir / f"{model_basename}_features.txt"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model for H={H}: {model_path}")
        if not feats_path.exists():
            raise FileNotFoundError(f"Missing feature list for H={H}: {feats_path}")

        print("\n" + "="*80)
        print(f"H={H}d | target={target_event_col} | duration={duration_col}")
        print("="*80)

        # Build cohort for this horizon (includes known outcome filter)
        X_raw, y, groups = make_X_y_groups_out(
            df=df_all,
            target_event_col=target_event_col,
            duration_col=duration_col,
            horizon_days=H,
            group_col=GROUP_COL,
            exclude=exclude_features,
            require_group=True
        )

        print(f"Cohort: n={len(y)} | pos_rate={float(np.mean(y)):.4f} | unique_subjects={len(set(groups))}")

        # Recreate same DEV/TEST split for THIS horizon cohort
        dev_idx, test_idx = _pick_best_sgkf_holdout(
            X_raw, y, groups,
            holdout_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        X_test_raw = X_raw.iloc[test_idx].copy()
        y_test = y.iloc[test_idx].values

        print(f"TEST: n={len(y_test)} | pos_rate={float(np.mean(y_test)):.4f}")

        # Load model + saved feature space
        saved_cols = load_saved_feature_list(feats_path)
        booster = load_booster(model_path)

        # One-hot (dummy_na=False) and align to saved cols
        X_test = one_hot_transform_to_saved_features(X_test_raw, saved_cols)

        # Predict
        p_test = predict_proba(booster, X_test)

        # DCA
        dca_df = dca_net_benefit_curve(y_test, p_test, thresholds)
        curves_by_h[H] = dca_df

        # Save CSV + plot
        dca_df.to_csv(horizon_dir / f"decision_curve_mimic_test_mort_{H}d.csv", index=False)
        plot_dca(dca_df, f"Decision Curve (MIMIC TEST) | Mortality {H}d", horizon_dir / f"dca_mimic_test_mort_{H}d")

        # small summary (optional)
        summary_rows.append({
            "horizon_days": int(H),
            "n_test": int(len(y_test)),
            "pos_rate_test": float(np.mean(y_test)),
        })

    # Combined plot (model curves only)
    plot_dca_models_across_horizons(curves_by_h, OUT_DIR / "dca_models_across_horizons")

    pd.DataFrame(summary_rows).sort_values("horizon_days").to_csv(OUT_DIR / "dca_test_summary.csv", index=False)

    print("\nSaved combined plot:", OUT_DIR / "dca_models_across_horizons.(png/pdf)")
    print("Saved summary:", OUT_DIR / "dca_test_summary.csv")
    print("\nDone. Outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
