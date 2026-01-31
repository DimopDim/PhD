# compare_mimic_vs_eicu_table1_patientlevel.py
# ============================================================
# Build Table 1 (patient-level, not per-Time_Zone row):
#   (A) Overall baseline: MIMIC vs eICU + p-values + effect sizes
#   (B) Survivors vs non-survivors within each DB (side-by-side) + p-values
#
# KEY CHANGE:
#   - Reduce to 1 row per patient using:
#       --time_zone 16   (default)  -> keep Time_Zone == 16 (48h snapshot)
#     OR if Time_Zone missing:
#       -> drop duplicates per patient id (subject_id / patientunitstayid)
#
# UPDATE:
#   - "Smart" formatting (no trailing zeros): 3.000 -> 3, 20.0% -> 20%
#   - p-values displayed as:  p < 0.001  (otherwise trimmed decimals)
#   - Keeps raw numeric p-values in *_raw columns (so BH-FDR remains correct)
# ============================================================

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    from scipy import stats
except Exception as e:
    raise RuntimeError("scipy is required for p-values. Please install scipy.") from e


# -------------------------
# Defaults (match your project)
# -------------------------
DEFAULT_MIMIC_PATH = "CSV/o01_mimic_for_ext_val.csv"
DEFAULT_EICU_PATH  = "CSV/o01_eicu_for_ext_val.csv"

TARGET_COL = "hospital_expire_flag"  # 0=survive, 1=die
GROUP_COL  = "subject_id"            # MIMIC patient id
EICU_GROUP_FALLBACK = "patientunitstayid"

OUT_DIR_DEFAULT = "CSV/Exports/Temp/13_Table1_MIMIC_vs_eICU_patientlevel"


# -------------------------
# IO helpers
# -------------------------
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def maybe_write_xlsx(df: pd.DataFrame, xlsx_path: Path, sheet_name: str = "Table1") -> bool:
    try:
        import openpyxl  # noqa: F401
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
        return True
    except Exception:
        return False


# -------------------------
# Smart formatting helpers (UPDATE)
# -------------------------
def fmt_num(x, nd: int = 3, na: str = "NA") -> str:
    """
    Round to nd decimals AND remove trailing zeros.
    Examples:
      3.000 -> '3'
      3.100 -> '3.1'
      3.125 -> '3.125'
    """
    try:
        v = float(x)
    except Exception:
        return na
    if not np.isfinite(v):
        return na
    s = f"{v:.{nd}f}"
    s = s.rstrip("0").rstrip(".")
    # keep '-0' from appearing
    if s == "-0":
        s = "0"
    return s


def fmt_p(p, nd: int = 3, threshold: float = 0.001, na: str = "") -> str:
    """
    p-value formatting:
      p < 0.001 -> '<0.001'
      else -> trimmed decimals (e.g., 0.050 -> '0.05')
    """
    try:
        v = float(p)
    except Exception:
        return na
    if not np.isfinite(v):
        return na
    if v < threshold:
        return f"<{threshold:g}"
    return fmt_num(v, nd=nd, na=na)


def fmt_count_pct(count: int, total: int, nd: int = 1) -> str:
    """
    '12 (20%)' instead of '12 (20.0%)' when pct ends in .0
    """
    total = max(int(total), 1)
    pct = 100.0 * (int(count) / total)
    return f"{int(count)} ({fmt_num(pct, nd=nd, na='0')}%)"


# -------------------------
# Label helpers
# -------------------------
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

    raise KeyError(f"Could not find '{target_col}' nor a fallback label column.")


def clean_label_rows(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    df = df.copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int)
    ok = y.isin([0, 1])
    df = df.loc[ok].copy()
    df[target_col] = y.loc[ok].values
    return df


# -------------------------
# Patient-level reducer
# -------------------------
def _infer_patient_id_col(df: pd.DataFrame) -> str | None:
    if GROUP_COL in df.columns:
        return GROUP_COL
    if EICU_GROUP_FALLBACK in df.columns:
        return EICU_GROUP_FALLBACK
    for c in ["subject_id", "patientunitstayid", "stay_id", "hadm_id"]:
        if c in df.columns:
            return c
    return None


def reduce_to_patient_level(df: pd.DataFrame, time_zone: int | None, prefer_last: bool = True) -> pd.DataFrame:
    df = df.copy()

    pid = _infer_patient_id_col(df)
    if pid is None:
        return df

    if (time_zone is not None) and ("Time_Zone" in df.columns):
        df = df.loc[pd.to_numeric(df["Time_Zone"], errors="coerce") == int(time_zone)].copy()

    keep = "last" if prefer_last else "first"
    df = df.sort_index().drop_duplicates(subset=[pid], keep=keep).copy()
    return df


# -------------------------
# Split helpers (MIMIC subset)
# -------------------------
def _pick_best_sgkf_holdout(groups, y, holdout_size: float, random_state: int = 42, show_progress: bool = True):
    n = len(y)
    X_dummy = np.zeros((n, 1), dtype=float)

    n_splits = max(int(round(1.0 / holdout_size)), 2)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    y_arr = np.asarray(y)
    target_n = int(round(n * holdout_size))
    target_pos = float(np.mean(y_arr))

    best = None
    best_score = float("inf")

    iterator = list(sgkf.split(X_dummy, y_arr, groups=groups))
    if show_progress:
        iterator = tqdm(iterator, desc="Picking best SGKF holdout", unit="fold", leave=False)

    for tr_idx, ho_idx in iterator:
        n_ho = len(ho_idx)
        pos_ho = float(np.mean(y_arr[ho_idx])) if n_ho else 0.0

        size_term = abs(n_ho - target_n) / max(target_n, 1)
        pos_term  = abs(pos_ho - target_pos) / max(target_pos, 1e-6)
        score = size_term + pos_term

        if score < best_score:
            best_score = score
            best = (tr_idx, ho_idx)

    return np.asarray(best[0]), np.asarray(best[1])


def subset_mimic(df_mimic: pd.DataFrame, subset: str, test_size: float, random_state: int, show_progress: bool) -> pd.DataFrame:
    if subset == "full":
        return df_mimic

    if GROUP_COL not in df_mimic.columns:
        raise KeyError(f"MIMIC must have '{GROUP_COL}' to create grouped DEV/TEST split.")

    y = df_mimic[TARGET_COL].astype(int).values
    g = df_mimic[GROUP_COL].values

    dev_idx, test_idx = _pick_best_sgkf_holdout(
        g, y, holdout_size=test_size, random_state=random_state, show_progress=show_progress
    )

    if subset == "test":
        return df_mimic.iloc[test_idx].copy()
    if subset == "dev":
        return df_mimic.iloc[dev_idx].copy()

    raise ValueError("subset must be one of: full, dev, test")


# -------------------------
# Stats helpers
# -------------------------
def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def is_binary_01(s: pd.Series) -> bool:
    x = safe_numeric(s).dropna()
    if len(x) == 0:
        return False
    vals = set(np.unique(x.values).tolist())
    return vals.issubset({0, 1})


def median_iqr(x: np.ndarray, nd: int = 3) -> str:
    if len(x) == 0:
        return "NA"
    q1 = np.quantile(x, 0.25)
    med = np.quantile(x, 0.50)
    q3 = np.quantile(x, 0.75)
    # UPDATE: trim trailing zeros
    return f"{fmt_num(med, nd=nd)} [{fmt_num(q1, nd=nd)}, {fmt_num(q3, nd=nd)}]"


def mean_std(x: np.ndarray) -> Tuple[float, float]:
    if len(x) == 0:
        return (np.nan, np.nan)
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)


def smd_continuous(x1: np.ndarray, x2: np.ndarray) -> float:
    m1, s1 = mean_std(x1)
    m2, s2 = mean_std(x2)
    if not np.isfinite(m1) or not np.isfinite(m2):
        return np.nan
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan
    sp = np.sqrt(((n1 - 1) * (s1**2) + (n2 - 1) * (s2**2)) / max(n1 + n2 - 2, 1))
    if sp == 0:
        return 0.0
    return float((m1 - m2) / sp)


def smd_binary(p1: float, p2: float) -> float:
    p = (p1 + p2) / 2.0
    denom = np.sqrt(max(p * (1 - p), 1e-12))
    return float((p1 - p2) / denom)


def cramers_v(chi2: float, n: int, r: int, k: int) -> float:
    if n <= 0:
        return np.nan
    denom = n * (min(r - 1, k - 1))
    if denom <= 0:
        return np.nan
    return float(np.sqrt(max(chi2 / denom, 0.0)))


def chi2_or_fisher(table: np.ndarray):
    table = np.asarray(table, dtype=int)
    if table.shape == (2, 2):
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        if (exp < 5).any():
            _oddsratio, p_f = stats.fisher_exact(table, alternative="two-sided")
            return "fisher_exact", float(p_f), np.nan
        return "chi2", float(p), float(chi2)

    chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
    return "chi2", float(p), float(chi2)


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out_ok = np.empty_like(pv)
    out_ok[order] = adj
    out[ok] = out_ok
    return out


# -------------------------
# Column selection (handles _(Median)/(Mean)/(Min)/(Max))
# -------------------------
SUMMARY_SUFFIXES = ["_(Median)", "_(Mean)", "_(Min)", "_(Max)"]
SUMMARY_PREF_FALLBACK = ["_(Median)", "_(Mean)", "_(Min)", "_(Max)"]

def _is_summary_col(c: str) -> bool:
    return any(c.endswith(suf) for suf in SUMMARY_SUFFIXES)

def _base_name(c: str) -> str:
    for suf in SUMMARY_SUFFIXES:
        if c.endswith(suf):
            return c[: -len(suf)]
    return c

def build_selected_columns(
    common_cols: List[str],
    summary: str = "Median",
    keep_all_summaries: bool = False,
    include_demographics: bool = True,
) -> List[str]:
    summary = summary.strip().lower()
    want = {
        "median": "_(Median)",
        "mean": "_(Mean)",
        "min": "_(Min)",
        "max": "_(Max)",
    }.get(summary, "_(Median)")

    exclude_like = {
        "row_count", "hadm_id", "stay_id", "icu_intime", "icu_outtime", "hosp_dischtime", "dod",
        "subject_id", "Time_Zone",
        TARGET_COL,
    }

    demographics = ["gender", "sex", "age", "race", "ethnicity", "BMI", "bmi", "sofa", "SOFA", "los", "LOS"]

    common_cols = [c for c in common_cols if c not in exclude_like]
    unsuffixed = [c for c in common_cols if not _is_summary_col(c)]

    if include_demographics:
        demo_keep = []
        demo_set = set(demographics)
        for c in unsuffixed:
            if c in demo_set:
                demo_keep.append(c)
        base_keep = [c for c in unsuffixed if c not in set(demo_keep)]
        unsuffixed_keep = demo_keep + base_keep
    else:
        unsuffixed_keep = unsuffixed

    summary_cols = [c for c in common_cols if _is_summary_col(c)]
    if keep_all_summaries:
        chosen_summary_cols = summary_cols
    else:
        by_base: Dict[str, List[str]] = {}
        for c in summary_cols:
            by_base.setdefault(_base_name(c), []).append(c)

        chosen_summary_cols = []
        for base, cols in by_base.items():
            cols_set = set(cols)
            if base + want in cols_set:
                chosen_summary_cols.append(base + want)
            else:
                picked = None
                for suf in SUMMARY_PREF_FALLBACK:
                    if base + suf in cols_set:
                        picked = base + suf
                        break
                if picked is not None:
                    chosen_summary_cols.append(picked)

        chosen_summary_cols = sorted(chosen_summary_cols)

    seen = set()
    out = []
    for c in unsuffixed_keep + chosen_summary_cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# -------------------------
# Table builders
# -------------------------
def _event_rate(df: pd.DataFrame) -> float:
    y = safe_numeric(df[TARGET_COL]).dropna()
    if len(y) == 0:
        return np.nan
    return float(np.mean(y.astype(int).values))


def build_table_overall(df_a, df_b, cols, name_a, name_b, show_progress=True) -> pd.DataFrame:
    rows = []
    n_a = len(df_a)
    n_b = len(df_b)
    er_a = _event_rate(df_a)
    er_b = _event_rate(df_b)

    rows.append({
        "variable": "N",
        "level": "",
        name_a: f"{n_a}",
        name_b: f"{n_b}",
        "test": "",
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_type": "",
        f"missing_%_{name_a}": 0.0,
        f"missing_%_{name_b}": 0.0,
    })
    rows.append({
        "variable": "In-hospital mortality (event rate)",
        "level": "",
        # UPDATE: trimmed formatting
        name_a: fmt_num(er_a, nd=4, na="NA"),
        name_b: fmt_num(er_b, nd=4, na="NA"),
        "test": "",
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_type": "",
        f"missing_%_{name_a}": float(df_a[TARGET_COL].isna().mean() * 100),
        f"missing_%_{name_b}": float(df_b[TARGET_COL].isna().mean() * 100),
    })

    col_iter = tqdm(cols, desc="Table A: overall MIMIC vs eICU", unit="var") if show_progress else cols

    for col in col_iter:
        if col not in df_a.columns or col not in df_b.columns:
            continue

        s1 = df_a[col]
        s2 = df_b[col]
        miss1 = float(s1.isna().mean() * 100)
        miss2 = float(s2.isna().mean() * 100)

        # categorical strings
        if s1.dtype == "object" or s2.dtype == "object":
            x1 = s1.astype("string").fillna("MISSING")
            x2 = s2.astype("string").fillna("MISSING")
            levels = sorted(set(x1.unique().tolist() + x2.unique().tolist()))

            tab1 = pd.Series(x1).value_counts().reindex(levels, fill_value=0).values
            tab2 = pd.Series(x2).value_counts().reindex(levels, fill_value=0).values
            cont = np.vstack([tab1, tab2])

            test, p, chi2 = chi2_or_fisher(cont)
            v = cramers_v(chi2 if np.isfinite(chi2) else np.nan, n=int(cont.sum()), r=2, k=cont.shape[1])

            total1 = int(cont[0].sum())
            total2 = int(cont[1].sum())

            first = True
            for lvl_i, lvl in enumerate(levels):
                c1 = int(cont[0, lvl_i]); c2 = int(cont[1, lvl_i])
                rows.append({
                    "variable": col,
                    "level": str(lvl),
                    # UPDATE: percent without trailing .0
                    name_a: fmt_count_pct(c1, total1, nd=1),
                    name_b: fmt_count_pct(c2, total2, nd=1),
                    "test": test if first else "",
                    "p_value": float(p) if first else np.nan,
                    "effect_size": float(v) if first else np.nan,
                    "effect_type": "Cramér's V" if first else "",
                    f"missing_%_{name_a}": miss1 if first else np.nan,
                    f"missing_%_{name_b}": miss2 if first else np.nan,
                })
                first = False
            continue

        # binary
        if is_binary_01(s1) and is_binary_01(s2):
            x1n = safe_numeric(s1).dropna().astype(int).values
            x2n = safe_numeric(s2).dropna().astype(int).values
            a1 = int(np.sum(x1n == 1)); a0 = int(np.sum(x1n == 0))
            b1 = int(np.sum(x2n == 1)); b0 = int(np.sum(x2n == 0))

            test, p, _chi2 = chi2_or_fisher(np.array([[a1, a0], [b1, b0]]))
            p1 = a1 / max(a1 + a0, 1)
            p2 = b1 / max(b1 + b0, 1)
            eff = smd_binary(p1, p2)

            rows.append({
                "variable": col,
                "level": "",
                name_a: fmt_count_pct(a1, a1 + a0, nd=1),
                name_b: fmt_count_pct(b1, b1 + b0, nd=1),
                "test": test,
                "p_value": float(p),
                "effect_size": float(eff),
                "effect_type": "SMD (binary)",
                f"missing_%_{name_a}": miss1,
                f"missing_%_{name_b}": miss2,
            })
            continue

        # continuous
        x1 = safe_numeric(s1).dropna().values.astype(float)
        x2 = safe_numeric(s2).dropna().values.astype(float)

        try:
            _u_stat, p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
            p = float(p)
        except Exception:
            p = np.nan

        eff = smd_continuous(x1, x2)

        rows.append({
            "variable": col,
            "level": "",
            name_a: median_iqr(x1, nd=3),
            name_b: median_iqr(x2, nd=3),
            "test": "mannwhitneyu",
            "p_value": p,
            "effect_size": float(eff) if np.isfinite(eff) else np.nan,
            "effect_type": "SMD (continuous)",
            f"missing_%_{name_a}": miss1,
            f"missing_%_{name_b}": miss2,
        })

    out = pd.DataFrame(rows)

    # BH-FDR on raw numeric p-values
    out["p_value_fdr_bh"] = bh_fdr(out["p_value"].values)

    # UPDATE: convert to "report" strings without trailing zeros + p<0.001
    out["p_value_raw"] = out["p_value"]
    out["p_value_fdr_bh_raw"] = out["p_value_fdr_bh"]
    out["p_value"] = out["p_value_raw"].apply(lambda v: fmt_p(v, nd=3, threshold=0.001, na=""))
    out["p_value_fdr_bh"] = out["p_value_fdr_bh_raw"].apply(lambda v: fmt_p(v, nd=3, threshold=0.001, na=""))

    # OPTIONAL: effect size also trimmed for readability
    out["effect_size_raw"] = out["effect_size"]
    out["effect_size"] = out["effect_size_raw"].apply(lambda v: fmt_num(v, nd=3, na=""))

    return out


def build_table_within_outcome(df, cols, dataset_name, show_progress=True) -> pd.DataFrame:
    df0 = df[df[TARGET_COL] == 0].copy()
    df1 = df[df[TARGET_COL] == 1].copy()

    rows = []
    rows.append({
        "variable": "N",
        "level": "",
        f"{dataset_name}_survive": f"{len(df0)}",
        f"{dataset_name}_non_survive": f"{len(df1)}",
        f"{dataset_name}_test": "",
        f"{dataset_name}_p_value": np.nan,
        f"{dataset_name}_effect_size": np.nan,
        f"{dataset_name}_effect_type": "",
        f"{dataset_name}_missing_%": 0.0,
    })

    col_iter = tqdm(cols, desc=f"Table B: surv vs non-surv ({dataset_name})", unit="var") if show_progress else cols

    for col in col_iter:
        if col not in df.columns:
            continue

        s_all = df[col]
        miss = float(s_all.isna().mean() * 100)

        s0 = df0[col]
        s1 = df1[col]

        # categorical strings
        if s0.dtype == "object" or s1.dtype == "object":
            x0 = s0.astype("string").fillna("MISSING")
            x1 = s1.astype("string").fillna("MISSING")
            levels = sorted(set(x0.unique().tolist() + x1.unique().tolist()))

            tab0 = pd.Series(x0).value_counts().reindex(levels, fill_value=0).values
            tab1 = pd.Series(x1).value_counts().reindex(levels, fill_value=0).values
            cont = np.vstack([tab0, tab1])

            test, p, chi2 = chi2_or_fisher(cont)
            v = cramers_v(chi2 if np.isfinite(chi2) else np.nan, n=int(cont.sum()), r=2, k=cont.shape[1])

            total0 = int(cont[0].sum())
            total1 = int(cont[1].sum())

            first = True
            for lvl_i, lvl in enumerate(levels):
                c0 = int(cont[0, lvl_i]); c1 = int(cont[1, lvl_i])
                rows.append({
                    "variable": col,
                    "level": str(lvl),
                    f"{dataset_name}_survive": fmt_count_pct(c0, total0, nd=1),
                    f"{dataset_name}_non_survive": fmt_count_pct(c1, total1, nd=1),
                    f"{dataset_name}_test": test if first else "",
                    f"{dataset_name}_p_value": float(p) if first else np.nan,
                    f"{dataset_name}_effect_size": float(v) if first else np.nan,
                    f"{dataset_name}_effect_type": "Cramér's V" if first else "",
                    f"{dataset_name}_missing_%": miss if first else np.nan,
                })
                first = False
            continue

        # binary
        if is_binary_01(s0) and is_binary_01(s1):
            x0n = safe_numeric(s0).dropna().astype(int).values
            x1n = safe_numeric(s1).dropna().astype(int).values
            a1 = int(np.sum(x0n == 1)); a0 = int(np.sum(x0n == 0))
            b1 = int(np.sum(x1n == 1)); b0 = int(np.sum(x1n == 0))

            test, p, _chi2 = chi2_or_fisher(np.array([[a1, a0], [b1, b0]]))
            p0 = a1 / max(a1 + a0, 1)
            p1 = b1 / max(b1 + b0, 1)
            eff = smd_binary(p0, p1)

            rows.append({
                "variable": col,
                "level": "",
                f"{dataset_name}_survive": fmt_count_pct(a1, a1 + a0, nd=1),
                f"{dataset_name}_non_survive": fmt_count_pct(b1, b1 + b0, nd=1),
                f"{dataset_name}_test": test,
                f"{dataset_name}_p_value": float(p),
                f"{dataset_name}_effect_size": float(eff),
                f"{dataset_name}_effect_type": "SMD (binary)",
                f"{dataset_name}_missing_%": miss,
            })
            continue

        # continuous
        x0 = safe_numeric(s0).dropna().values.astype(float)
        x1 = safe_numeric(s1).dropna().values.astype(float)

        try:
            _u_stat, p = stats.mannwhitneyu(x0, x1, alternative="two-sided")
            p = float(p)
        except Exception:
            p = np.nan

        eff = smd_continuous(x0, x1)

        rows.append({
            "variable": col,
            "level": "",
            f"{dataset_name}_survive": median_iqr(x0, nd=3),
            f"{dataset_name}_non_survive": median_iqr(x1, nd=3),
            f"{dataset_name}_test": "mannwhitneyu",
            f"{dataset_name}_p_value": p,
            f"{dataset_name}_effect_size": float(eff) if np.isfinite(eff) else np.nan,
            f"{dataset_name}_effect_type": "SMD (continuous)",
            f"{dataset_name}_missing_%": miss,
        })

    out = pd.DataFrame(rows)

    pv_col = f"{dataset_name}_p_value"
    fdr_col = f"{dataset_name}_p_value_fdr_bh"
    out[fdr_col] = bh_fdr(out[pv_col].values)

    # UPDATE: report strings
    out[pv_col + "_raw"] = out[pv_col]
    out[fdr_col + "_raw"] = out[fdr_col]
    out[pv_col] = out[pv_col + "_raw"].apply(lambda v: fmt_p(v, nd=3, threshold=0.001, na=""))
    out[fdr_col] = out[fdr_col + "_raw"].apply(lambda v: fmt_p(v, nd=3, threshold=0.001, na=""))

    eff_col = f"{dataset_name}_effect_size"
    out[eff_col + "_raw"] = out[eff_col]
    out[eff_col] = out[eff_col + "_raw"].apply(lambda v: fmt_num(v, nd=3, na=""))

    return out


def combine_two_within_tables(tbl_mimic: pd.DataFrame, tbl_eicu: pd.DataFrame) -> pd.DataFrame:
    key = ["variable", "level"]
    out = pd.merge(tbl_mimic, tbl_eicu, on=key, how="outer", sort=False)
    mimic_keys = tbl_mimic[key].drop_duplicates()
    out = pd.merge(mimic_keys, out, on=key, how="right")
    return out


# -------------------------
# Saving
# -------------------------
def save_csv_and_optional_xlsx(df: pd.DataFrame, out_dir: Path, base: str, no_xlsx: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{base}.csv"
    df.to_csv(csv_path, index=False)
    print("Saved CSV:", csv_path)

    if no_xlsx:
        return

    xlsx_path = out_dir / f"{base}.xlsx"
    ok = maybe_write_xlsx(df, xlsx_path, sheet_name="Table1")
    if ok:
        print("Saved XLSX:", xlsx_path)
    else:
        print("Skipped XLSX (openpyxl not available). CSV is saved.")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_path", default=DEFAULT_MIMIC_PATH)
    ap.add_argument("--eicu_path", default=DEFAULT_EICU_PATH)
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)

    ap.add_argument("--mimic_subset", choices=["full", "dev", "test"], default="full",
                    help="Use full MIMIC or grouped DEV/TEST subset like your model pipeline.")
    ap.add_argument("--test_size", type=float, default=0.10)
    ap.add_argument("--random_state", type=int, default=42)

    ap.add_argument("--time_zone", type=int, default=16,
                    help="Reduce to patient-level by keeping only this Time_Zone (default=16 -> 48h snapshot). "
                         "If Time_Zone is missing, we deduplicate per patient id.")
    ap.add_argument("--prefer_first", action="store_true",
                    help="When dropping duplicate patients, keep first row instead of last.")

    ap.add_argument("--summary", choices=["Median", "Mean", "Min", "Max"], default="Median")
    ap.add_argument("--keep_all_summaries", action="store_true")
    ap.add_argument("--no_tqdm", action="store_true")
    ap.add_argument("--no_xlsx", action="store_true")
    ap.add_argument("--skip_overall", action="store_true")
    ap.add_argument("--skip_outcome", action="store_true")
    args = ap.parse_args()

    show_progress = not args.no_tqdm
    out_dir = Path(args.out_dir)

    print("[1] Load datasets...")
    mimic = ensure_binary_label(load_csv(args.mimic_path), TARGET_COL)
    eicu  = ensure_binary_label(load_csv(args.eicu_path),  TARGET_COL)

    print("[2] Keep rows with valid label 0/1...")
    mimic = clean_label_rows(mimic, TARGET_COL)
    eicu  = clean_label_rows(eicu,  TARGET_COL)

    print("[3] Reduce to patient-level...")
    mimic = reduce_to_patient_level(mimic, time_zone=args.time_zone, prefer_last=not args.prefer_first)
    eicu  = reduce_to_patient_level(eicu,  time_zone=args.time_zone, prefer_last=not args.prefer_first)

    print("[4] Subset MIMIC if requested:", args.mimic_subset)
    mimic_sub = subset_mimic(
        mimic,
        subset=args.mimic_subset,
        test_size=args.test_size,
        random_state=args.random_state,
        show_progress=show_progress
    )

    print(f"   MIMIC({args.mimic_subset}): n={len(mimic_sub)} | event_rate={mimic_sub[TARGET_COL].mean():.4f}")
    print(f"   eICU:            n={len(eicu)}      | event_rate={eicu[TARGET_COL].mean():.4f}")

    common_cols = sorted(set(mimic_sub.columns).intersection(set(eicu.columns)))
    selected_cols = build_selected_columns(
        common_cols,
        summary=args.summary,
        keep_all_summaries=args.keep_all_summaries,
        include_demographics=True,
    )

    if "los" in common_cols and "los" not in selected_cols:
        selected_cols.insert(0, "los")
    if "LOS" in common_cols and "LOS" not in selected_cols:
        selected_cols.insert(0, "LOS")

    tag = f"{args.summary}_TZ{args.time_zone}"
    if args.keep_all_summaries:
        tag += "_ALLSUM"

    if not args.skip_overall:
        print("[5A] Build Table A (overall MIMIC vs eICU)...")
        tableA = build_table_overall(
            mimic_sub, eicu,
            cols=selected_cols,
            name_a=f"MIMIC_{args.mimic_subset}",
            name_b="eICU",
            show_progress=show_progress
        )
        baseA = f"tableA_overall_mimic_{args.mimic_subset}_vs_eicu_{tag}"
        print("[6A] Save Table A...")
        save_csv_and_optional_xlsx(tableA, out_dir, baseA, no_xlsx=args.no_xlsx)

    if not args.skip_outcome:
        print("[5B] Build Table B (survive vs non-survive within each DB)...")
        tbl_mimic = build_table_within_outcome(
            mimic_sub, cols=selected_cols, dataset_name=f"MIMIC_{args.mimic_subset}", show_progress=show_progress
        )
        tbl_eicu = build_table_within_outcome(
            eicu, cols=selected_cols, dataset_name="eICU", show_progress=show_progress
        )
        combined = combine_two_within_tables(tbl_mimic, tbl_eicu)

        baseB = f"tableB_survive_vs_non_survive_mimic_{args.mimic_subset}_and_eicu_{tag}"
        print("[6B] Save Table B...")
        save_csv_and_optional_xlsx(combined, out_dir, baseB, no_xlsx=args.no_xlsx)

    print("\nDone.")


if __name__ == "__main__":
    main()
