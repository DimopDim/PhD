#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04b_audit_temporal_leakage.py

Target-leakage audit for the Stage-04 cumulative-window representation.

Question
--------
Can ICU LOS be inferred from the temporal NaN pattern alone, even when LOS,
icu_los_hours, window_available, endpoint metadata, and all clinical VALUES are
excluded from model predictors?

Why this matters
----------------
Stage 04 correctly enforces:

    endpoint > ICU discharge  -> all clinical descriptors at that endpoint = NaN

For cumulative features, once at least one clinical concept has been observed,
some cumulative values normally remain finite at each subsequent endpoint
while the patient is still in the ICU. Therefore, an all-NaN tail can reveal
approximately when the ICU stay ended. This script quantifies that channel
without changing any prior-stage output.

Audit layers
------------
A. Structural oracle (diagnostic only; NEVER a valid predictor)
   Uses Stage-04 metadata `window_available` to compute the last available
   integer endpoint. This shows how directly the representation boundary is
   mathematically linked to LOS.

B. Implicit NaN-tail detector (no LOS/discharge metadata in predictors)
   Uses only the 304 clinical columns to compute:
       last endpoint containing any finite clinical value.
   This tests whether the structural boundary is visible from X itself.

C. Missingness-only ML audit
   Discards every clinical value and retains only 48 endpoint-wise observed
   fractions:
       fraction of the 304 clinical descriptors that are finite at hour h.
   An ExtraTreesRegressor is evaluated with patient-level out-of-fold CV.

Three cohorts are evaluated:
1. full:
       target = total ICU LOS hours
2. early_discharge:
       ICU LOS <= 48 h
       target = total ICU LOS hours
       This is where a post-discharge NaN tail can directly encode the target.
3. landmark_48h:
       ICU LOS > 48 h
       target = remaining LOS hours = total LOS - 48
       No structural post-discharge tail exists inside the observation window.

Interpretation
--------------
A large performance gap:

    early_discharge missingness-only >> landmark_48h missingness-only

together with a high match rate between:

    implicit_last_observed_hour == oracle_last_available_hour

is strong evidence that the full-cohort 48 h representation contains a
discharge-timing leakage channel.

This audit DOES NOT:
- modify any Stage-04 file,
- alter the target,
- perform model selection for the manuscript,
- use clinical values as predictors,
- use demographics,
- use window_available / icu_los_hours / endpoint_hour as ML predictors.

Inputs
------
<project-root>/data/04_cumulative_windows/
    mimic/o1_01h.parquet
    eicu/o1_01h.parquet
    feature_order.json

Only o1 is required because its 48 hourly endpoints expose the finest temporal
boundary. The same final 48 h summary consistency was already checked by
Stage 04.

Outputs
-------
<project-root>/data/04_cumulative_windows/leakage_audit/
    leakage_patient_summary.csv
    leakage_metrics.csv
    missingness_oof_predictions.csv
    endpoint_missingness_summary.csv
    cohort_leakage_summary.csv
    04b_temporal_leakage_manifest.json
    04b_audit_temporal_leakage.log

Example
-------
cd /home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

python 04b_audit_temporal_leakage.py --self-test

python 04b_audit_temporal_leakage.py \
    --database all \
    --cv-folds 5 \
    --trees 300 \
    --model-workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor"
)
WINDOW_DIR_DEFAULT = (
    PROJECT_ROOT_DEFAULT / "data" / "04_cumulative_windows"
)
OUTPUT_DIR_DEFAULT = WINDOW_DIR_DEFAULT / "leakage_audit"

CV_FOLDS_DEFAULT = 5
TREES_DEFAULT = 300
MODEL_WORKERS_DEFAULT = 10
SEED_DEFAULT = 42

EXPECTED_HOURS = np.arange(1, 49, dtype=np.int16)


def require_pyarrow() -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "04b audit requires pyarrow. Install with: pip install pyarrow"
        ) from exc


def require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "04b audit requires scikit-learn. Install with: "
            "pip install scikit-learn"
        ) from exc


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("temporal_leakage_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        output_dir / "04b_audit_temporal_leakage.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_feature_order(window_dir: Path) -> List[str]:
    path = window_dir / "feature_order.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    feature_columns = payload.get("feature_columns")

    if not isinstance(feature_columns, list):
        raise ValueError(
            f"{path} does not contain a valid feature_columns list."
        )

    if len(feature_columns) != 304:
        raise ValueError(
            f"Leakage audit expects the locked primary Stage-04 "
            f"representation with 304 clinical columns; found "
            f"{len(feature_columns)}."
        )

    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("Duplicate clinical feature columns detected.")

    return [str(x) for x in feature_columns]


def scan_o1_missingness(
    *,
    path: Path,
    feature_columns: Sequence[str],
    batch_rows: int,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reduce the wide o1 Parquet to endpoint-level missingness summaries.

    Clinical values themselves are never retained. For each patient/hour we
    retain only:
      - observed_count
      - observed_fraction
      - all_clinical_nan

    LOS/window metadata are retained separately only for labels/diagnostics.
    """
    import pyarrow.parquet as pq

    if not path.is_file():
        raise FileNotFoundError(path)

    pf = pq.ParquetFile(path)

    required = {
        "patient_id",
        "stay_id",
        "endpoint_hour",
        "icu_los_hours",
        "window_available",
        *feature_columns,
    }
    missing = sorted(required - set(pf.schema.names))
    if missing:
        raise ValueError(
            f"{path} missing required columns: {missing[:20]}"
        )

    columns = [
        "patient_id",
        "stay_id",
        "endpoint_hour",
        "icu_los_hours",
        "window_available",
        *feature_columns,
    ]

    endpoint_parts: List[pd.DataFrame] = []
    scanned_rows = 0
    start = time.monotonic()

    for batch in pf.iter_batches(
        batch_size=batch_rows,
        columns=columns,
    ):
        df = batch.to_pandas()
        if df.empty:
            continue

        scanned_rows += len(df)

        clinical = df[list(feature_columns)]
        observed_count = clinical.notna().sum(axis=1).astype("int16")

        reduced = pd.DataFrame(
            {
                "patient_id": (
                    df["patient_id"].astype(str).str.strip()
                ),
                "stay_id": (
                    df["stay_id"].astype(str).str.strip()
                ),
                "endpoint_hour": pd.to_numeric(
                    df["endpoint_hour"], errors="coerce"
                ).astype("Int16"),
                "icu_los_hours": pd.to_numeric(
                    df["icu_los_hours"], errors="coerce"
                ),
                "window_available": (
                    df["window_available"].astype(bool)
                ),
                "observed_count": observed_count,
            }
        )

        reduced["observed_fraction"] = (
            reduced["observed_count"].astype("float32")
            / float(len(feature_columns))
        )
        reduced["all_clinical_nan"] = (
            reduced["observed_count"].eq(0)
        )

        endpoint_parts.append(reduced)

    elapsed = time.monotonic() - start

    if not endpoint_parts:
        raise RuntimeError(f"No rows read from {path}")

    endpoint_df = pd.concat(
        endpoint_parts,
        ignore_index=True,
    )

    # Strict Stage-04 shape checks.
    if endpoint_df[
        ["patient_id", "endpoint_hour"]
    ].duplicated().any():
        raise ValueError(
            f"{path}: duplicate patient/hour rows detected."
        )

    per_patient_counts = endpoint_df.groupby(
        "patient_id", sort=False
    ).size()

    if not (per_patient_counts.to_numpy() == 48).all():
        bad = per_patient_counts[
            per_patient_counts.ne(48)
        ].head(10)
        raise ValueError(
            f"{path}: every patient must have 48 o1 rows. "
            f"Examples: {bad.to_dict()}"
        )

    observed_hours = np.sort(
        endpoint_df["endpoint_hour"]
        .dropna()
        .unique()
        .astype(int)
    )
    if not np.array_equal(
        observed_hours,
        EXPECTED_HOURS.astype(int),
    ):
        raise ValueError(
            f"{path}: expected endpoint hours 1..48; "
            f"found {observed_hours.tolist()}."
        )

    # Ensure LOS/stay metadata are constant per patient.
    meta_check = (
        endpoint_df.groupby("patient_id", sort=False)
        .agg(
            stay_count=("stay_id", "nunique"),
            los_count=("icu_los_hours", "nunique"),
        )
    )
    if (meta_check["stay_count"] != 1).any():
        raise ValueError(f"{path}: stay_id changes within patient.")
    if (meta_check["los_count"] != 1).any():
        raise ValueError(f"{path}: icu_los_hours changes within patient.")

    # Patient-level structural and implicit boundary summaries.
    patient_rows: List[dict] = []

    for patient_id, g in endpoint_df.groupby(
        "patient_id",
        sort=False,
    ):
        g = g.sort_values(
            "endpoint_hour",
            kind="stable",
        )

        los_hours = float(g["icu_los_hours"].iloc[0])
        stay_id = str(g["stay_id"].iloc[0])

        available_hours = g.loc[
            g["window_available"],
            "endpoint_hour",
        ].to_numpy(dtype=int)

        finite_hours = g.loc[
            ~g["all_clinical_nan"],
            "endpoint_hour",
        ].to_numpy(dtype=int)

        oracle_last_available = (
            int(available_hours.max())
            if len(available_hours)
            else 0
        )
        implicit_last_observed = (
            int(finite_hours.max())
            if len(finite_hours)
            else 0
        )

        # Midpoint estimator for an integer-hour boundary:
        # if last available endpoint is h<48, true LOS usually lies in
        # [h, h+1). 48 is a lower bound for longer stays.
        oracle_midpoint = (
            min(48.0, oracle_last_available + 0.5)
            if oracle_last_available < 48
            else 48.0
        )
        implicit_midpoint = (
            min(48.0, implicit_last_observed + 0.5)
            if implicit_last_observed < 48
            else 48.0
        )

        patient_rows.append(
            {
                "patient_id": patient_id,
                "stay_id": stay_id,
                "los_hours": los_hours,
                "los_days": los_hours / 24.0,
                "early_discharge_le_48h": bool(
                    los_hours <= 48.0 + 1e-9
                ),
                "landmark_48h_gt_48h": bool(
                    los_hours > 48.0 + 1e-9
                ),
                "remaining_los_after_48h": max(
                    0.0,
                    los_hours - 48.0,
                ),
                # Diagnostic oracle: NOT a legitimate model predictor.
                "oracle_last_available_hour": oracle_last_available,
                "oracle_tail_midpoint_prediction_hours": oracle_midpoint,
                # Derived only from the clinical NaN pattern.
                "implicit_last_observed_hour": implicit_last_observed,
                "implicit_tail_midpoint_prediction_hours": implicit_midpoint,
                "implicit_matches_oracle_boundary": bool(
                    implicit_last_observed
                    == oracle_last_available
                ),
                "has_any_clinical_value": bool(
                    implicit_last_observed > 0
                ),
                "available_endpoint_count": int(
                    g["window_available"].sum()
                ),
                "all_nan_endpoint_count": int(
                    g["all_clinical_nan"].sum()
                ),
            }
        )

    patient_df = pd.DataFrame(patient_rows)

    logger.info(
        "%s scan complete | rows=%d patients=%d elapsed=%.1fs",
        path,
        scanned_rows,
        len(patient_df),
        elapsed,
    )

    return endpoint_df, patient_df


def build_missingness_matrix(
    endpoint_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    One patient per row; 48 predictors containing ONLY observed fractions.
    """
    pivot = endpoint_df.pivot(
        index="patient_id",
        columns="endpoint_hour",
        values="observed_fraction",
    )

    pivot = pivot.reindex(
        columns=EXPECTED_HOURS.astype(int),
    )

    if pivot.isna().any().any():
        raise ValueError(
            "Missing patient/hour rows while building missingness matrix."
        )

    feature_names = [
        f"observed_fraction_h{hour:02d}"
        for hour in EXPECTED_HOURS
    ]
    pivot.columns = feature_names

    return pivot.astype("float32"), feature_names


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae_hours": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse_hours": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        ) if len(y_true) >= 2 else np.nan,
        "mae_days": float(
            mean_absolute_error(y_true, y_pred) / 24.0
        ),
        "rmse_days": float(
            np.sqrt(mean_squared_error(y_true, y_pred)) / 24.0
        ),
    }


def correlation_metrics(
    x: np.ndarray,
    y: np.ndarray,
) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
        }

    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        pearson = np.nan
    else:
        pearson = float(
            np.corrcoef(x, y)[0, 1]
        )

    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()

    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        spearman = np.nan
    else:
        spearman = float(
            np.corrcoef(xr, yr)[0, 1]
        )

    return {
        "pearson_r": pearson,
        "spearman_r": spearman,
    }


def deterministic_audit_metrics(
    *,
    database: str,
    patient_df: pd.DataFrame,
) -> List[dict]:
    """
    Deterministic summaries for the structural oracle and the implicit NaN tail.
    """
    rows: List[dict] = []

    scenarios = [
        (
            "full",
            patient_df.index.to_numpy(),
            "los_hours",
        ),
        (
            "early_discharge_le_48h",
            patient_df.index[
                patient_df["early_discharge_le_48h"]
            ].to_numpy(),
            "los_hours",
        ),
        (
            "landmark_48h_gt_48h",
            patient_df.index[
                patient_df["landmark_48h_gt_48h"]
            ].to_numpy(),
            "remaining_los_after_48h",
        ),
    ]

    for scenario, indices, target_col in scenarios:
        sub = patient_df.loc[indices].copy()
        if sub.empty:
            continue

        y = sub[target_col].to_numpy(dtype=float)

        # Mean target baseline for scale/context.
        mean_pred = np.full(
            len(sub),
            np.mean(y),
            dtype=float,
        )
        base = regression_metrics(y, mean_pred)
        rows.append(
            {
                "database": database,
                "scenario": scenario,
                "audit_type": "naive_mean_target_baseline",
                "predictor_source": "target-distribution baseline only",
                "target": target_col,
                "n": int(len(sub)),
                **base,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
            }
        )

        # For full/early cohorts, direct LOS boundary comparisons are meaningful.
        if scenario != "landmark_48h_gt_48h":
            for label, pred_col, assoc_col in [
                (
                    "oracle_availability_tail",
                    "oracle_tail_midpoint_prediction_hours",
                    "oracle_last_available_hour",
                ),
                (
                    "implicit_nan_tail",
                    "implicit_tail_midpoint_prediction_hours",
                    "implicit_last_observed_hour",
                ),
            ]:
                pred = sub[pred_col].to_numpy(dtype=float)
                metrics = regression_metrics(y, pred)
                corr = correlation_metrics(
                    sub[assoc_col].to_numpy(dtype=float),
                    y,
                )
                rows.append(
                    {
                        "database": database,
                        "scenario": scenario,
                        "audit_type": label,
                        "predictor_source": (
                            "window_available metadata (diagnostic oracle; "
                            "never valid X)"
                            if label == "oracle_availability_tail"
                            else "clinical NaN pattern only"
                        ),
                        "target": target_col,
                        "n": int(len(sub)),
                        **metrics,
                        **corr,
                    }
                )

        # For landmark patients, the structural boundary should be constant at 48.
        if scenario == "landmark_48h_gt_48h":
            boundary = sub[
                "implicit_last_observed_hour"
            ].to_numpy(dtype=float)
            rows.append(
                {
                    "database": database,
                    "scenario": scenario,
                    "audit_type": "implicit_boundary_variability",
                    "predictor_source": "clinical NaN pattern only",
                    "target": target_col,
                    "n": int(len(sub)),
                    "mae_hours": np.nan,
                    "rmse_hours": np.nan,
                    "r2": np.nan,
                    "mae_days": np.nan,
                    "rmse_days": np.nan,
                    **correlation_metrics(boundary, y),
                }
            )

    return rows


def fit_oof_missingness_model(
    *,
    database: str,
    scenario: str,
    X: pd.DataFrame,
    meta: pd.DataFrame,
    target_col: str,
    cv_folds: int,
    trees: int,
    model_workers: int,
    seed: int,
) -> Tuple[dict, pd.DataFrame]:
    """
    ExtraTrees OOF audit using ONLY 48 observed-fraction predictors.
    """
    require_sklearn()

    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import KFold

    if len(meta) < max(20, cv_folds * 2):
        return (
            {
                "database": database,
                "scenario": scenario,
                "audit_type": "missingness_only_extratrees_oof",
                "predictor_source": (
                    "48 endpoint observed fractions; clinical values excluded"
                ),
                "target": target_col,
                "n": int(len(meta)),
                "status": "skipped_too_few_patients",
                "mae_hours": np.nan,
                "rmse_hours": np.nan,
                "r2": np.nan,
                "mae_days": np.nan,
                "rmse_days": np.nan,
                "pearson_r": np.nan,
                "spearman_r": np.nan,
            },
            pd.DataFrame(),
        )

    patient_ids = meta["patient_id"].astype(str).tolist()
    Xs = X.loc[patient_ids].to_numpy(dtype=np.float32)
    y = meta[target_col].to_numpy(dtype=np.float64)

    # Guard against accidental target/metadata leakage.
    if Xs.shape[1] != 48:
        raise ValueError(
            f"Missingness-only matrix must have exactly 48 columns; "
            f"found {Xs.shape[1]}."
        )
    if not np.isfinite(Xs).all():
        raise ValueError(
            "Observed-fraction matrix contains NaN/inf."
        )

    splitter = KFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    oof = np.full(len(meta), np.nan, dtype=float)
    fold_id = np.full(len(meta), -1, dtype=int)

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(Xs),
        start=1,
    ):
        model = ExtraTreesRegressor(
            n_estimators=trees,
            random_state=seed + fold,
            n_jobs=model_workers,
            min_samples_leaf=2,
            max_features=1.0,
        )
        model.fit(
            Xs[train_idx],
            y[train_idx],
        )
        oof[test_idx] = model.predict(
            Xs[test_idx]
        )
        fold_id[test_idx] = fold

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions incomplete.")

    metrics = regression_metrics(y, oof)
    corr = correlation_metrics(oof, y)

    metric_row = {
        "database": database,
        "scenario": scenario,
        "audit_type": "missingness_only_extratrees_oof",
        "predictor_source": (
            "48 endpoint observed fractions; all clinical values and "
            "LOS/discharge metadata excluded from X"
        ),
        "target": target_col,
        "n": int(len(meta)),
        "status": "ok",
        **metrics,
        **corr,
    }

    pred_df = pd.DataFrame(
        {
            "database": database,
            "scenario": scenario,
            "patient_id": patient_ids,
            "target": target_col,
            "y_true_hours": y,
            "y_pred_hours": oof,
            "fold": fold_id,
        }
    )

    return metric_row, pred_df


def endpoint_summary(
    database: str,
    endpoint_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cohort_name, mask in [
        (
            "full",
            np.ones(len(endpoint_df), dtype=bool),
        ),
        (
            "early_discharge_le_48h",
            endpoint_df["icu_los_hours"].to_numpy(dtype=float)
            <= 48.0 + 1e-9,
        ),
        (
            "landmark_48h_gt_48h",
            endpoint_df["icu_los_hours"].to_numpy(dtype=float)
            > 48.0 + 1e-9,
        ),
    ]:
        sub = endpoint_df.loc[mask]
        if sub.empty:
            continue

        grouped = (
            sub.groupby("endpoint_hour", sort=True)
            .agg(
                rows=("patient_id", "size"),
                mean_observed_fraction=(
                    "observed_fraction",
                    "mean",
                ),
                median_observed_fraction=(
                    "observed_fraction",
                    "median",
                ),
                all_nan_rows=(
                    "all_clinical_nan",
                    "sum",
                ),
                structurally_unavailable_rows=(
                    "window_available",
                    lambda x: int((~x.astype(bool)).sum()),
                ),
            )
            .reset_index()
        )
        grouped.insert(0, "cohort", cohort_name)
        grouped.insert(0, "database", database)
        grouped["all_nan_row_pct"] = (
            100.0
            * grouped["all_nan_rows"]
            / grouped["rows"]
        )
        grouped["structurally_unavailable_pct"] = (
            100.0
            * grouped["structurally_unavailable_rows"]
            / grouped["rows"]
        )

        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def cohort_summary(
    database: str,
    patient_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for cohort_name, mask in [
        (
            "full",
            np.ones(len(patient_df), dtype=bool),
        ),
        (
            "early_discharge_le_48h",
            patient_df["early_discharge_le_48h"].to_numpy(dtype=bool),
        ),
        (
            "landmark_48h_gt_48h",
            patient_df["landmark_48h_gt_48h"].to_numpy(dtype=bool),
        ),
    ]:
        sub = patient_df.loc[mask]
        if sub.empty:
            continue

        rows.append(
            {
                "database": database,
                "cohort": cohort_name,
                "patients": int(len(sub)),
                "los_mean_hours": float(sub["los_hours"].mean()),
                "los_median_hours": float(sub["los_hours"].median()),
                "los_min_hours": float(sub["los_hours"].min()),
                "los_max_hours": float(sub["los_hours"].max()),
                "patients_with_any_clinical_value": int(
                    sub["has_any_clinical_value"].sum()
                ),
                "patients_without_any_clinical_value": int(
                    (~sub["has_any_clinical_value"]).sum()
                ),
                "implicit_boundary_matches_oracle_count": int(
                    sub["implicit_matches_oracle_boundary"].sum()
                ),
                "implicit_boundary_matches_oracle_pct": float(
                    100.0
                    * sub["implicit_matches_oracle_boundary"].mean()
                ),
                "implicit_last_observed_hour_mean": float(
                    sub["implicit_last_observed_hour"].mean()
                ),
                "oracle_last_available_hour_mean": float(
                    sub["oracle_last_available_hour"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def run_database(
    *,
    database: str,
    window_dir: Path,
    output_dir: Path,
    feature_columns: Sequence[str],
    batch_rows: int,
    cv_folds: int,
    trees: int,
    model_workers: int,
    seed: int,
    skip_ml: bool,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict], dict]:
    path = (
        window_dir
        / database
        / "o1_01h.parquet"
    )

    logger.info("Auditing %s: %s", database, path)

    endpoint_df, patient_df = scan_o1_missingness(
        path=path,
        feature_columns=feature_columns,
        batch_rows=batch_rows,
        logger=logger,
    )

    patient_df.insert(
        0,
        "database",
        database,
    )

    X, missingness_features = build_missingness_matrix(
        endpoint_df
    )

    # Align patient summary to X.
    patient_df = patient_df.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(drop=True)

    if set(patient_df["patient_id"]) != set(X.index.astype(str)):
        raise ValueError(
            f"{database}: patient set mismatch between endpoint summary and X."
        )

    metric_rows = deterministic_audit_metrics(
        database=database,
        patient_df=patient_df,
    )

    oof_frames = []

    if not skip_ml:
        scenarios = [
            (
                "full",
                np.ones(len(patient_df), dtype=bool),
                "los_hours",
            ),
            (
                "early_discharge_le_48h",
                patient_df["early_discharge_le_48h"].to_numpy(dtype=bool),
                "los_hours",
            ),
            (
                "landmark_48h_gt_48h",
                patient_df["landmark_48h_gt_48h"].to_numpy(dtype=bool),
                "remaining_los_after_48h",
            ),
        ]

        for scenario, mask, target_col in scenarios:
            meta = patient_df.loc[mask].copy()

            logger.info(
                "%s/%s missingness-only OOF | n=%d target=%s",
                database,
                scenario,
                len(meta),
                target_col,
            )

            metric, preds = fit_oof_missingness_model(
                database=database,
                scenario=scenario,
                X=X,
                meta=meta,
                target_col=target_col,
                cv_folds=cv_folds,
                trees=trees,
                model_workers=model_workers,
                seed=seed,
            )
            metric_rows.append(metric)

            if not preds.empty:
                oof_frames.append(preds)

    endpoint_report = endpoint_summary(
        database,
        endpoint_df,
    )
    cohort_report = cohort_summary(
        database,
        patient_df,
    )

    # Core leakage diagnostics.
    valid_for_boundary = patient_df[
        patient_df["has_any_clinical_value"]
    ]
    boundary_match_pct = (
        100.0
        * valid_for_boundary[
            "implicit_matches_oracle_boundary"
        ].mean()
        if len(valid_for_boundary)
        else np.nan
    )

    early = patient_df[
        patient_df["early_discharge_le_48h"]
        & patient_df["has_any_clinical_value"]
    ]
    early_match_pct = (
        100.0
        * early[
            "implicit_matches_oracle_boundary"
        ].mean()
        if len(early)
        else np.nan
    )

    summary = {
        "database": database,
        "patients": int(len(patient_df)),
        "early_discharge_patients_le_48h": int(
            patient_df["early_discharge_le_48h"].sum()
        ),
        "landmark_patients_gt_48h": int(
            patient_df["landmark_48h_gt_48h"].sum()
        ),
        "patients_with_any_clinical_value": int(
            patient_df["has_any_clinical_value"].sum()
        ),
        "patients_without_any_clinical_value": int(
            (~patient_df["has_any_clinical_value"]).sum()
        ),
        "implicit_boundary_match_oracle_pct_among_patients_with_data": (
            float(boundary_match_pct)
        ),
        "early_discharge_implicit_boundary_match_oracle_pct": (
            float(early_match_pct)
        ),
        "missingness_predictor_count": int(
            len(missingness_features)
        ),
        "missingness_predictors": missingness_features,
    }

    oof = (
        pd.concat(
            oof_frames,
            ignore_index=True,
        )
        if oof_frames
        else pd.DataFrame()
    )

    return (
        patient_df,
        endpoint_report,
        cohort_report,
        oof,
        metric_rows,
        summary,
    )


def run_self_test() -> None:
    """
    Synthetic demonstration of the leakage mechanism.
    """
    # Patient A has LOS 18.4h and cumulative clinical information remains
    # present through hour 18, then the representation is all NaN.
    los = 18.4
    hours = np.arange(1, 49)
    available = hours <= los

    observed_count = np.where(
        available,
        25,
        0,
    )

    implicit_last = int(
        hours[observed_count > 0].max()
    )
    oracle_last = int(
        hours[available].max()
    )

    assert implicit_last == 18
    assert oracle_last == 18
    assert implicit_last == oracle_last

    midpoint = implicit_last + 0.5
    assert abs(midpoint - los) <= 0.5

    # Landmark patient remains in ICU through 48h: no structural tail.
    los_landmark = 83.2
    available_landmark = hours <= los_landmark
    assert available_landmark.all()

    print("Stage 04b self-test: PASS")
    print(
        "Synthetic early-discharge example: "
        "LOS=18.4h -> implicit NaN boundary=18h."
    )
    print(
        "Synthetic 48h-landmark example: "
        "LOS=83.2h -> all 48 observation endpoints remain available."
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Audit whether Stage-04 temporal NaN patterns leak ICU LOS."
        )
    )
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--window-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/04_cumulative_windows",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <window-dir>/leakage_audit",
    )
    p.add_argument(
        "--database",
        choices=["all", "mimic", "eicu"],
        default="all",
    )
    p.add_argument(
        "--batch-rows",
        type=int,
        default=5000,
        help="Parquet rows per scan batch (default: 5000).",
    )
    p.add_argument(
        "--cv-folds",
        type=int,
        default=CV_FOLDS_DEFAULT,
    )
    p.add_argument(
        "--trees",
        type=int,
        default=TREES_DEFAULT,
    )
    p.add_argument(
        "--model-workers",
        type=int,
        default=MODEL_WORKERS_DEFAULT,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )
    p.add_argument(
        "--skip-ml",
        action="store_true",
        help=(
            "Run deterministic NaN-tail diagnostics only; "
            "skip ExtraTrees OOF audit."
        ),
    )
    p.add_argument(
        "--self-test",
        action="store_true",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    require_pyarrow()

    if args.self_test:
        run_self_test()
        return 0

    if not args.skip_ml:
        require_sklearn()

    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be >=2.")
    if args.trees <= 0:
        raise ValueError("--trees must be positive.")
    if args.model_workers <= 0:
        raise ValueError("--model-workers must be positive.")

    project_root = args.project_root.expanduser().resolve()
    window_dir = (
        args.window_dir.expanduser().resolve()
        if args.window_dir is not None
        else project_root / "data" / "04_cumulative_windows"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else window_dir / "leakage_audit"
    )

    logger = configure_logging(output_dir)

    logger.info("Project root: %s", project_root)
    logger.info("Window directory: %s", window_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("CV folds: %d", args.cv_folds)
    logger.info("Trees: %d", args.trees)
    logger.info("Model workers: %d", args.model_workers)
    logger.info("Seed: %d", args.seed)
    logger.info("Skip ML: %s", args.skip_ml)

    feature_columns = load_feature_order(
        window_dir
    )
    logger.info(
        "Clinical Stage-04 columns: %d",
        len(feature_columns),
    )

    databases = (
        ["mimic", "eicu"]
        if args.database == "all"
        else [args.database]
    )

    patient_frames = []
    endpoint_frames = []
    cohort_frames = []
    oof_frames = []
    metric_rows: List[dict] = []
    db_summaries = {}

    start = time.monotonic()

    for database in databases:
        (
            patient_df,
            endpoint_report,
            cohort_report,
            oof,
            metrics,
            summary,
        ) = run_database(
            database=database,
            window_dir=window_dir,
            output_dir=output_dir,
            feature_columns=feature_columns,
            batch_rows=args.batch_rows,
            cv_folds=args.cv_folds,
            trees=args.trees,
            model_workers=args.model_workers,
            seed=args.seed,
            skip_ml=args.skip_ml,
            logger=logger,
        )

        patient_frames.append(patient_df)
        endpoint_frames.append(endpoint_report)
        cohort_frames.append(cohort_report)
        if not oof.empty:
            oof_frames.append(oof)
        metric_rows.extend(metrics)
        db_summaries[database] = summary

    patient_summary = pd.concat(
        patient_frames,
        ignore_index=True,
    )
    patient_summary.to_csv(
        output_dir / "leakage_patient_summary.csv",
        index=False,
    )

    endpoint_summary_df = pd.concat(
        endpoint_frames,
        ignore_index=True,
    )
    endpoint_summary_df.to_csv(
        output_dir / "endpoint_missingness_summary.csv",
        index=False,
    )

    cohort_summary_df = pd.concat(
        cohort_frames,
        ignore_index=True,
    )
    cohort_summary_df.to_csv(
        output_dir / "cohort_leakage_summary.csv",
        index=False,
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(
        output_dir / "leakage_metrics.csv",
        index=False,
    )

    if oof_frames:
        oof_df = pd.concat(
            oof_frames,
            ignore_index=True,
        )
    else:
        oof_df = pd.DataFrame(
            columns=[
                "database",
                "scenario",
                "patient_id",
                "target",
                "y_true_hours",
                "y_pred_hours",
                "fold",
            ]
        )

    oof_df.to_csv(
        output_dir / "missingness_oof_predictions.csv",
        index=False,
    )

    elapsed = time.monotonic() - start

    manifest = {
        "script": "04b_audit_temporal_leakage.py",
        "project_root": str(project_root),
        "window_dir": str(window_dir),
        "output_dir": str(output_dir),
        "database_requested": args.database,
        "clinical_values_used_as_ml_predictors": False,
        "demographics_used_as_ml_predictors": False,
        "los_or_discharge_metadata_used_as_ml_predictors": False,
        "ml_predictors": (
            "48 observed-fraction features, one per hourly endpoint; "
            "each equals finite clinical descriptor count / 304"
        ),
        "diagnostic_oracle_uses_window_available": True,
        "diagnostic_oracle_is_not_a_model_predictor": True,
        "full_target": "total ICU LOS hours",
        "early_discharge_target": "total ICU LOS hours for LOS <= 48h",
        "landmark_target": (
            "remaining ICU LOS hours after 48h for LOS > 48h"
        ),
        "cv_folds": int(args.cv_folds),
        "trees": int(args.trees),
        "model_workers": int(args.model_workers),
        "seed": int(args.seed),
        "skip_ml": bool(args.skip_ml),
        "databases": db_summaries,
        "elapsed_seconds": float(elapsed),
        "interpretation_rule": (
            "A high implicit-boundary/oracle match rate and strong "
            "early-discharge missingness-only LOS performance, especially "
            "if substantially stronger than the 48h-landmark result, "
            "supports a discharge-timing leakage channel in the full-cohort "
            "48h representation."
        ),
        "outputs": {
            "patient_summary": str(
                output_dir / "leakage_patient_summary.csv"
            ),
            "metrics": str(
                output_dir / "leakage_metrics.csv"
            ),
            "oof_predictions": str(
                output_dir / "missingness_oof_predictions.csv"
            ),
            "endpoint_missingness_summary": str(
                output_dir / "endpoint_missingness_summary.csv"
            ),
            "cohort_leakage_summary": str(
                output_dir / "cohort_leakage_summary.csv"
            ),
        },
    }

    manifest_path = (
        output_dir
        / "04b_temporal_leakage_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Audit complete in %.1fs. Saved: %s",
        elapsed,
        output_dir,
    )
    logger.info(
        "Primary reports: leakage_metrics.csv, "
        "cohort_leakage_summary.csv, "
        "04b_temporal_leakage_manifest.json"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
