#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_build_multilandmark_tensor_cubes.py

Stage 06 — causal hourly alignment and multi-landmark tensor construction.

This stage consumes the leakage-safe Stage-05 landmark files and constructs
patient-level tensors for the requested dynamic prediction landmarks.

Primary tensor
--------------
For landmark t and F frozen base predictors:

    X shape = (N, t, F, 4)

with channel order:

    o1 = 1-hour cumulative resolution
    o2 = 2-hour cumulative resolution
    o3 = 3-hour cumulative resolution
    o4 = 4-hour cumulative resolution

Causal alignment
----------------
A native cumulative summary may become visible only when its endpoint has
actually been reached. It is NEVER copied backward to earlier hours.

For a native resolution w, aligned hour h uses the latest completed endpoint

    e(h, w) = floor(h / w) * w,

provided e >= w. Before the first completed endpoint, clinical features remain
NaN. After a summary becomes available, it is carried forward only until the
next completed summary.

Example at an 8-hour landmark:

    hour:  1   2   3   4   5   6   7   8
    o1:    X1  X2  X3  X4  X5  X6  X7  X8
    o2:    NA  X2  X2  X4  X4  X6  X6  X8
    o3:    NA  NA  X3  X3  X3  X6  X6  X6
    o4:    NA  NA  NA  X4  X4  X4  X4  X8

Thus the o3 channel at t=8h contains information only through hour 6. The
future 9-hour endpoint is never used.

Static demographics
-------------------
Age, gender, and race are time-invariant and known independently of temporal
resolution. Stage 05 appends them to every native endpoint. Stage 06 validates
that they are constant within patient and then repeats them across the full
hourly axis and across all four resolution channels. This avoids artificial
resolution-dependent missingness in static covariates.

At t=1h, only the o1 clinical channel exists. o2/o3/o4 clinical values remain
NaN, while valid static demographics remain available. The manifest marks the
full four-resolution representation as unavailable at this landmark.

No imputation or scaling is performed.

Inputs
------
<project-root>/data/05_splits/
    05_multilandmark_manifest.json
    feature_order.json
    landmark_XXXh/
        mimic/train/o1_01h.parquet ...
        mimic/test/o1_01h.parquet ...
        eicu/external/o1_01h.parquet ...
        targets/*.parquet

Outputs
-------
<project-root>/data/06_numpy_cubes/
    landmark_001h/
        mimic_train.npz
        mimic_train.json
        mimic_test.npz
        mimic_test.json
        eicu_external.npz
        eicu_external.json
    ...
    reports/
        tensor_diagnostics.csv
        causal_alignment_audit.csv
        final_channel_consistency.csv
        target_alignment_audit.csv
        variant_availability.csv
    06_tensor_manifest.json
    06_build_multilandmark_tensor_cubes.log

NPZ arrays
----------
X
    float32, shape (N, landmark_hour, F, 4)

y_los_remaining_days
y_los_total_days
    float32, shape (N,)

y_mortality
    float32, shape (N,), NaN when mortality is unknown

mortality_known
    bool, shape (N,)

patient_id
stay_id
    Unicode arrays, shape (N,)

time_hours
    int16, [1, ..., landmark_hour]

channel_available
    bool, shape (4,); indicates whether at least one completed clinical native
    endpoint exists by the landmark.

source_endpoint_hour
    int16, shape (landmark_hour, 4); value 0 means that no native clinical
    endpoint is yet available at that aligned hour.

The script is deliberately strict. It aborts on feature-order mismatch,
patient/target misalignment, non-constant demographics, future-information
alignment, incorrect native endpoint grids, or final-channel inconsistency at
fully aligned landmarks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor"
)
STAGE05_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "05_splits"
OUTPUT_DIR_DEFAULT = PROJECT_ROOT_DEFAULT / "data" / "06_numpy_cubes"

CHANNEL_ORDER = ["o1", "o2", "o3", "o4"]
RESOLUTION_HOURS = {
    "o1": 1,
    "o2": 2,
    "o3": 3,
    "o4": 4,
}
RESOLUTION_FILENAMES = {
    "o1": "o1_01h.parquet",
    "o2": "o2_02h.parquet",
    "o3": "o3_03h.parquet",
    "o4": "o4_04h.parquet",
}
SPLIT_SPECS = [
    ("mimic", "train"),
    ("mimic", "test"),
    ("eicu", "external"),
]

METADATA_COLUMNS = [
    "database",
    "patient_id",
    "stay_id",
    "native_resolution_hours",
    "native_step_index",
    "endpoint_hour",
    "endpoint_minutes",
]

DEFAULT_ATOL = 1e-6
DEFAULT_RTOL = 1e-6


def normalize_database_label(value: object) -> str:
    """
    Normalize source metadata labels to the canonical Stage-05/06 names.

    Stage-04/05 Parquet files may preserve labels such as "MIMIC",
    "MIMIC-IV", "mimic_iv", "eICU", or "eICU-CRD" in the metadata column,
    while the directory structure uses canonical names "mimic" and "eicu".
    This function normalizes only the metadata label; it does not alter
    patient data or feature values.
    """
    if value is None or pd.isna(value):
        return ""

    raw = str(value).strip().lower()
    compact = re.sub(r"[^a-z0-9]+", "", raw)

    mimic_aliases = {
        "mimic",
        "mimiciv",
        "mimiciv31",
        "mimicivv31",
    }
    eicu_aliases = {
        "eicu",
        "eicucrd",
        "eicuv20",
        "eicu20",
    }

    if compact in mimic_aliases or compact.startswith("mimiciv"):
        return "mimic"

    if compact in eicu_aliases or compact.startswith("eicu"):
        return "eicu"

    return raw


@dataclass
class FeatureSchema:
    feature_names: List[str]
    clinical_names: List[str]
    demographic_names: List[str]
    clinical_count: int
    demographic_count: int
    feature_count: int


@dataclass
class TensorBundle:
    X: np.ndarray
    y_los_remaining_days: np.ndarray
    y_los_total_days: np.ndarray
    y_mortality: np.ndarray
    mortality_known: np.ndarray
    patient_ids: np.ndarray
    stay_ids: np.ndarray
    time_hours: np.ndarray
    channel_available: np.ndarray
    source_endpoint_hour: np.ndarray
    feature_schema: FeatureSchema


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage06_tensor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        output_dir / "06_build_multilandmark_tensor_cubes.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def parse_landmarks(text: Optional[str], stage05_manifest: dict) -> List[int]:
    available = [
        int(x)
        for x in stage05_manifest["landmarks_hours"]
    ]

    if text is None:
        return available

    requested = sorted(
        {
            int(x.strip())
            for x in text.split(",")
            if x.strip()
        }
    )

    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(
            "Requested landmarks are not present in Stage 05: "
            f"{missing}. Available={available}"
        )

    return requested


def load_feature_schema(
    stage05_dir: Path,
    stage05_manifest: dict,
) -> FeatureSchema:
    feature_json = load_json(
        stage05_dir / "feature_order.json"
    )

    names = [
        str(x)
        for x in feature_json["feature_columns"]
    ]

    clinical_count = int(
        stage05_manifest["clinical_feature_count"]
    )
    demographic_count = int(
        stage05_manifest["demographic_feature_count"]
    )
    feature_count = int(
        stage05_manifest["base_feature_count"]
    )

    if len(names) != feature_count:
        raise ValueError(
            f"Feature-order file contains {len(names)} columns, "
            f"manifest expects {feature_count}."
        )

    if clinical_count + demographic_count != feature_count:
        raise ValueError(
            "Manifest clinical + demographic counts do not equal "
            "base_feature_count."
        )

    if len(set(names)) != len(names):
        raise ValueError(
            "Duplicate names in Stage-05 feature order."
        )

    clinical_names = names[:clinical_count]
    demographic_names = names[clinical_count:]

    if len(demographic_names) != demographic_count:
        raise AssertionError(
            "Demographic feature slice width mismatch."
        )

    return FeatureSchema(
        feature_names=names,
        clinical_names=clinical_names,
        demographic_names=demographic_names,
        clinical_count=clinical_count,
        demographic_count=demographic_count,
        feature_count=feature_count,
    )


def canonical_target_path(
    stage05_dir: Path,
    landmark: int,
    database: str,
    split_name: str,
) -> Path:
    return (
        stage05_dir
        / f"landmark_{landmark:03d}h"
        / "targets"
        / f"{database}_{split_name}_targets.parquet"
    )


def canonical_native_path(
    stage05_dir: Path,
    landmark: int,
    database: str,
    split_name: str,
    channel: str,
) -> Path:
    return (
        stage05_dir
        / f"landmark_{landmark:03d}h"
        / database
        / split_name
        / RESOLUTION_FILENAMES[channel]
    )


def load_targets(
    path: Path,
    *,
    landmark: int,
    database: str,
    split_name: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {
        "database",
        "split",
        "landmark_hour",
        "patient_id",
        "stay_id",
        "los_total_days",
        "los_remaining_days",
        "hospital_expire_flag",
        "mortality_known",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path}: missing target columns {missing}"
        )

    out = df[
        [
            "database",
            "split",
            "landmark_hour",
            "patient_id",
            "stay_id",
            "los_total_days",
            "los_remaining_days",
            "hospital_expire_flag",
            "mortality_known",
        ]
    ].copy()

    out["patient_id"] = (
        out["patient_id"].astype(str).str.strip()
    )
    out["stay_id"] = (
        out["stay_id"].astype(str).str.strip()
    )

    if out["patient_id"].duplicated().any():
        raise ValueError(
            f"{database}/{split_name}/{landmark}h: duplicate target patient_id."
        )

    if out["stay_id"].duplicated().any():
        raise ValueError(
            f"{database}/{split_name}/{landmark}h: duplicate target stay_id."
        )

    if not out["database"].eq(database).all():
        raise ValueError(
            f"{path}: database label mismatch."
        )

    if not out["split"].eq(split_name).all():
        raise ValueError(
            f"{path}: split label mismatch."
        )

    if not (
        pd.to_numeric(
            out["landmark_hour"],
            errors="coerce",
        )
        .eq(landmark)
        .all()
    ):
        raise ValueError(
            f"{path}: landmark label mismatch."
        )

    for column in [
        "los_total_days",
        "los_remaining_days",
    ]:
        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    if out["los_total_days"].isna().any():
        raise ValueError(
            f"{path}: missing total LOS."
        )

    if out["los_remaining_days"].isna().any():
        raise ValueError(
            f"{path}: missing remaining LOS."
        )

    if (out["los_remaining_days"] <= 0).any():
        raise ValueError(
            f"{path}: remaining LOS must be strictly positive."
        )

    out["mortality_known"] = (
        out["mortality_known"].astype(bool)
    )

    known = out["mortality_known"]
    if (
        out.loc[
            known,
            "hospital_expire_flag",
        ]
        .isna()
        .any()
    ):
        raise ValueError(
            f"{path}: mortality_known=True with missing mortality label."
        )

    if (
        out.loc[
            ~known,
            "hospital_expire_flag",
        ]
        .notna()
        .any()
    ):
        raise ValueError(
            f"{path}: mortality_known=False with non-missing mortality label."
        )

    if known.any():
        known_values = pd.to_numeric(
            out.loc[
                known,
                "hospital_expire_flag",
            ],
            errors="coerce",
        )
        if not known_values.isin([0, 1]).all():
            raise ValueError(
                f"{path}: known mortality labels must be binary."
            )

    return out.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(drop=True)


def manifest_resolution_entry(
    stage05_manifest: dict,
    *,
    landmark: int,
    database: str,
    split_name: str,
    channel: str,
) -> dict:
    key = f"{database}_{split_name}"
    return (
        stage05_manifest["landmarks"][str(landmark)]
        ["splits"][key]["resolutions"][channel]
    )


def validate_expected_native_endpoints(
    df: pd.DataFrame,
    *,
    landmark: int,
    resolution_hours: int,
    database: str,
    split_name: str,
    channel: str,
) -> List[int]:
    expected = list(
        range(
            resolution_hours,
            landmark + 1,
            resolution_hours,
        )
    )

    observed = sorted(
        pd.to_numeric(
            df["endpoint_hour"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if observed != expected:
        raise ValueError(
            f"{database}/{split_name}/{landmark}h/{channel}: "
            f"native endpoint mismatch. expected={expected}, observed={observed}"
        )

    return expected


def load_native(
    path: Path,
    *,
    landmark: int,
    database: str,
    split_name: str,
    channel: str,
    schema: FeatureSchema,
    target_patient_ids: Sequence[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    expected_columns = (
        METADATA_COLUMNS
        + schema.feature_names
    )

    df = pd.read_parquet(
        path,
        columns=expected_columns,
    )

    df["patient_id"] = (
        df["patient_id"].astype(str).str.strip()
    )
    df["stay_id"] = (
        df["stay_id"].astype(str).str.strip()
    )

    if list(df.columns) != expected_columns:
        raise ValueError(
            f"{path}: feature schema/order differs from Stage-05 frozen schema."
        )

    raw_database_labels = sorted(
        df["database"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    normalized_database = df[
        "database"
    ].map(normalize_database_label)

    if not normalized_database.eq(database).all():
        normalized_unique = sorted(
            normalized_database.unique().tolist()
        )
        raise ValueError(
            f"{path}: database label mismatch after normalization. "
            f"expected={database!r}, raw_labels={raw_database_labels}, "
            f"normalized_labels={normalized_unique}"
        )

    # Canonicalize metadata in memory so all Stage-06 outputs use the
    # directory/manifest naming convention consistently.
    df["database"] = database

    expected_resolution = RESOLUTION_HOURS[channel]
    if not (
        pd.to_numeric(
            df["native_resolution_hours"],
            errors="raise",
        )
        .astype(int)
        .eq(expected_resolution)
        .all()
    ):
        raise ValueError(
            f"{path}: native_resolution_hours mismatch for {channel}."
        )

    if (
        pd.to_numeric(
            df["endpoint_hour"],
            errors="raise",
        )
        .gt(landmark)
        .any()
    ):
        raise ValueError(
            f"{path}: future endpoint > landmark found."
        )

    validate_expected_native_endpoints(
        df,
        landmark=landmark,
        resolution_hours=expected_resolution,
        database=database,
        split_name=split_name,
        channel=channel,
    )

    expected_patient_set = set(
        map(str, target_patient_ids)
    )
    observed_patient_set = set(
        df["patient_id"].astype(str).unique()
    )

    if observed_patient_set != expected_patient_set:
        missing = sorted(
            expected_patient_set - observed_patient_set
        )[:10]
        extra = sorted(
            observed_patient_set - expected_patient_set
        )[:10]
        raise ValueError(
            f"{database}/{split_name}/{landmark}h/{channel}: "
            "patient set differs from target file. "
            f"missing_examples={missing}, extra_examples={extra}"
        )

    expected_steps = len(
        range(
            expected_resolution,
            landmark + 1,
            expected_resolution,
        )
    )

    counts = (
        df.groupby(
            "patient_id",
            sort=False,
        )
        .size()
    )
    if not (
        counts.to_numpy() == expected_steps
    ).all():
        raise ValueError(
            f"{database}/{split_name}/{landmark}h/{channel}: "
            "incorrect native row count per patient."
        )

    return df.sort_values(
        ["patient_id", "endpoint_hour"],
        kind="stable",
    ).reset_index(drop=True)


def values_equal_with_nan(
    a: np.ndarray,
    b: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> Tuple[bool, int, float]:
    if a.shape != b.shape:
        return False, max(a.size, b.size), math.inf

    a = np.asarray(a)
    b = np.asarray(b)

    nan_a = np.isnan(a)
    nan_b = np.isnan(b)

    nan_mask_mismatch = nan_a ^ nan_b
    n_nan_mismatch = int(
        nan_mask_mismatch.sum()
    )

    finite = ~(nan_a | nan_b)
    finite_close = np.isclose(
        a[finite],
        b[finite],
        atol=atol,
        rtol=rtol,
    )

    n_finite_mismatch = int(
        (~finite_close).sum()
    )

    if finite.any():
        max_abs = float(
            np.max(
                np.abs(
                    a[finite] - b[finite]
                )
            )
        )
    else:
        max_abs = 0.0

    return (
        n_nan_mismatch + n_finite_mismatch == 0,
        n_nan_mismatch + n_finite_mismatch,
        max_abs,
    )


def extract_static_demographics(
    o1_df: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    schema: FeatureSchema,
    atol: float,
    rtol: float,
) -> Tuple[np.ndarray, dict]:
    """
    Use o1 as the canonical source of time-invariant demographics.

    Returns array shape (N, D) in target patient order.
    """
    n_patients = len(targets)
    demo = np.full(
        (
            n_patients,
            schema.demographic_count,
        ),
        np.nan,
        dtype=np.float32,
    )

    patient_to_index = {
        pid: i
        for i, pid in enumerate(
            targets["patient_id"].astype(str)
        )
    }

    nonconstant_patients = 0
    max_abs_difference = 0.0

    for patient_id, group in o1_df.groupby(
        "patient_id",
        sort=False,
    ):
        arr = group[
            schema.demographic_names
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )

        reference = arr[0]

        if arr.shape[0] > 1:
            repeated = np.repeat(
                reference[None, :],
                arr.shape[0],
                axis=0,
            )
            equal, _, max_abs = values_equal_with_nan(
                arr,
                repeated,
                atol=atol,
                rtol=rtol,
            )
            max_abs_difference = max(
                max_abs_difference,
                max_abs,
            )
            if not equal:
                nonconstant_patients += 1

        demo[
            patient_to_index[str(patient_id)],
            :,
        ] = reference.astype(
            np.float32,
            copy=False,
        )

    if nonconstant_patients:
        raise ValueError(
            f"Found {nonconstant_patients} patients with non-constant "
            "demographics across o1 native endpoints."
        )

    if np.isnan(
        demo[:, 1:]
    ).any():
        # Gender/race one-hot columns should never be missing. Age may be NaN.
        bad = int(
            np.isnan(
                demo[:, 1:]
            ).sum()
        )
        raise ValueError(
            f"Found {bad} missing values in encoded categorical demographics."
        )

    audit = {
        "patients": int(n_patients),
        "demographic_features": int(
            schema.demographic_count
        ),
        "age_missing_patients": int(
            np.isnan(demo[:, 0]).sum()
        ),
        "nonconstant_demographic_patients": int(
            nonconstant_patients
        ),
        "max_abs_demographic_difference": float(
            max_abs_difference
        ),
    }

    return demo, audit


def validate_native_demographics(
    native_df: pd.DataFrame,
    *,
    targets: pd.DataFrame,
    static_demo: np.ndarray,
    schema: FeatureSchema,
    atol: float,
    rtol: float,
    label: str,
) -> dict:
    patient_to_index = {
        pid: i
        for i, pid in enumerate(
            targets["patient_id"].astype(str)
        )
    }

    mismatched_patients = 0
    max_abs_difference = 0.0

    for patient_id, group in native_df.groupby(
        "patient_id",
        sort=False,
    ):
        idx = patient_to_index[str(patient_id)]
        reference = static_demo[idx].astype(
            np.float64,
            copy=False,
        )
        arr = group[
            schema.demographic_names
        ].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        expected = np.repeat(
            reference[None, :],
            arr.shape[0],
            axis=0,
        )

        equal, _, max_abs = values_equal_with_nan(
            arr,
            expected,
            atol=atol,
            rtol=rtol,
        )
        max_abs_difference = max(
            max_abs_difference,
            max_abs,
        )
        if not equal:
            mismatched_patients += 1

    if mismatched_patients:
        raise ValueError(
            f"{label}: demographics differ from canonical o1 demographics "
            f"for {mismatched_patients} patients."
        )

    return {
        "label": label,
        "mismatched_patients": int(
            mismatched_patients
        ),
        "max_abs_difference": float(
            max_abs_difference
        ),
    }


def build_source_endpoint_map(
    landmark: int,
    resolution_hours: int,
) -> np.ndarray:
    """
    For each aligned hour 1..t, return latest native endpoint <= hour.
    0 means no completed native endpoint exists yet.
    """
    hours = np.arange(
        1,
        landmark + 1,
        dtype=np.int16,
    )

    source = (
        hours // int(resolution_hours)
    ) * int(resolution_hours)

    source[
        source < int(resolution_hours)
    ] = 0

    if np.any(
        source > hours
    ):
        raise AssertionError(
            "Causal alignment generated future endpoint."
        )

    return source.astype(
        np.int16,
        copy=False,
    )


def build_causal_channel(
    native_df: Optional[pd.DataFrame],
    *,
    targets: pd.DataFrame,
    static_demo: np.ndarray,
    schema: FeatureSchema,
    landmark: int,
    channel: str,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Construct one aligned channel of shape (N, t, F).

    Clinical block:
        NaN before first completed endpoint.
        Latest completed cumulative summary carried forward causally.

    Demographic block:
        repeated across all t positions because it is time-invariant.
    """
    n_patients = len(targets)
    t = int(landmark)
    f = int(schema.feature_count)
    c = int(schema.clinical_count)

    aligned = np.full(
        (n_patients, t, f),
        np.nan,
        dtype=np.float32,
    )

    # Static covariates are known at admission and are not resolution-specific.
    aligned[
        :,
        :,
        c:,
    ] = static_demo[:, None, :]

    resolution = RESOLUTION_HOURS[channel]
    source_endpoint = build_source_endpoint_map(
        landmark,
        resolution,
    )

    if native_df is None:
        # No clinical native endpoint exists by this landmark.
        if np.any(source_endpoint != 0):
            raise AssertionError(
                f"{channel}: native data missing despite available endpoint."
            )
        return (
            aligned,
            source_endpoint,
            False,
        )

    patient_order = (
        targets["patient_id"].astype(str).tolist()
    )

    # Matrix of native clinical summaries keyed by patient and endpoint.
    for patient_index, patient_id in enumerate(
        patient_order
    ):
        group = native_df.loc[
            native_df["patient_id"].eq(patient_id)
        ]

        endpoint_to_values = {
            int(endpoint): values
            for endpoint, values in zip(
                group["endpoint_hour"].astype(int),
                group[
                    schema.clinical_names
                ].to_numpy(
                    dtype=np.float32,
                    copy=True,
                ),
            )
        }

        for hour_index, endpoint in enumerate(
            source_endpoint
        ):
            endpoint = int(endpoint)
            if endpoint == 0:
                continue

            if endpoint not in endpoint_to_values:
                raise ValueError(
                    f"{channel}: patient={patient_id} missing expected "
                    f"native endpoint={endpoint}h."
                )

            aligned[
                patient_index,
                hour_index,
                :c,
            ] = endpoint_to_values[
                endpoint
            ]

    return (
        aligned,
        source_endpoint,
        True,
    )


def make_mortality_array(
    targets: pd.DataFrame,
) -> np.ndarray:
    y = np.full(
        len(targets),
        np.nan,
        dtype=np.float32,
    )

    known = targets[
        "mortality_known"
    ].to_numpy(dtype=bool)

    if known.any():
        y[known] = pd.to_numeric(
            targets.loc[
                known,
                "hospital_expire_flag",
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.float32,
        )

    return y


def build_split_tensor(
    *,
    stage05_dir: Path,
    stage05_manifest: dict,
    schema: FeatureSchema,
    landmark: int,
    database: str,
    split_name: str,
    atol: float,
    rtol: float,
) -> Tuple[
    TensorBundle,
    List[dict],
    List[dict],
    List[dict],
]:
    target_path = canonical_target_path(
        stage05_dir,
        landmark,
        database,
        split_name,
    )
    targets = load_targets(
        target_path,
        landmark=landmark,
        database=database,
        split_name=split_name,
    )

    patient_ids = (
        targets["patient_id"]
        .astype(str)
        .to_numpy(dtype=str)
    )
    stay_ids = (
        targets["stay_id"]
        .astype(str)
        .to_numpy(dtype=str)
    )

    # o1 is always available for landmark >=1 and is the static-demographic
    # reference source.
    o1_entry = manifest_resolution_entry(
        stage05_manifest,
        landmark=landmark,
        database=database,
        split_name=split_name,
        channel="o1",
    )
    if not bool(
        o1_entry["available"]
    ):
        raise RuntimeError(
            f"{database}/{split_name}/{landmark}h: o1 must be available."
        )

    o1_df = load_native(
        canonical_native_path(
            stage05_dir,
            landmark,
            database,
            split_name,
            "o1",
        ),
        landmark=landmark,
        database=database,
        split_name=split_name,
        channel="o1",
        schema=schema,
        target_patient_ids=patient_ids,
    )

    static_demo, static_audit = (
        extract_static_demographics(
            o1_df,
            targets=targets,
            schema=schema,
            atol=atol,
            rtol=rtol,
        )
    )

    alignment_rows: List[dict] = []
    demographic_rows: List[dict] = []
    target_rows: List[dict] = []

    channels: List[np.ndarray] = []
    source_maps: List[np.ndarray] = []
    available_flags: List[bool] = []

    for channel in CHANNEL_ORDER:
        entry = manifest_resolution_entry(
            stage05_manifest,
            landmark=landmark,
            database=database,
            split_name=split_name,
            channel=channel,
        )
        available = bool(
            entry["available"]
        )

        native_df: Optional[pd.DataFrame]
        if available:
            native_df = load_native(
                canonical_native_path(
                    stage05_dir,
                    landmark,
                    database,
                    split_name,
                    channel,
                ),
                landmark=landmark,
                database=database,
                split_name=split_name,
                channel=channel,
                schema=schema,
                target_patient_ids=patient_ids,
            )

            demo_check = validate_native_demographics(
                native_df,
                targets=targets,
                static_demo=static_demo,
                schema=schema,
                atol=atol,
                rtol=rtol,
                label=(
                    f"{database}/{split_name}/"
                    f"{landmark}h/{channel}"
                ),
            )
            demographic_rows.append(
                {
                    "database": database,
                    "split": split_name,
                    "landmark_hour": landmark,
                    "channel": channel,
                    **demo_check,
                }
            )
        else:
            native_df = None

        aligned, source_endpoint, built_available = (
            build_causal_channel(
                native_df,
                targets=targets,
                static_demo=static_demo,
                schema=schema,
                landmark=landmark,
                channel=channel,
            )
        )

        if built_available != available:
            raise AssertionError(
                f"{database}/{split_name}/{landmark}h/{channel}: "
                "availability mismatch between manifest and tensor builder."
            )

        # Core no-look-ahead assertion.
        hours = np.arange(
            1,
            landmark + 1,
            dtype=np.int16,
        )
        future_violations = int(
            np.sum(
                source_endpoint > hours
            )
        )
        if future_violations:
            raise RuntimeError(
                f"{database}/{split_name}/{landmark}h/{channel}: "
                f"{future_violations} future-information alignment violations."
            )

        nonzero = source_endpoint[
            source_endpoint > 0
        ]
        latest_source = (
            int(nonzero[-1])
            if len(nonzero)
            else None
        )

        manifest_latest = entry.get(
            "latest_native_endpoint_hour"
        )
        if manifest_latest is not None:
            manifest_latest = int(
                manifest_latest
            )

        if latest_source != manifest_latest:
            raise RuntimeError(
                f"{database}/{split_name}/{landmark}h/{channel}: "
                f"source map latest={latest_source}, "
                f"Stage05 manifest latest={manifest_latest}."
            )

        leading_hours_without_clinical_endpoint = int(
            np.sum(
                source_endpoint == 0
            )
        )

        alignment_rows.append(
            {
                "database": database,
                "split": split_name,
                "landmark_hour": int(
                    landmark
                ),
                "channel": channel,
                "resolution_hours": int(
                    RESOLUTION_HOURS[channel]
                ),
                "available": bool(
                    available
                ),
                "aligned_hours": int(
                    landmark
                ),
                "native_steps": int(
                    len(
                        np.unique(
                            source_endpoint[
                                source_endpoint > 0
                            ]
                        )
                    )
                ),
                "latest_source_endpoint_hour": latest_source,
                "lag_to_landmark_hours": (
                    int(
                        landmark - latest_source
                    )
                    if latest_source is not None
                    else np.nan
                ),
                "leading_hours_without_clinical_endpoint": int(
                    leading_hours_without_clinical_endpoint
                ),
                "future_endpoint_violations": int(
                    future_violations
                ),
                "source_endpoint_map": json.dumps(
                    source_endpoint.tolist()
                ),
            }
        )

        channels.append(
            aligned[..., None]
        )
        source_maps.append(
            source_endpoint[:, None]
        )
        available_flags.append(
            available
        )

    X = np.concatenate(
        channels,
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )

    source_endpoint_hour = np.concatenate(
        source_maps,
        axis=1,
    ).astype(
        np.int16,
        copy=False,
    )

    expected_shape = (
        len(targets),
        landmark,
        schema.feature_count,
        4,
    )
    if X.shape != expected_shape:
        raise RuntimeError(
            f"{database}/{split_name}/{landmark}h: "
            f"tensor shape={X.shape}, expected={expected_shape}."
        )

    y_los_remaining = targets[
        "los_remaining_days"
    ].to_numpy(
        dtype=np.float32,
    )
    y_los_total = targets[
        "los_total_days"
    ].to_numpy(
        dtype=np.float32,
    )
    mortality_known = targets[
        "mortality_known"
    ].to_numpy(
        dtype=bool,
    )
    y_mortality = make_mortality_array(
        targets
    )

    # Alignment report is intentionally explicit although the target file is
    # the canonical order.
    target_rows.append(
        {
            "database": database,
            "split": split_name,
            "landmark_hour": int(
                landmark
            ),
            "patients": int(
                len(targets)
            ),
            "unique_patient_ids": int(
                targets["patient_id"].nunique()
            ),
            "unique_stay_ids": int(
                targets["stay_id"].nunique()
            ),
            "remaining_los_missing": int(
                np.isnan(
                    y_los_remaining
                ).sum()
            ),
            "remaining_los_nonpositive": int(
                np.sum(
                    y_los_remaining <= 0
                )
            ),
            "mortality_known": int(
                mortality_known.sum()
            ),
            "mortality_missing": int(
                (~mortality_known).sum()
            ),
            "mortality_events_among_known": int(
                np.nansum(
                    y_mortality
                )
            ),
            "static_age_missing_patients": int(
                static_audit[
                    "age_missing_patients"
                ]
            ),
            "target_alignment_pass": True,
        }
    )

    bundle = TensorBundle(
        X=X,
        y_los_remaining_days=y_los_remaining,
        y_los_total_days=y_los_total,
        y_mortality=y_mortality,
        mortality_known=mortality_known,
        patient_ids=patient_ids,
        stay_ids=stay_ids,
        time_hours=np.arange(
            1,
            landmark + 1,
            dtype=np.int16,
        ),
        channel_available=np.asarray(
            available_flags,
            dtype=bool,
        ),
        source_endpoint_hour=source_endpoint_hour,
        feature_schema=schema,
    )

    return (
        bundle,
        alignment_rows,
        demographic_rows,
        target_rows,
    )


def final_channel_consistency_rows(
    *,
    bundle: TensorBundle,
    landmark: int,
    database: str,
    split_name: str,
    atol: float,
    rtol: float,
) -> List[dict]:
    rows = []

    reference = bundle.X[
        :,
        -1,
        :,
        0,
    ]

    ref_endpoint = int(
        bundle.source_endpoint_hour[
            -1,
            0,
        ]
    )

    for channel_index, channel in enumerate(
        CHANNEL_ORDER
    ):
        candidate = bundle.X[
            :,
            -1,
            :,
            channel_index,
        ]

        candidate_endpoint = int(
            bundle.source_endpoint_hour[
                -1,
                channel_index,
            ]
        )

        comparable = (
            ref_endpoint == landmark
            and candidate_endpoint == landmark
        )

        if comparable:
            equal, n_mismatch, max_abs = (
                values_equal_with_nan(
                    reference.astype(
                        np.float64
                    ),
                    candidate.astype(
                        np.float64
                    ),
                    atol=atol,
                    rtol=rtol,
                )
            )
        else:
            equal = None
            n_mismatch = None
            max_abs = None

        rows.append(
            {
                "database": database,
                "split": split_name,
                "landmark_hour": int(
                    landmark
                ),
                "reference_channel": "o1",
                "compared_channel": channel,
                "reference_endpoint_hour": int(
                    ref_endpoint
                ),
                "candidate_endpoint_hour": int(
                    candidate_endpoint
                ),
                "comparable_same_landmark_endpoint": bool(
                    comparable
                ),
                "all_equal_with_nan": (
                    bool(equal)
                    if equal is not None
                    else np.nan
                ),
                "n_mismatched": (
                    int(n_mismatch)
                    if n_mismatch is not None
                    else np.nan
                ),
                "max_abs_finite_difference": (
                    float(max_abs)
                    if max_abs is not None
                    else np.nan
                ),
            }
        )

    return rows


def build_variant_availability_row(
    bundle: TensorBundle,
    *,
    landmark: int,
) -> dict:
    available = {
        channel: bool(
            bundle.channel_available[i]
        )
        for i, channel in enumerate(
            CHANNEL_ORDER
        )
    }

    latest = {
        channel: int(
            bundle.source_endpoint_hour[
                -1,
                i,
            ]
        )
        for i, channel in enumerate(
            CHANNEL_ORDER
        )
    }

    all_four_available = all(
        available.values()
    )
    all_four_aligned = (
        all_four_available
        and all(
            latest[channel] == landmark
            for channel in CHANNEL_ORDER
        )
    )

    return {
        "landmark_hour": int(
            landmark
        ),
        "o1_available": available["o1"],
        "o2_available": available["o2"],
        "o3_available": available["o3"],
        "o4_available": available["o4"],
        "full_four_channel_available": bool(
            all_four_available
        ),
        "full_four_channel_fully_aligned": bool(
            all_four_aligned
        ),
        "static_o1_landmark_summary_available": bool(
            latest["o1"] == landmark
        ),
        "o1_final_endpoint_hour": latest["o1"],
        "o2_final_endpoint_hour": latest["o2"],
        "o3_final_endpoint_hour": latest["o3"],
        "o4_final_endpoint_hour": latest["o4"],
        "recommended_primary_variants": (
            "o1,static"
            if not all_four_available
            else "full,o1,o2,o3,o4,static"
        ),
        "strict_four_resolution_comparison": bool(
            all_four_aligned
        ),
    }


def save_bundle(
    bundle: TensorBundle,
    *,
    output_dir: Path,
    database: str,
    split_name: str,
    landmark: int,
    compressed: bool,
) -> Tuple[Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stem = f"{database}_{split_name}"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    arrays = {
        "X": bundle.X,
        "y_los_remaining_days": (
            bundle.y_los_remaining_days
        ),
        "y_los_total_days": (
            bundle.y_los_total_days
        ),
        "y_mortality": bundle.y_mortality,
        "mortality_known": (
            bundle.mortality_known
        ),
        "patient_id": bundle.patient_ids,
        "stay_id": bundle.stay_ids,
        "time_hours": bundle.time_hours,
        "channel_available": (
            bundle.channel_available
        ),
        "source_endpoint_hour": (
            bundle.source_endpoint_hour
        ),
    }

    if compressed:
        np.savez_compressed(
            npz_path,
            **arrays,
        )
    else:
        np.savez(
            npz_path,
            **arrays,
        )

    known = bundle.mortality_known

    metadata = {
        "database": database,
        "split": split_name,
        "landmark_hour": int(
            landmark
        ),
        "tensor_shape": list(
            bundle.X.shape
        ),
        "dtype": str(
            bundle.X.dtype
        ),
        "feature_count": int(
            bundle.feature_schema.feature_count
        ),
        "clinical_feature_count": int(
            bundle.feature_schema.clinical_count
        ),
        "demographic_feature_count": int(
            bundle.feature_schema.demographic_count
        ),
        "channel_order": CHANNEL_ORDER,
        "channel_available": {
            channel: bool(
                bundle.channel_available[i]
            )
            for i, channel in enumerate(
                CHANNEL_ORDER
            )
        },
        "time_hours": (
            bundle.time_hours.tolist()
        ),
        "source_endpoint_hour": {
            channel: (
                bundle.source_endpoint_hour[
                    :,
                    i,
                ].tolist()
            )
            for i, channel in enumerate(
                CHANNEL_ORDER
            )
        },
        "alignment": (
            "causal last-completed cumulative endpoint; "
            "no backward copying and no future endpoint use"
        ),
        "static_demographics": (
            "repeated across all aligned hours and channels"
        ),
        "patients": int(
            bundle.X.shape[0]
        ),
        "tensor_nan_fraction": float(
            np.isnan(
                bundle.X
            ).mean()
        ),
        "los_remaining_mean_days": float(
            np.mean(
                bundle.y_los_remaining_days
            )
        ),
        "los_remaining_median_days": float(
            np.median(
                bundle.y_los_remaining_days
            )
        ),
        "mortality_known_patients": int(
            known.sum()
        ),
        "mortality_missing_patients": int(
            (~known).sum()
        ),
        "mortality_events_among_known": int(
            np.nansum(
                bundle.y_mortality
            )
        ),
        "mortality_prevalence_among_known": (
            float(
                np.nanmean(
                    bundle.y_mortality
                )
            )
            if known.any()
            else None
        ),
        "feature_names": (
            bundle.feature_schema.feature_names
        ),
        "npz_file": str(
            npz_path
        ),
    }

    json_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return npz_path, json_path


def diagnostic_row(
    bundle: TensorBundle,
    *,
    database: str,
    split_name: str,
    landmark: int,
    npz_path: Path,
) -> dict:
    clinical = bundle.X[
        :,
        :,
        :bundle.feature_schema.clinical_count,
        :,
    ]

    demographic = bundle.X[
        :,
        :,
        bundle.feature_schema.clinical_count:,
        :,
    ]

    known = bundle.mortality_known

    return {
        "database": database,
        "split": split_name,
        "landmark_hour": int(
            landmark
        ),
        "patients": int(
            bundle.X.shape[0]
        ),
        "time_steps": int(
            bundle.X.shape[1]
        ),
        "features": int(
            bundle.X.shape[2]
        ),
        "channels": int(
            bundle.X.shape[3]
        ),
        "tensor_shape": str(
            tuple(
                bundle.X.shape
            )
        ),
        "tensor_nan_fraction": float(
            np.isnan(
                bundle.X
            ).mean()
        ),
        "clinical_nan_fraction": float(
            np.isnan(
                clinical
            ).mean()
        ),
        "demographic_nan_fraction": float(
            np.isnan(
                demographic
            ).mean()
        ),
        "remaining_los_mean_days": float(
            np.mean(
                bundle.y_los_remaining_days
            )
        ),
        "remaining_los_median_days": float(
            np.median(
                bundle.y_los_remaining_days
            )
        ),
        "remaining_los_std_days": float(
            np.std(
                bundle.y_los_remaining_days
            )
        ),
        "mortality_known_patients": int(
            known.sum()
        ),
        "mortality_missing_patients": int(
            (~known).sum()
        ),
        "mortality_events_among_known": int(
            np.nansum(
                bundle.y_mortality
            )
        ),
        "mortality_prevalence_among_known": (
            float(
                np.nanmean(
                    bundle.y_mortality
                )
            )
            if known.any()
            else np.nan
        ),
        "npz_file": str(
            npz_path
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build causal multi-landmark 4-channel tensor cubes from "
            "Stage-05 leakage-safe native cumulative summaries."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    parser.add_argument(
        "--stage05-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--landmarks",
        type=str,
        default=None,
        help=(
            "Optional comma-separated subset. Default: all landmarks "
            "listed in the Stage-05 manifest."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=DEFAULT_ATOL,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=DEFAULT_RTOL,
    )
    parser.add_argument(
        "--uncompressed",
        action="store_true",
        help=(
            "Use np.savez instead of np.savez_compressed. Faster but "
            "substantially larger on disk."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    stage05_dir = (
        args.stage05_dir
        .expanduser()
        .resolve()
        if args.stage05_dir is not None
        else project_root
        / "data"
        / "05_splits"
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else project_root
        / "data"
        / "06_numpy_cubes"
    )

    logger = configure_logging(
        output_dir
    )

    stage05_manifest_path = (
        stage05_dir
        / "05_multilandmark_manifest.json"
    )
    stage05_manifest = load_json(
        stage05_manifest_path
    )

    landmarks = parse_landmarks(
        args.landmarks,
        stage05_manifest,
    )

    schema = load_feature_schema(
        stage05_dir,
        stage05_manifest,
    )

    logger.info(
        "Stage 05: %s",
        stage05_dir,
    )
    logger.info(
        "Output: %s",
        output_dir,
    )
    logger.info(
        "Landmarks: %s",
        landmarks,
    )
    logger.info(
        "Frozen feature schema | clinical=%d demographic=%d total=%d",
        schema.clinical_count,
        schema.demographic_count,
        schema.feature_count,
    )
    logger.info(
        "Alignment: causal last-completed endpoint; static demographics "
        "repeated across time/channels."
    )

    reports_dir = (
        output_dir
        / "reports"
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_rows: List[dict] = []
    alignment_rows: List[dict] = []
    demographic_rows: List[dict] = []
    target_rows: List[dict] = []
    consistency_rows: List[dict] = []
    variant_rows: List[dict] = []

    output_records: List[dict] = []

    for landmark in landmarks:
        logger.info(
            "===== Landmark %dh =====",
            landmark,
        )

        variant_row_for_landmark = None

        for database, split_name in SPLIT_SPECS:
            logger.info(
                "Building %dh %s/%s",
                landmark,
                database,
                split_name,
            )

            (
                bundle,
                split_alignment_rows,
                split_demographic_rows,
                split_target_rows,
            ) = build_split_tensor(
                stage05_dir=stage05_dir,
                stage05_manifest=stage05_manifest,
                schema=schema,
                landmark=landmark,
                database=database,
                split_name=split_name,
                atol=args.atol,
                rtol=args.rtol,
            )

            landmark_output_dir = (
                output_dir
                / f"landmark_{landmark:03d}h"
            )

            npz_path, json_path = save_bundle(
                bundle,
                output_dir=landmark_output_dir,
                database=database,
                split_name=split_name,
                landmark=landmark,
                compressed=not args.uncompressed,
            )

            diagnostics_rows.append(
                diagnostic_row(
                    bundle,
                    database=database,
                    split_name=split_name,
                    landmark=landmark,
                    npz_path=npz_path,
                )
            )

            alignment_rows.extend(
                split_alignment_rows
            )
            demographic_rows.extend(
                split_demographic_rows
            )
            target_rows.extend(
                split_target_rows
            )

            split_consistency = (
                final_channel_consistency_rows(
                    bundle=bundle,
                    landmark=landmark,
                    database=database,
                    split_name=split_name,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            consistency_rows.extend(
                split_consistency
            )

            # Every channel whose final endpoint is the landmark must be
            # identical to o1 because all are cumulative [0,t] summaries.
            failed_comparable = [
                row
                for row in split_consistency
                if (
                    row[
                        "comparable_same_landmark_endpoint"
                    ]
                    and not row[
                        "all_equal_with_nan"
                    ]
                )
            ]
            if failed_comparable:
                raise RuntimeError(
                    f"{database}/{split_name}/{landmark}h: final "
                    "same-endpoint channel consistency FAILED."
                )

            if variant_row_for_landmark is None:
                variant_row_for_landmark = (
                    build_variant_availability_row(
                        bundle,
                        landmark=landmark,
                    )
                )

            output_records.append(
                {
                    "landmark_hour": int(
                        landmark
                    ),
                    "database": database,
                    "split": split_name,
                    "npz": str(
                        npz_path
                    ),
                    "metadata_json": str(
                        json_path
                    ),
                    "shape": list(
                        bundle.X.shape
                    ),
                }
            )

            logger.info(
                "%dh %s/%s | shape=%s | NaN=%.2f%% | "
                "RLOS mean=%.3fd | mortality known=%d/%d",
                landmark,
                database,
                split_name,
                bundle.X.shape,
                float(
                    np.isnan(
                        bundle.X
                    ).mean()
                    * 100.0
                ),
                float(
                    np.mean(
                        bundle.y_los_remaining_days
                    )
                ),
                int(
                    bundle.mortality_known.sum()
                ),
                int(
                    len(
                        bundle.mortality_known
                    )
                ),
            )

            # Release the largest object before the next split.
            del bundle

        assert variant_row_for_landmark is not None
        variant_rows.append(
            variant_row_for_landmark
        )

    diagnostics_df = pd.DataFrame(
        diagnostics_rows
    )
    diagnostics_df.to_csv(
        reports_dir
        / "tensor_diagnostics.csv",
        index=False,
    )

    alignment_df = pd.DataFrame(
        alignment_rows
    )
    alignment_df.to_csv(
        reports_dir
        / "causal_alignment_audit.csv",
        index=False,
    )

    demographic_df = pd.DataFrame(
        demographic_rows
    )
    demographic_df.to_csv(
        reports_dir
        / "demographic_consistency_audit.csv",
        index=False,
    )

    target_df = pd.DataFrame(
        target_rows
    )
    target_df.to_csv(
        reports_dir
        / "target_alignment_audit.csv",
        index=False,
    )

    consistency_df = pd.DataFrame(
        consistency_rows
    )
    consistency_df.to_csv(
        reports_dir
        / "final_channel_consistency.csv",
        index=False,
    )

    variant_df = pd.DataFrame(
        variant_rows
    )
    variant_df.to_csv(
        reports_dir
        / "variant_availability.csv",
        index=False,
    )

    if not (
        alignment_df[
            "future_endpoint_violations"
        ].eq(0).all()
    ):
        raise RuntimeError(
            "Causal alignment audit contains future endpoint violations."
        )

    if not target_df[
        "target_alignment_pass"
    ].all():
        raise RuntimeError(
            "Target alignment audit failed."
        )

    if (
        demographic_df[
            "mismatched_patients"
        ].gt(0).any()
    ):
        raise RuntimeError(
            "Demographic consistency audit failed."
        )

    manifest = {
        "script": (
            "06_build_multilandmark_tensor_cubes.py"
        ),
        "input_stage": str(
            stage05_dir
        ),
        "input_manifest": str(
            stage05_manifest_path
        ),
        "output_dir": str(
            output_dir
        ),
        "landmarks_hours": [
            int(x)
            for x in landmarks
        ],
        "tensor_shape_rule": (
            "(N, landmark_hour, feature_count, 4)"
        ),
        "feature_count": int(
            schema.feature_count
        ),
        "clinical_feature_count": int(
            schema.clinical_count
        ),
        "demographic_feature_count": int(
            schema.demographic_count
        ),
        "channel_order": CHANNEL_ORDER,
        "resolution_hours": (
            RESOLUTION_HOURS
        ),
        "alignment_policy": (
            "causal last-completed cumulative endpoint: aligned hour h "
            "uses floor(h/w)*w when >=w; clinical values are never copied "
            "backward and future native endpoints are forbidden"
        ),
        "demographic_policy": (
            "time-invariant demographics are validated as constant and "
            "repeated across all aligned hours and resolution channels"
        ),
        "database_metadata_policy": (
            "source database labels are normalized to canonical mimic/eicu "
            "metadata names; feature values and patient records are unchanged"
        ),
        "clinical_missingness_preserved": True,
        "imputation_performed": False,
        "scaling_performed": False,
        "los_primary_target": (
            "remaining ICU LOS after landmark, days"
        ),
        "mortality_missingness_policy": (
            "unknown mortality labels retained as NaN with "
            "mortality_known=False"
        ),
        "npz_compressed": bool(
            not args.uncompressed
        ),
        "floating_dtype": "float32",
        "integer_time_dtype": "int16",
        "strict_checks": {
            "feature_order": True,
            "patient_target_alignment": True,
            "native_endpoint_grid": True,
            "future_endpoint_violations_required_zero": True,
            "demographic_constancy": True,
            "same_endpoint_final_channel_consistency": True,
        },
        "reports": {
            "tensor_diagnostics": str(
                reports_dir
                / "tensor_diagnostics.csv"
            ),
            "causal_alignment_audit": str(
                reports_dir
                / "causal_alignment_audit.csv"
            ),
            "demographic_consistency_audit": str(
                reports_dir
                / "demographic_consistency_audit.csv"
            ),
            "target_alignment_audit": str(
                reports_dir
                / "target_alignment_audit.csv"
            ),
            "final_channel_consistency": str(
                reports_dir
                / "final_channel_consistency.csv"
            ),
            "variant_availability": str(
                reports_dir
                / "variant_availability.csv"
            ),
        },
        "outputs": output_records,
    }

    manifest_path = (
        output_dir
        / "06_tensor_manifest.json"
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
        "Stage 06 complete | landmarks=%s | feature_count=%d",
        landmarks,
        schema.feature_count,
    )
    logger.info(
        "Causal alignment audit: PASS (future endpoint violations = 0)"
    )
    logger.info(
        "Manifest: %s",
        manifest_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
