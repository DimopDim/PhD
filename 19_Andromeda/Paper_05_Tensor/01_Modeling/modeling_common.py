#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modeling_common.py

Shared utilities for the multi-landmark XGBoost pipeline.

Design fixed upstream
---------------------
- Stage 06 tensors: (N, t, 337, 4)
- 304 clinical predictors + 33 demographics
- channels o1/o2/o3/o4 = 1h/2h/3h/4h
- primary LOS target: remaining ICU LOS after landmark
- mortality: in-hospital mortality; unknown external labels retained as NaN
- MIMIC development/test split fixed upstream
- no external data used for fitting/model selection/calibration/thresholding

Model representations
---------------------
full:
    six trajectory descriptors for each of the 304 clinical features in all
    four channels, plus the 33 demographics once:
        304 * 6 * 4 + 33 = 7329

o1/o2/o3/o4:
    six descriptors for one channel + demographics once:
        304 * 6 + 33 = 1857

static:
    final o1 landmark summary, including demographics once:
        304 + 33 = 337

Trajectory descriptors:
    last, mean, std, min, max, slope

No imputation or scaling is performed. XGBoost handles missing values natively.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CHANNEL_ORDER = ["o1", "o2", "o3", "o4"]
CHANNEL_TO_INDEX = {
    name: i for i, name in enumerate(CHANNEL_ORDER)
}
TRAJECTORY_DESCRIPTORS = [
    "last",
    "mean",
    "std",
    "min",
    "max",
    "slope",
]
VALID_VARIANTS = [
    "full",
    "o1",
    "o2",
    "o3",
    "o4",
    "static",
]
VALID_OUTCOMES = [
    "los",
    "mortality",
]


# Identifiers / administrative timing variables that must never enter X.
# Checks are case-insensitive and examine both exact names and component tokens.
FORBIDDEN_PREDICTOR_EXACT = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
    "window_index",
    "window_start_hour",
    "window_end_hour",
    "window_label",
    "landmark_hour",
    "icu_los",
    "los_total_days",
    "los_remaining_days",
    "hospital_expire_flag",
    "mortality_known",
    "database",
    "split",
}

FORBIDDEN_PREDICTOR_TOKENS = {
    "subject_id",
    "hadm_id",
    "stay_id",
    "patient_id",
    "patientunitstayid",
    "patienthealthsystemstayid",
    "uniquepid",
}


@dataclass
class TensorData:
    X: np.ndarray
    y_los_remaining_days: np.ndarray
    y_los_total_days: np.ndarray
    y_mortality: np.ndarray
    mortality_known: np.ndarray
    patient_id: np.ndarray
    stay_id: np.ndarray
    time_hours: np.ndarray
    channel_available: np.ndarray
    source_endpoint_hour: np.ndarray
    feature_names: List[str]
    clinical_count: int
    demographic_count: int


@dataclass
class ModelMatrix:
    X: np.ndarray
    feature_names: List[str]
    patient_id: np.ndarray
    stay_id: np.ndarray
    y_los_remaining_days: np.ndarray
    y_los_total_days: np.ndarray
    y_mortality: np.ndarray
    mortality_known: np.ndarray


def assert_no_identifier_predictors(
    feature_names: Sequence[str],
) -> None:
    """
    Fail closed if any patient/stay/admission identifier or target/admin field
    appears in the model feature list.

    `patient_id` and `stay_id` remain metadata arrays only and are used for
    fold grouping/auditing. They are never part of X.
    """
    violations: List[str] = []

    for raw_name in feature_names:
        name = str(raw_name).strip().lower()

        if name in FORBIDDEN_PREDICTOR_EXACT:
            violations.append(str(raw_name))
            continue

        # Model features may have prefixes such as static__, o1__, etc.
        components = set(
            re.split(r"__|[^a-z0-9_]+", name)
        )
        components.discard("")

        if any(
            token in name
            for token in FORBIDDEN_PREDICTOR_TOKENS
        ):
            violations.append(str(raw_name))
            continue

        # Extra guard against common identifier suffix/prefix patterns.
        if (
            name.endswith("_id")
            or "__id" in name
            or name.startswith("id__")
        ):
            violations.append(str(raw_name))
            continue

    if violations:
        raise ValueError(
            "Identifier/administrative variables detected in predictors: "
            f"{sorted(set(violations))[:20]}"
        )


def assert_unique_patient_rows(
    patient_id: np.ndarray,
    *,
    label: str,
) -> None:
    ids = np.asarray(patient_id).astype(str)
    unique, counts = np.unique(
        ids,
        return_counts=True,
    )
    duplicated = unique[
        counts > 1
    ]
    if len(duplicated):
        raise ValueError(
            f"{label}: expected one model row per patient, but found "
            f"{len(duplicated)} duplicated patient_id values. "
            f"Examples={duplicated[:10].tolist()}"
        )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_arg(text: str, valid: Sequence[str]) -> List[str]:
    values = [
        x.strip()
        for x in text.split(",")
        if x.strip()
    ]
    invalid = sorted(
        set(values) - set(valid)
    )
    if invalid:
        raise ValueError(
            f"Invalid values {invalid}; valid={list(valid)}"
        )
    return values


def parse_landmarks(
    text: Optional[str],
    manifest: dict,
) -> List[int]:
    available = [
        int(x)
        for x in manifest["landmarks_hours"]
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
    missing = sorted(
        set(requested) - set(available)
    )
    if missing:
        raise ValueError(
            f"Landmarks absent from tensor manifest: {missing}. "
            f"Available={available}"
        )
    return requested


def tensor_paths(
    tensor_root: Path,
    landmark: int,
    database: str,
    split_name: str,
) -> Tuple[Path, Path]:
    stem = f"{database}_{split_name}"
    directory = (
        tensor_root
        / f"landmark_{landmark:03d}h"
    )
    return (
        directory / f"{stem}.npz",
        directory / f"{stem}.json",
    )


def load_tensor(
    tensor_root: Path,
    landmark: int,
    database: str,
    split_name: str,
) -> TensorData:
    npz_path, metadata_path = tensor_paths(
        tensor_root,
        landmark,
        database,
        split_name,
    )

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    meta = load_json(metadata_path)

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as z:
        X = z["X"].astype(
            np.float32,
            copy=False,
        )
        y_los_remaining_days = z[
            "y_los_remaining_days"
        ].astype(np.float32, copy=False)
        y_los_total_days = z[
            "y_los_total_days"
        ].astype(np.float32, copy=False)
        y_mortality = z[
            "y_mortality"
        ].astype(np.float32, copy=False)
        mortality_known = z[
            "mortality_known"
        ].astype(bool, copy=False)
        patient_id = z[
            "patient_id"
        ].astype(str)
        stay_id = z[
            "stay_id"
        ].astype(str)
        time_hours = z[
            "time_hours"
        ].astype(np.float64)
        channel_available = z[
            "channel_available"
        ].astype(bool)
        source_endpoint_hour = z[
            "source_endpoint_hour"
        ].astype(np.int16)

    feature_names = [
        str(x)
        for x in meta["feature_names"]
    ]

    clinical_count = int(
        meta["clinical_feature_count"]
    )
    demographic_count = int(
        meta["demographic_feature_count"]
    )

    expected_shape = (
        len(patient_id),
        int(landmark),
        len(feature_names),
        4,
    )
    if X.shape != expected_shape:
        raise ValueError(
            f"{npz_path}: tensor shape {X.shape}, "
            f"expected {expected_shape}."
        )

    if (
        clinical_count
        + demographic_count
        != len(feature_names)
    ):
        raise ValueError(
            f"{metadata_path}: feature counts inconsistent."
        )

    return TensorData(
        X=X,
        y_los_remaining_days=y_los_remaining_days,
        y_los_total_days=y_los_total_days,
        y_mortality=y_mortality,
        mortality_known=mortality_known,
        patient_id=patient_id,
        stay_id=stay_id,
        time_hours=time_hours,
        channel_available=channel_available,
        source_endpoint_hour=source_endpoint_hour,
        feature_names=feature_names,
        clinical_count=clinical_count,
        demographic_count=demographic_count,
    )


def variant_available(
    tensor: TensorData,
    variant: str,
) -> bool:
    if variant == "static":
        return bool(
            tensor.channel_available[
                CHANNEL_TO_INDEX["o1"]
            ]
        )
    if variant == "full":
        return bool(
            tensor.channel_available.all()
        )
    if variant in CHANNEL_TO_INDEX:
        return bool(
            tensor.channel_available[
                CHANNEL_TO_INDEX[variant]
            ]
        )
    raise ValueError(variant)


def _nan_last(a: np.ndarray) -> np.ndarray:
    """
    Last finite value over time for shape (N,T,C).
    """
    n, t, c = a.shape
    finite = np.isfinite(a)
    has_any = finite.any(axis=1)

    rev_idx = np.argmax(
        finite[:, ::-1, :],
        axis=1,
    )
    last_idx = (
        t - 1 - rev_idx
    )

    rows = np.arange(n)[:, None]
    cols = np.arange(c)[None, :]
    out = a[
        rows,
        last_idx,
        cols,
    ].astype(
        np.float32,
        copy=False,
    )
    out = out.copy()
    out[
        ~has_any
    ] = np.nan
    return out


def _nan_slope(
    a: np.ndarray,
    time_hours: np.ndarray,
) -> np.ndarray:
    """
    Vectorized OLS slope for each patient-feature pair, ignoring NaNs.
    Shape input: (N,T,C); output: (N,C).

    For exactly one finite time point, slope is 0.
    For zero finite points, slope is NaN.
    """
    y = a.astype(
        np.float64,
        copy=False,
    )
    x = np.asarray(
        time_hours,
        dtype=np.float64,
    )[None, :, None]

    mask = np.isfinite(y)
    y0 = np.where(
        mask,
        y,
        0.0,
    )
    x0 = np.where(
        mask,
        x,
        0.0,
    )

    n = mask.sum(
        axis=1
    ).astype(
        np.float64
    )
    sum_x = x0.sum(
        axis=1
    )
    sum_y = y0.sum(
        axis=1
    )
    sum_xx = (
        x0 * x0
    ).sum(axis=1)
    sum_xy = (
        x0 * y0
    ).sum(axis=1)

    denom = (
        n * sum_xx
        - sum_x * sum_x
    )
    numer = (
        n * sum_xy
        - sum_x * sum_y
    )

    slope = np.full(
        n.shape,
        np.nan,
        dtype=np.float64,
    )

    one = n == 1
    slope[one] = 0.0

    valid = (
        (n >= 2)
        & (np.abs(denom) > 0)
    )
    slope[valid] = (
        numer[valid]
        / denom[valid]
    )

    degenerate = (
        (n >= 2)
        & ~valid
    )
    slope[degenerate] = 0.0

    return slope.astype(
        np.float32
    )


def clinical_descriptors(
    clinical: np.ndarray,
    time_hours: np.ndarray,
) -> np.ndarray:
    """
    Input: (N,T,C)
    Output: (N,C,6) in TRAJECTORY_DESCRIPTORS order.
    """
    if clinical.ndim != 3:
        raise ValueError(
            f"Expected clinical array (N,T,C), got {clinical.shape}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            category=RuntimeWarning,
        )
        last = _nan_last(
            clinical
        )
        mean = np.nanmean(
            clinical,
            axis=1,
        ).astype(np.float32)
        std = np.nanstd(
            clinical,
            axis=1,
            ddof=0,
        ).astype(np.float32)
        minimum = np.nanmin(
            clinical,
            axis=1,
        ).astype(np.float32)
        maximum = np.nanmax(
            clinical,
            axis=1,
        ).astype(np.float32)

    finite_count = np.isfinite(
        clinical
    ).sum(axis=1)

    # np.nanstd already gives 0 for one observation; make this explicit.
    std[
        finite_count == 1
    ] = 0.0
    std[
        finite_count == 0
    ] = np.nan

    slope = _nan_slope(
        clinical,
        time_hours,
    )

    return np.stack(
        [
            last,
            mean,
            std,
            minimum,
            maximum,
            slope,
        ],
        axis=-1,
    ).astype(
        np.float32,
        copy=False,
    )


def descriptor_feature_names(
    clinical_names: Sequence[str],
    channels: Sequence[str],
) -> List[str]:
    names: List[str] = []

    # Flatten order follows array reshape:
    # channel-major -> clinical-feature-major -> descriptor-major.
    for channel in channels:
        for base in clinical_names:
            for descriptor in TRAJECTORY_DESCRIPTORS:
                names.append(
                    f"{channel}__{base}__traj_{descriptor}"
                )

    return names


def build_model_matrix(
    tensor: TensorData,
    variant: str,
) -> ModelMatrix:
    if variant not in VALID_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant}"
        )

    if not variant_available(
        tensor,
        variant,
    ):
        raise ValueError(
            f"Variant {variant} is unavailable for this landmark."
        )

    c = tensor.clinical_count
    clinical_names = (
        tensor.feature_names[:c]
    )
    demographic_names = (
        tensor.feature_names[c:]
    )

    # Canonical static demographics: final o1 position.
    demographics = tensor.X[
        :,
        -1,
        c:,
        CHANNEL_TO_INDEX["o1"],
    ].astype(
        np.float32,
        copy=False,
    )

    if variant == "static":
        X = tensor.X[
            :,
            -1,
            :,
            CHANNEL_TO_INDEX["o1"],
        ].astype(
            np.float32,
            copy=False,
        )
        feature_names = [
            f"static__{name}"
            for name in tensor.feature_names
        ]
    else:
        channels = (
            CHANNEL_ORDER
            if variant == "full"
            else [variant]
        )

        blocks: List[np.ndarray] = []

        for channel in channels:
            channel_index = (
                CHANNEL_TO_INDEX[channel]
            )
            clinical = tensor.X[
                :,
                :,
                :c,
                channel_index,
            ]

            desc = clinical_descriptors(
                clinical,
                tensor.time_hours,
            )

            # (N,C,6) -> (N,C*6)
            blocks.append(
                desc.reshape(
                    desc.shape[0],
                    -1,
                )
            )

        X = np.concatenate(
            blocks + [demographics],
            axis=1,
        ).astype(
            np.float32,
            copy=False,
        )

        feature_names = (
            descriptor_feature_names(
                clinical_names,
                channels,
            )
            + list(
                demographic_names
            )
        )

    if X.shape[1] != len(
        feature_names
    ):
        raise AssertionError(
            f"Feature-name width mismatch: X={X.shape}, "
            f"names={len(feature_names)}"
        )

    assert_no_identifier_predictors(
        feature_names
    )
    assert_unique_patient_rows(
        tensor.patient_id,
        label=f"model matrix variant={variant}",
    )

    return ModelMatrix(
        X=X,
        feature_names=feature_names,
        patient_id=tensor.patient_id,
        stay_id=tensor.stay_id,
        y_los_remaining_days=(
            tensor.y_los_remaining_days
        ),
        y_los_total_days=(
            tensor.y_los_total_days
        ),
        y_mortality=(
            tensor.y_mortality
        ),
        mortality_known=(
            tensor.mortality_known
        ),
    )


def save_model_matrix(
    matrix: ModelMatrix,
    output_path: Path,
    feature_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.savez_compressed(
        output_path,
        X=matrix.X,
        patient_id=matrix.patient_id,
        stay_id=matrix.stay_id,
        y_los_remaining_days=(
            matrix.y_los_remaining_days
        ),
        y_los_total_days=(
            matrix.y_los_total_days
        ),
        y_mortality=matrix.y_mortality,
        mortality_known=(
            matrix.mortality_known
        ),
    )
    feature_path.write_text(
        json.dumps(
            {
                "feature_names": matrix.feature_names,
                "feature_count": len(
                    matrix.feature_names
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_model_matrix(
    matrix_path: Path,
    feature_path: Path,
) -> ModelMatrix:
    meta = load_json(feature_path)
    with np.load(
        matrix_path,
        allow_pickle=False,
    ) as z:
        X = z["X"].astype(
            np.float32,
            copy=False,
        )
        patient_id = z[
            "patient_id"
        ].astype(str)
        stay_id = z[
            "stay_id"
        ].astype(str)
        y_los_remaining_days = z[
            "y_los_remaining_days"
        ].astype(np.float32)
        y_los_total_days = z[
            "y_los_total_days"
        ].astype(np.float32)
        y_mortality = z[
            "y_mortality"
        ].astype(np.float32)
        mortality_known = z[
            "mortality_known"
        ].astype(bool)

    feature_names = [
        str(x)
        for x in meta["feature_names"]
    ]
    if X.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            f"{matrix_path}: feature width mismatch."
        )

    assert_no_identifier_predictors(
        feature_names
    )
    assert_unique_patient_rows(
        patient_id,
        label=str(matrix_path),
    )

    return ModelMatrix(
        X=X,
        feature_names=feature_names,
        patient_id=patient_id,
        stay_id=stay_id,
        y_los_remaining_days=(
            y_los_remaining_days
        ),
        y_los_total_days=(
            y_los_total_days
        ),
        y_mortality=y_mortality,
        mortality_known=mortality_known,
    )


def matrix_cache_paths(
    modeling_root: Path,
    landmark: int,
    variant: str,
    database: str,
    split_name: str,
) -> Tuple[Path, Path]:
    directory = (
        modeling_root
        / "data"
        / "model_matrices"
        / f"landmark_{landmark:03d}h"
        / variant
    )
    stem = f"{database}_{split_name}"
    return (
        directory / f"{stem}.npz",
        directory / "feature_names.json",
    )


def task_dir(
    modeling_root: Path,
    landmark: int,
    outcome: str,
    variant: str,
) -> Path:
    return (
        modeling_root
        / "results"
        / f"landmark_{landmark:03d}h"
        / outcome
        / variant
    )


def concept_from_model_feature(
    feature_name: str,
) -> str:
    """
    Collapse channels, temporal descriptors, and within-window aggregators to
    one clinical concept for grouped SHAP.

    Demographics:
        age -> Age
        gender__* -> Gender
        race__* -> Race
    """
    name = str(feature_name)

    if name.startswith("gender__"):
        return "Gender"
    if name.startswith("race__"):
        return "Race"
    if (
        name == "age"
        or name.endswith("__age")
        or name == "static__age"
    ):
        return "Age"

    # Remove representation prefix.
    for prefix in [
        "static__",
        "o1__",
        "o2__",
        "o3__",
        "o4__",
    ]:
        if name.startswith(prefix):
            name = name[
                len(prefix):
            ]
            break

    # Remove temporal descriptor.
    name = re_sub_traj(
        name
    )

    # Remove within-window aggregation suffix.
    for suffix in [
        "__mean",
        "__median",
        "__min",
        "__max",
    ]:
        if name.endswith(
            suffix
        ):
            name = name[
                :-len(suffix)
            ]
            break

    return name


def re_sub_traj(name: str) -> str:
    marker = "__traj_"
    if marker in name:
        return name.split(
            marker,
            1,
        )[0]
    return name


def group_feature_indices(
    feature_names: Sequence[str],
) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    for i, feature in enumerate(
        feature_names
    ):
        concept = concept_from_model_feature(
            feature
        )
        groups.setdefault(
            concept,
            [],
        ).append(i)
    return groups
