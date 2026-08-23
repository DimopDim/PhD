#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_create_splits_and_encode.py

Stage 05 — fixed patient split + dynamic landmark risk sets + frozen
demographic encoding.

Primary methodological design
-----------------------------
The patient must still be in the ICU at the prediction landmark t.

Default landmarks:
    1 h, 4 h, 8 h, 12 h, 16 h, 20 h, 24 h, 36 h, 48 h

For each landmark t:
    eligibility: total ICU LOS > t
    observation: only Stage-04 cumulative endpoints <= t
    LOS target: remaining ICU LOS = total ICU LOS - t
    mortality target: subsequent in-hospital mortality

The MIMIC patient-level 80/20 development/test split is created ONCE on the full selected cohort.
The same patient assignment is then reused at every landmark. The eICU cohort
is never used for split creation or demographic-schema fitting.

Leakage barriers
----------------
- Patients with ICU LOS <= t are excluded from the risk set at landmark t.
- No row with endpoint_hour > t enters the landmark dataset.
- No structurally unavailable row enters a landmark dataset.
- ICU LOS, remaining LOS, hospital mortality, and window_available never enter
  model-ready predictor files.
- Targets are written separately from predictors.
- Clinical values are not imputed or scaled here.
- Demographic categorical vocabulary is frozen from the MIMIC training split
  and applied unchanged to validation, test, and eICU.

Feature dimensionality
----------------------
    304 Stage-04 clinical descriptors
    + 1 age
    + 3 gender columns
    + R race columns observed in MIMIC training
    = 304 + 1 + 3 + R predictors per native temporal endpoint

Temporal-grid note
------------------
A landmark need not be divisible by every native resolution. For example,
at t=6h the available completed native endpoints are:

    o1 (1h): 1,2,3,4,5,6
    o2 (2h): 2,4,6
    o3 (3h): 3,6
    o4 (4h): 4

No future coarse endpoint is used. Thus o4 at t=6h uses information only
through hour 4. This is leakage-safe. Stage 06 must preserve this rule when
aligning channels; it must never use an 8h o4 summary for a 6h landmark.

Inputs
------
data/03_harmonized/
    mimic/harmonized_demographics.parquet
    eicu/harmonized_demographics.parquet

data/04_cumulative_windows/
    mimic/o1_01h.parquet ... o4_04h.parquet
    eicu/o1_01h.parquet ... o4_04h.parquet
    feature_order.json

Outputs
-------
data/05_splits/
    demographic_schema.json
    feature_order.json
    05_multilandmark_manifest.json
    05_create_splits_and_encode.log

    reports/
        full_cohort_split_assignments.csv
        demographic_category_counts.csv
        demographic_encoding_audit.csv
        feature_order.csv
        landmark_risk_set_summary.csv
        landmark_grid_summary.csv
        nested_risk_set_audit.csv
        landmark_split_assignments.csv

    landmark_006h/
        mimic/train/o1_01h.parquet ... o4_04h.parquet
        mimic/validate/...
        mimic/test/...
        eicu/external/...
        targets/...

    landmark_012h/
    landmark_024h/
    landmark_048h/

Example
-------
cd /home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor

python 05_create_splits_and_encode.py \
    --landmarks 1,4,8,12,16,20,24,36,48 \
    --seed 42 \
    --train-fraction 0.80 \
    --test-fraction 0.20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor"
)
HARMONIZED_DIR_DEFAULT = (
    PROJECT_ROOT_DEFAULT / "data" / "03_harmonized"
)
WINDOW_DIR_DEFAULT = (
    PROJECT_ROOT_DEFAULT / "data" / "04_cumulative_windows"
)
OUTPUT_DIR_DEFAULT = (
    PROJECT_ROOT_DEFAULT / "data" / "05_splits"
)

DEFAULT_LANDMARKS = [1, 4, 8, 12, 16, 20, 24, 36, 48]

SEED_DEFAULT = 42
TRAIN_FRACTION_DEFAULT = 0.80
TEST_FRACTION_DEFAULT = 0.20

EXPECTED_CLINICAL_COLUMNS = 304
EXPECTED_GENDER_COLUMNS = 3

EXPECTED_GENDER_CATEGORIES = ["F", "M", "UNKNOWN"]

RESOLUTIONS = {
    "o1": {
        "filename": "o1_01h.parquet",
        "hours": 1,
    },
    "o2": {
        "filename": "o2_02h.parquet",
        "hours": 2,
    },
    "o3": {
        "filename": "o3_03h.parquet",
        "hours": 3,
    },
    "o4": {
        "filename": "o4_04h.parquet",
        "hours": 4,
    },
}

MODEL_METADATA_COLUMNS = [
    "database",
    "patient_id",
    "stay_id",
    "native_resolution_hours",
    "native_step_index",
    "endpoint_hour",
    "endpoint_minutes",
]


def require_dependencies() -> None:
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Stage 05 requires pyarrow. Install with: pip install pyarrow"
        ) from exc

    try:
        import sklearn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Stage 05 requires scikit-learn. Install with: "
            "pip install scikit-learn"
        ) from exc


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("stage05_multilandmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(
        output_dir / "05_create_splits_and_encode.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def parse_landmarks(value: str) -> List[int]:
    try:
        landmarks = sorted(
            {
                int(x.strip())
                for x in value.split(",")
                if x.strip()
            }
        )
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            "Landmarks must be comma-separated integers, e.g. 6,12,24,48"
        ) from exc

    if not landmarks:
        raise argparse.ArgumentTypeError(
            "At least one landmark is required."
        )

    if landmarks[0] <= 0:
        raise argparse.ArgumentTypeError(
            "Landmarks must be >0 hours."
        )

    if landmarks[-1] > 48:
        raise argparse.ArgumentTypeError(
            "Stage 04 currently contains only the first 48 ICU hours."
        )

    return landmarks


def normalize_gender(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"

    text = str(value).strip().upper()

    mapping = {
        "F": "F",
        "FEMALE": "F",
        "M": "M",
        "MALE": "M",
        "UNKNOWN": "UNKNOWN",
        "UNK": "UNKNOWN",
        "": "UNKNOWN",
    }
    return mapping.get(text, "UNKNOWN")


def normalize_race(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"

    text = re.sub(
        r"\s+",
        " ",
        str(value).strip().upper(),
    )

    if text in {
        "",
        "NAN",
        "NONE",
        "NULL",
        "UNABLE TO OBTAIN",
        "PATIENT DECLINED TO ANSWER",
    }:
        return "UNKNOWN"

    return text


def safe_slug(text: str) -> str:
    slug = re.sub(
        r"[^A-Z0-9]+",
        "_",
        text.upper(),
    ).strip("_")
    return slug or "UNKNOWN"


def validate_split_fractions(
    train_fraction: float,
    test_fraction: float,
) -> None:
    if train_fraction <= 0 or train_fraction >= 1:
        raise ValueError("--train-fraction must be inside (0,1).")
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("--test-fraction must be inside (0,1).")

    if not math.isclose(
        train_fraction + test_fraction,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "train + test fractions must sum to 1."
        )


def load_feature_order(window_dir: Path) -> List[str]:
    path = window_dir / "feature_order.json"
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    columns = payload.get("feature_columns")

    if not isinstance(columns, list):
        raise ValueError(
            f"{path}: feature_columns missing or invalid."
        )

    columns = [str(x) for x in columns]

    if len(columns) != EXPECTED_CLINICAL_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_CLINICAL_COLUMNS} clinical columns, "
            f"found {len(columns)}."
        )

    if len(set(columns)) != len(columns):
        raise ValueError(
            "Duplicate clinical feature names."
        )

    return columns


def load_demographics(
    path: Path,
    database: str,
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    required = {
        "patient_id",
        "stay_id",
        "age",
        "gender",
        "race",
        "icu_los_days",
        "hospital_expire_flag",
    }
    missing = sorted(
        required - set(df.columns)
    )
    if missing:
        raise ValueError(
            f"{path}: missing columns {missing}"
        )

    out = pd.DataFrame(
        {
            "database": database,
            "patient_id": (
                df["patient_id"]
                .astype(str)
                .str.strip()
            ),
            "stay_id": (
                df["stay_id"]
                .astype(str)
                .str.strip()
            ),
            "age": pd.to_numeric(
                df["age"],
                errors="coerce",
            ),
            "gender": df["gender"].map(
                normalize_gender
            ),
            "race": df["race"].map(
                normalize_race
            ),
            "icu_los_days": pd.to_numeric(
                df["icu_los_days"],
                errors="coerce",
            ),
            "hospital_expire_flag": pd.to_numeric(
                df["hospital_expire_flag"],
                errors="coerce",
            ),
        }
    )

    if out["patient_id"].duplicated().any():
        raise ValueError(
            f"{database}: duplicate patient_id values."
        )

    if out["stay_id"].duplicated().any():
        raise ValueError(
            f"{database}: duplicate stay_id values."
        )

    if out["icu_los_days"].isna().any():
        n = int(
            out["icu_los_days"].isna().sum()
        )
        raise ValueError(
            f"{database}: {n} patients have missing ICU LOS."
        )

    if (out["icu_los_days"] < 0).any():
        raise ValueError(
            f"{database}: negative ICU LOS detected."
        )

    # Mortality completeness is outcome-specific.
    # MIMIC mortality must be complete because the development split is
    # stratified on mortality. eICU patients with unknown mortality remain
    # eligible for LOS and are excluded only from mortality evaluation.
    missing_mortality_n = int(
        out["hospital_expire_flag"].isna().sum()
    )

    if database == "mimic" and missing_mortality_n:
        raise ValueError(
            f"{database}: {missing_mortality_n} missing mortality labels; "
            "MIMIC mortality must be complete for split stratification."
        )

    nonmissing_mortality = out[
        "hospital_expire_flag"
    ].notna()

    invalid_mortality = (
        nonmissing_mortality
        & ~out["hospital_expire_flag"].isin([0, 1])
    )

    if invalid_mortality.any():
        values = sorted(
            out.loc[
                invalid_mortality,
                "hospital_expire_flag",
            ].unique().tolist()
        )
        raise ValueError(
            f"{database}: non-binary mortality labels: {values}"
        )

    out["hospital_expire_flag"] = out[
        "hospital_expire_flag"
    ].astype("Int8")
    out["mortality_known"] = out[
        "hospital_expire_flag"
    ].notna()

    out["icu_los_hours"] = (
        out["icu_los_days"].astype("float64")
        * 24.0
    )

    return out.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(drop=True)


def create_full_mimic_split(
    demo: pd.DataFrame,
    *,
    train_fraction: float,
    test_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """
    Create one patient-level development/test split on the full MIMIC cohort.

    Default:
        80% development/training
        20% internal test

    Five-fold cross-validation is performed later inside the development set.
    Stratification is by in-hospital mortality only. LOS is not used to assign
    patients to train/test.
    """
    from sklearn.model_selection import train_test_split

    if demo["hospital_expire_flag"].isna().any():
        raise RuntimeError(
            "MIMIC split cannot be stratified with missing mortality labels."
        )

    ids = demo["patient_id"].to_numpy()
    y = (
        demo["hospital_expire_flag"]
        .astype("int8")
        .to_numpy(dtype=int)
    )

    train_ids, test_ids = train_test_split(
        ids,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )

    split_map: Dict[str, str] = {}
    split_map.update({str(pid): "train" for pid in train_ids})
    split_map.update({str(pid): "test" for pid in test_ids})

    if len(split_map) != len(demo):
        raise AssertionError(
            "Every MIMIC patient must receive exactly one split."
        )

    out = demo[
        [
            "patient_id",
            "stay_id",
            "icu_los_days",
            "icu_los_hours",
            "hospital_expire_flag",
        ]
    ].copy()

    out["split"] = out["patient_id"].map(split_map)

    if out["split"].isna().any():
        raise AssertionError(
            "Missing MIMIC split assignment."
        )

    return out.sort_values(
        ["split", "patient_id"],
        kind="stable",
    ).reset_index(drop=True)


def freeze_demographic_schema(
    mimic_demo: pd.DataFrame,
    full_assignment: pd.DataFrame,
    logger: logging.Logger,
) -> dict:
    """
    Freeze the demographic encoder using only the MIMIC training partition.

    Race vocabulary is learned strictly from the MIMIC training partition.
    Test and eICU categories are never used to define the vocabulary, and no
    historical zero-only race columns are padded to preserve an old width.
    """
    train_ids = set(
        full_assignment.loc[
            full_assignment["split"].eq("train"),
            "patient_id",
        ].astype(str)
    )

    train = mimic_demo[
        mimic_demo["patient_id"].isin(
            train_ids
        )
    ].copy()

    if len(train) != len(train_ids):
        raise RuntimeError(
            "Training demographic alignment failed."
        )

    gender_categories = list(
        EXPECTED_GENDER_CATEGORIES
    )

    observed_gender = set(
        train["gender"].unique()
    )
    unknown_gender = (
        observed_gender
        - set(gender_categories)
    )
    if unknown_gender:
        raise RuntimeError(
            f"Unexpected normalized genders: {sorted(unknown_gender)}"
        )

    race_categories = sorted(
        train["race"].unique().tolist()
    )

    if "UNKNOWN" not in race_categories:
        race_categories = sorted(
            race_categories + ["UNKNOWN"]
        )

    race_column_map = {}
    used = set()

    for category in race_categories:
        column = (
            "race__"
            + safe_slug(category)
        )
        if column in used:
            raise RuntimeError(
                "Race-column slug collision for "
                f"{category!r}: {column}"
            )
        used.add(column)
        race_column_map[
            category
        ] = column

    schema = {
        "fit_source": (
            "MIMIC full-cohort training partition only"
        ),
        "fit_uses_validation": False,
        "fit_uses_test": False,
        "fit_uses_eicu": False,
        "age_column": "age",
        "gender_categories": gender_categories,
        "gender_columns": [
            "gender__F",
            "gender__M",
            "gender__UNKNOWN",
        ],
        "race_categories": race_categories,
        "race_column_map": race_column_map,
        "race_columns": [
            race_column_map[c]
            for c in race_categories
        ],
        "demographic_feature_count": (
            1
            + len(gender_categories)
            + len(race_categories)
        ),
    }

    expected_demographic_count = (
        1 + len(gender_categories) + len(race_categories)
    )
    if schema["demographic_feature_count"] != expected_demographic_count:
        raise AssertionError(
            "Dynamic demographic feature-count mismatch."
        )

    logger.info(
        "Frozen demographic schema | age=1 gender=%d race=%d total=%d",
        len(gender_categories),
        len(race_categories),
        schema["demographic_feature_count"],
    )
    logger.info(
        "Per-endpoint predictors | clinical=%d demographic=%d total=%d",
        EXPECTED_CLINICAL_COLUMNS,
        schema["demographic_feature_count"],
        EXPECTED_CLINICAL_COLUMNS + schema["demographic_feature_count"],
    )

    return schema


def encode_demographics(
    demo: pd.DataFrame,
    *,
    schema: dict,
    database: str,
    split_name: str,
    landmark_hour: int,
) -> Tuple[pd.DataFrame, dict]:
    out = pd.DataFrame(
        {
            "patient_id": (
                demo["patient_id"]
                .astype(str)
            ),
            "stay_id": (
                demo["stay_id"]
                .astype(str)
            ),
            "age": pd.to_numeric(
                demo["age"],
                errors="coerce",
            ).astype("float32"),
        }
    )

    gender = demo[
        "gender"
    ].map(normalize_gender)

    gender_unseen = ~gender.isin(
        schema["gender_categories"]
    )
    gender = gender.where(
        ~gender_unseen,
        "UNKNOWN",
    )

    for category, column in zip(
        schema["gender_categories"],
        schema["gender_columns"],
    ):
        out[column] = (
            gender.eq(category)
            .astype("float32")
        )

    race = demo[
        "race"
    ].map(normalize_race)

    race_unseen = ~race.isin(
        schema["race_categories"]
    )
    race = race.where(
        ~race_unseen,
        "UNKNOWN",
    )

    for category in schema[
        "race_categories"
    ]:
        column = schema[
            "race_column_map"
        ][category]
        out[column] = (
            race.eq(category)
            .astype("float32")
        )

    gender_cols = list(
        schema["gender_columns"]
    )
    race_cols = list(
        schema["race_columns"]
    )

    if not np.allclose(
        out[gender_cols]
        .sum(axis=1)
        .to_numpy(dtype=float),
        1.0,
    ):
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h: invalid gender one-hot."
        )

    if not np.allclose(
        out[race_cols]
        .sum(axis=1)
        .to_numpy(dtype=float),
        1.0,
    ):
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h: invalid race one-hot."
        )

    audit = {
        "database": database,
        "split": split_name,
        "landmark_hour": int(
            landmark_hour
        ),
        "patients": int(
            len(demo)
        ),
        "age_missing_patients": int(
            out["age"].isna().sum()
        ),
        "gender_unseen_mapped_to_unknown": int(
            gender_unseen.sum()
        ),
        "race_unseen_mapped_to_unknown": int(
            race_unseen.sum()
        ),
        "gender_unknown_final": int(
            gender.eq("UNKNOWN").sum()
        ),
        "race_unknown_final": int(
            race.eq("UNKNOWN").sum()
        ),
    }

    expected = (
        ["patient_id", "stay_id", "age"]
        + gender_cols
        + race_cols
    )
    out = out[expected]

    expected_demo_features = int(
        schema["demographic_feature_count"]
    )
    if len(out.columns) - 2 != expected_demo_features:
        raise AssertionError(
            f"Encoded demographics must contain {expected_demo_features} "
            "predictors."
        )

    return out, audit


def build_feature_order(
    clinical_columns: Sequence[str],
    schema: dict,
) -> Tuple[List[str], pd.DataFrame]:
    demographics = (
        [schema["age_column"]]
        + list(schema["gender_columns"])
        + list(schema["race_columns"])
    )

    features = (
        list(clinical_columns)
        + demographics
    )

    expected_base_columns = (
        len(clinical_columns)
        + int(schema["demographic_feature_count"])
    )
    if len(features) != expected_base_columns:
        raise ValueError(
            f"Expected {expected_base_columns} predictors; "
            f"found {len(features)}."
        )

    if len(set(features)) != len(features):
        raise ValueError(
            "Duplicate feature names in 341-column schema."
        )

    rows = []

    for i, name in enumerate(
        features
    ):
        if i < EXPECTED_CLINICAL_COLUMNS:
            group = "clinical"
            source = "Stage 04"
        elif name == "age":
            group = "age"
            source = "MIMIC-training frozen schema"
        elif name.startswith("gender__"):
            group = "gender"
            source = "MIMIC-training frozen schema"
        elif name.startswith("race__"):
            group = "race"
            source = "MIMIC-training frozen schema"
        else:
            group = "demographic"
            source = "MIMIC-training frozen schema"

        rows.append(
            {
                "feature_index_0based": i,
                "feature_index_1based": i + 1,
                "feature_name": name,
                "feature_group": group,
                "source": source,
            }
        )

    return features, pd.DataFrame(
        rows
    )


def risk_set(
    demo: pd.DataFrame,
    patient_ids: Sequence[str],
    landmark_hour: int,
) -> pd.DataFrame:
    """
    Patients who are still in ICU at t.

    Strict criterion:
        total ICU LOS > t
    """
    ids = set(
        map(str, patient_ids)
    )

    sub = demo[
        demo["patient_id"].isin(ids)
        & (
            demo["icu_los_hours"]
            > float(landmark_hour) + 1e-9
        )
    ].copy()

    sub["landmark_hour"] = int(
        landmark_hour
    )
    sub["remaining_los_hours"] = (
        sub["icu_los_hours"]
        - float(landmark_hour)
    )
    sub["remaining_los_days"] = (
        sub["remaining_los_hours"]
        / 24.0
    )

    if (
        sub["remaining_los_hours"] <= 0
    ).any():
        raise AssertionError(
            "Risk set contains non-positive remaining LOS."
        )

    return sub.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(drop=True)


def make_targets(
    demo: pd.DataFrame,
    *,
    database: str,
    split_name: str,
    landmark_hour: int,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "database": database,
            "split": split_name,
            "landmark_hour": int(
                landmark_hour
            ),
            "patient_id": (
                demo["patient_id"]
                .astype(str)
            ),
            "stay_id": (
                demo["stay_id"]
                .astype(str)
            ),
            "los_total_days": (
                demo["icu_los_days"]
                .astype("float64")
            ),
            "los_total_hours": (
                demo["icu_los_hours"]
                .astype("float64")
            ),
            "los_remaining_days": (
                demo["remaining_los_days"]
                .astype("float64")
            ),
            "los_remaining_hours": (
                demo["remaining_los_hours"]
                .astype("float64")
            ),
            "hospital_expire_flag": (
                demo["hospital_expire_flag"]
                .astype("Int8")
            ),
            "mortality_known": (
                demo["hospital_expire_flag"]
                .notna()
                .astype(bool)
            ),
        }
    )

    if (
        out["los_remaining_hours"] <= 0
    ).any():
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h: "
            "invalid remaining LOS target."
        )

    return out.sort_values(
        "patient_id",
        kind="stable",
    ).reset_index(drop=True)


def completed_native_endpoints(
    landmark_hour: int,
    resolution_hours: int,
) -> List[int]:
    return list(
        range(
            resolution_hours,
            landmark_hour + 1,
            resolution_hours,
        )
    )


def resolution_landmark_status(
    landmark_hour: int,
    resolution_hours: int,
) -> dict:
    """
    Describe whether a native resolution is available at a landmark.

    A resolution is available only if at least one completed native endpoint
    exists at or before t. If available but not exactly aligned with t, the
    latest completed endpoint is used and the lag is recorded explicitly.

    Examples
    --------
    t=1h:
        o1 available at 1h
        o2/o3/o4 unavailable

    t=8h:
        o1/o2/o4 aligned at 8h
        o3 available only through 6h (lag=2h)
    """
    endpoints = completed_native_endpoints(
        landmark_hour,
        resolution_hours,
    )

    if not endpoints:
        return {
            "available": False,
            "native_endpoints_hours": [],
            "native_steps_per_patient": 0,
            "latest_native_endpoint_hour": None,
            "latest_native_endpoint_equals_landmark": False,
            "lag_to_landmark_hours": None,
        }

    latest = int(endpoints[-1])

    return {
        "available": True,
        "native_endpoints_hours": endpoints,
        "native_steps_per_patient": int(len(endpoints)),
        "latest_native_endpoint_hour": latest,
        "latest_native_endpoint_equals_landmark": bool(
            latest == landmark_hour
        ),
        "lag_to_landmark_hours": int(
            landmark_hour - latest
        ),
    }


def write_landmark_resolution(
    *,
    source_path: Path,
    risk_demo: pd.DataFrame,
    encoded_demo: pd.DataFrame,
    clinical_columns: Sequence[str],
    model_features: Sequence[str],
    database: str,
    split_name: str,
    landmark_hour: int,
    resolution_name: str,
    resolution_hours: int,
    output_path: Path,
) -> dict:
    if not source_path.is_file():
        raise FileNotFoundError(
            source_path
        )

    status = resolution_landmark_status(
        landmark_hour,
        resolution_hours,
    )
    endpoints = status[
        "native_endpoints_hours"
    ]

    if not status["available"]:
        return {
            "path": None,
            "available": False,
            "patients": int(len(risk_demo)),
            "rows": 0,
            "native_resolution_hours": int(
                resolution_hours
            ),
            "native_endpoints_hours": [],
            "native_steps_per_patient": 0,
            "latest_native_endpoint_hour": None,
            "latest_native_endpoint_equals_landmark": False,
            "lag_to_landmark_hours": None,
            "predictor_count": int(
                len(model_features)
            ),
            "clinical_nan_fraction": None,
            "reason": (
                f"No completed {resolution_hours}h native endpoint "
                f"exists by landmark {landmark_hour}h."
            ),
        }

    source_columns = (
        MODEL_METADATA_COLUMNS
        + [
            "icu_los_hours",
            "window_available",
        ]
        + list(clinical_columns)
    )

    df = pd.read_parquet(
        source_path,
        columns=source_columns,
    )

    df["patient_id"] = (
        df["patient_id"]
        .astype(str)
        .str.strip()
    )
    df["stay_id"] = (
        df["stay_id"]
        .astype(str)
        .str.strip()
    )

    risk_ids = set(
        risk_demo["patient_id"]
        .astype(str)
    )

    df = df[
        df["patient_id"].isin(
            risk_ids
        )
        & df["endpoint_hour"].isin(
            endpoints
        )
    ].copy()

    expected_steps = len(
        endpoints
    )
    expected_rows = (
        len(risk_demo)
        * expected_steps
    )

    if len(df) != expected_rows:
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            f"expected {expected_rows} rows, found {len(df)}."
        )

    observed_endpoints = sorted(
        df["endpoint_hour"]
        .astype(int)
        .unique()
        .tolist()
    )
    if observed_endpoints != endpoints:
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            f"endpoint grid mismatch. Expected {endpoints}, "
            f"observed {observed_endpoints}."
        )

    counts = (
        df.groupby(
            "patient_id",
            sort=False,
        )
        .size()
    )

    if len(counts) != len(
        risk_demo
    ):
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            "patient count mismatch."
        )

    if not (
        counts.to_numpy()
        == expected_steps
    ).all():
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            "wrong number of native rows for one or more patients."
        )

    # Core dynamic-landmark leakage barrier.
    if not df[
        "window_available"
    ].astype(bool).all():
        bad = df.loc[
            ~df["window_available"].astype(bool),
            [
                "patient_id",
                "endpoint_hour",
                "icu_los_hours",
            ],
        ].head(10)

        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            "structurally unavailable rows entered the risk set. "
            f"Examples: {bad.to_dict(orient='records')}"
        )

    if not (
        pd.to_numeric(
            df["icu_los_hours"],
            errors="coerce",
        )
        > float(landmark_hour) + 1e-9
    ).all():
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            "patient not in ICU at the landmark entered the dataset."
        )

    # Never use a future native endpoint.
    if (
        df["endpoint_hour"].astype(int)
        > landmark_hour
    ).any():
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h/{resolution_name}: "
            "future temporal row entered the landmark dataset."
        )

    # Exact demographic alignment.
    if set(
        encoded_demo["patient_id"]
        .astype(str)
    ) != risk_ids:
        raise RuntimeError(
            f"{database}/{split_name}/{landmark_hour}h: demographic "
            "patient set mismatch."
        )

    merged = df.merge(
        encoded_demo,
        on=[
            "patient_id",
            "stay_id",
        ],
        how="left",
        validate="many_to_one",
    )

    demographic_columns = [
        c
        for c in encoded_demo.columns
        if c not in {
            "patient_id",
            "stay_id",
        }
    ]

    gender_cols = [
        c
        for c in demographic_columns
        if c.startswith(
            "gender__"
        )
    ]
    race_cols = [
        c
        for c in demographic_columns
        if c.startswith(
            "race__"
        )
    ]

    if (
        merged[gender_cols]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Gender demographic merge failed."
        )

    if (
        merged[race_cols]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Race demographic merge failed."
        )

    model_ready = merged[
        MODEL_METADATA_COLUMNS
        + list(model_features)
    ].copy()

    forbidden = {
        "icu_los_days",
        "icu_los_hours",
        "los_total_days",
        "los_total_hours",
        "remaining_los_days",
        "remaining_los_hours",
        "hospital_expire_flag",
        "window_available",
        "landmark_eligible",
    }

    leaked = sorted(
        forbidden
        & set(
            model_ready.columns
        )
    )
    if leaked:
        raise RuntimeError(
            f"Leakage-prone columns in model-ready file: {leaked}"
        )

    if len(model_ready.columns) != (
        len(MODEL_METADATA_COLUMNS)
        + len(model_features)
    ):
        raise RuntimeError(
            "Model-ready width does not match metadata + frozen predictors."
        )

    model_ready = model_ready.sort_values(
        [
            "patient_id",
            "native_step_index",
        ],
        kind="stable",
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_ready.to_parquet(
        output_path,
        index=False,
        compression="zstd",
    )

    clinical = model_ready[
        list(clinical_columns)
    ]

    return {
        "path": str(
            output_path
        ),
        "available": True,
        "patients": int(
            model_ready[
                "patient_id"
            ].nunique()
        ),
        "rows": int(
            len(model_ready)
        ),
        "native_resolution_hours": int(
            resolution_hours
        ),
        "native_endpoints_hours": endpoints,
        "native_steps_per_patient": int(
            expected_steps
        ),
        "latest_native_endpoint_hour": int(
            endpoints[-1]
        ),
        "latest_native_endpoint_equals_landmark": bool(
            endpoints[-1]
            == landmark_hour
        ),
        "lag_to_landmark_hours": int(
            landmark_hour - endpoints[-1]
        ),
        "predictor_count": int(
            len(model_features)
        ),
        "clinical_nan_fraction": float(
            clinical
            .isna()
            .to_numpy()
            .mean()
        ),
    }


def risk_set_summary_row(
    *,
    database: str,
    split_name: str,
    landmark_hour: int,
    full_partition_n: int,
    demo: pd.DataFrame,
) -> dict:
    n = len(demo)

    mortality_known_mask = demo[
        "hospital_expire_flag"
    ].notna()
    mortality_known_n = int(
        mortality_known_mask.sum()
    )
    mortality_missing_n = int(
        (~mortality_known_mask).sum()
    )
    deaths = int(
        demo.loc[
            mortality_known_mask,
            "hospital_expire_flag",
        ].astype("int8").sum()
    )

    return {
        "database": database,
        "split": split_name,
        "landmark_hour": int(
            landmark_hour
        ),
        "full_partition_patients": int(
            full_partition_n
        ),
        "risk_set_patients_los_gt_t": int(
            n
        ),
        "risk_set_fraction_pct": (
            100.0
            * n
            / full_partition_n
            if full_partition_n
            else np.nan
        ),
        "mortality_known_patients": mortality_known_n,
        "mortality_missing_patients": mortality_missing_n,
        "mortality_events": deaths,
        "mortality_prevalence_among_known": (
            deaths / mortality_known_n
            if mortality_known_n
            else np.nan
        ),
        "remaining_los_mean_days": (
            float(
                demo[
                    "remaining_los_days"
                ].mean()
            )
            if n
            else np.nan
        ),
        "remaining_los_median_days": (
            float(
                demo[
                    "remaining_los_days"
                ].median()
            )
            if n
            else np.nan
        ),
        "remaining_los_iqr_days": (
            float(
                demo[
                    "remaining_los_days"
                ].quantile(0.75)
                - demo[
                    "remaining_los_days"
                ].quantile(0.25)
            )
            if n
            else np.nan
        ),
        "total_los_mean_days": (
            float(
                demo[
                    "icu_los_days"
                ].mean()
            )
            if n
            else np.nan
        ),
    }


def nested_risk_set_audit(
    landmark_sets: Mapping[
        Tuple[str, str, int],
        Sequence[str],
    ],
    landmarks: Sequence[int],
) -> pd.DataFrame:
    rows = []

    groups = sorted(
        {
            (
                database,
                split_name,
            )
            for (
                database,
                split_name,
                _
            ) in landmark_sets.keys()
        }
    )

    for database, split_name in groups:
        for smaller, larger in zip(
            landmarks[:-1],
            landmarks[1:],
        ):
            small_set = set(
                landmark_sets[
                    (
                        database,
                        split_name,
                        smaller,
                    )
                ]
            )
            large_set = set(
                landmark_sets[
                    (
                        database,
                        split_name,
                        larger,
                    )
                ]
            )

            # Later risk set must be a subset of the earlier risk set.
            passed = large_set.issubset(
                small_set
            )

            rows.append(
                {
                    "database": database,
                    "split": split_name,
                    "earlier_landmark_hour": int(
                        smaller
                    ),
                    "later_landmark_hour": int(
                        larger
                    ),
                    "earlier_n": int(
                        len(small_set)
                    ),
                    "later_n": int(
                        len(large_set)
                    ),
                    "later_subset_of_earlier": bool(
                        passed
                    ),
                    "violating_patient_count": int(
                        len(
                            large_set
                            - small_set
                        )
                    ),
                }
            )

    df = pd.DataFrame(
        rows
    )

    if not df.empty and not df[
        "later_subset_of_earlier"
    ].all():
        raise RuntimeError(
            "Dynamic landmark risk sets are not nested."
        )

    return df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Create one MIMIC 80/20 development/test split and leakage-safe "
            "dynamic landmark risk sets. Five-fold CV is performed later "
            "within the development set."
        )
    )

    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--harmonized-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--window-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--landmarks",
        type=parse_landmarks,
        default=DEFAULT_LANDMARKS,
        help=(
            "Comma-separated landmark hours. "
            "Default: 1,4,8,12,16,20,24,36,48"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SEED_DEFAULT,
    )
    p.add_argument(
        "--train-fraction",
        type=float,
        default=TRAIN_FRACTION_DEFAULT,
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=TEST_FRACTION_DEFAULT,
    )

    return p


def main() -> int:
    args = build_parser().parse_args()
    require_dependencies()

    validate_split_fractions(
        args.train_fraction,
        args.test_fraction,
    )

    landmarks = list(
        args.landmarks
    )

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    harmonized_dir = (
        args.harmonized_dir
        .expanduser()
        .resolve()
        if args.harmonized_dir is not None
        else project_root
        / "data"
        / "03_harmonized"
    )
    window_dir = (
        args.window_dir
        .expanduser()
        .resolve()
        if args.window_dir is not None
        else project_root
        / "data"
        / "04_cumulative_windows"
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
        if args.output_dir is not None
        else project_root
        / "data"
        / "05_splits"
    )

    logger = configure_logging(
        output_dir
    )

    logger.info(
        "Project root: %s",
        project_root,
    )
    logger.info(
        "Harmonized directory: %s",
        harmonized_dir,
    )
    logger.info(
        "Window directory: %s",
        window_dir,
    )
    logger.info(
        "Output directory: %s",
        output_dir,
    )
    logger.info(
        "Dynamic landmarks: %s",
        landmarks,
    )
    logger.info(
        "MIMIC split: train=%.2f test=%.2f seed=%d | "
        "validation=5-fold CV inside train",
        args.train_fraction,
        args.test_fraction,
        args.seed,
    )

    # Report native-grid availability at each landmark.
    # No future endpoint is ever used.
    for landmark in landmarks:
        for name, spec in RESOLUTIONS.items():
            status = resolution_landmark_status(
                landmark,
                spec["hours"],
            )
            if not status["available"]:
                logger.warning(
                    "Landmark %dh: %s (%dh) is unavailable because no "
                    "completed native endpoint exists by t. This resolution "
                    "will be skipped for this landmark.",
                    landmark,
                    name,
                    spec["hours"],
                )
            elif not status[
                "latest_native_endpoint_equals_landmark"
            ]:
                logger.warning(
                    "Landmark %dh: %s (%dh) is available only through %dh "
                    "(lag=%dh). No future coarse summary will be used.",
                    landmark,
                    name,
                    spec["hours"],
                    status["latest_native_endpoint_hour"],
                    status["lag_to_landmark_hours"],
                )

    clinical_columns = load_feature_order(
        window_dir
    )

    mimic_demo = load_demographics(
        harmonized_dir
        / "mimic"
        / "harmonized_demographics.parquet",
        "mimic",
    )
    eicu_demo = load_demographics(
        harmonized_dir
        / "eicu"
        / "harmonized_demographics.parquet",
        "eicu",
    )

    logger.info(
        "Loaded full cohorts | MIMIC=%d eICU=%d",
        len(mimic_demo),
        len(eicu_demo),
    )
    logger.info(
        "Mortality labels | MIMIC known=%d missing=%d | eICU known=%d missing=%d",
        int(mimic_demo["mortality_known"].sum()),
        int((~mimic_demo["mortality_known"]).sum()),
        int(eicu_demo["mortality_known"].sum()),
        int((~eicu_demo["mortality_known"]).sum()),
    )

    # --------------------------------------------------------------
    # One frozen MIMIC patient split.
    # --------------------------------------------------------------
    full_assignment = (
        create_full_mimic_split(
            mimic_demo,
            train_fraction=args.train_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
    )

    reports_dir = (
        output_dir
        / "reports"
    )
    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_assignment.to_csv(
        reports_dir
        / "full_cohort_split_assignments.csv",
        index=False,
    )

    split_ids = {
        split_name: (
            full_assignment.loc[
                full_assignment[
                    "split"
                ].eq(
                    split_name
                ),
                "patient_id",
            ]
            .astype(str)
            .tolist()
        )
        for split_name in [
            "train",
            "test",
        ]
    }

    full_split_counts = {
        name: len(ids)
        for name, ids in split_ids.items()
    }

    logger.info(
        "Full MIMIC split | train=%d test=%d",
        full_split_counts["train"],
        full_split_counts["test"],
    )

    # --------------------------------------------------------------
    # Freeze demographic schema ONCE from MIMIC training.
    # --------------------------------------------------------------
    demographic_schema = (
        freeze_demographic_schema(
            mimic_demo,
            full_assignment,
            logger,
        )
    )

    schema_path = (
        output_dir
        / "demographic_schema.json"
    )
    schema_path.write_text(
        json.dumps(
            demographic_schema,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    model_features, feature_order_df = (
        build_feature_order(
            clinical_columns,
            demographic_schema,
        )
    )

    feature_order_df.to_csv(
        reports_dir
        / "feature_order.csv",
        index=False,
    )

    (
        output_dir
        / "feature_order.json"
    ).write_text(
        json.dumps(
            {
                "clinical_feature_count": 304,
                "demographic_feature_count": int(
                    demographic_schema["demographic_feature_count"]
                ),
                "base_feature_count": int(len(model_features)),
                "feature_columns": model_features,
                "demographic_schema_file": str(
                    schema_path
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Training category report.
    train_id_set = set(
        split_ids["train"]
    )
    train_demo_full = mimic_demo[
        mimic_demo["patient_id"].isin(
            train_id_set
        )
    ]

    category_rows = []
    for variable in [
        "gender",
        "race",
    ]:
        counts = (
            train_demo_full[
                variable
            ]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )
        for category, n in counts.items():
            category_rows.append(
                {
                    "fit_database": "mimic",
                    "fit_split": "train_full_cohort",
                    "variable": variable,
                    "category": category,
                    "patient_count": int(
                        n
                    ),
                    "included_in_frozen_schema": True,
                }
            )

    pd.DataFrame(
        category_rows
    ).to_csv(
        reports_dir
        / "demographic_category_counts.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Build all dynamic risk sets.
    # --------------------------------------------------------------
    risk_sets: Dict[
        Tuple[str, str, int],
        pd.DataFrame,
    ] = {}
    landmark_set_ids: Dict[
        Tuple[str, str, int],
        Sequence[str],
    ] = {}

    risk_summary_rows = []
    assignment_parts = []

    for landmark in landmarks:
        for split_name in [
            "train",
            "test",
        ]:
            rs = risk_set(
                mimic_demo,
                split_ids[
                    split_name
                ],
                landmark,
            )
            risk_sets[
                (
                    "mimic",
                    split_name,
                    landmark,
                )
            ] = rs
            landmark_set_ids[
                (
                    "mimic",
                    split_name,
                    landmark,
                )
            ] = rs[
                "patient_id"
            ].astype(str).tolist()

            risk_summary_rows.append(
                risk_set_summary_row(
                    database="mimic",
                    split_name=split_name,
                    landmark_hour=landmark,
                    full_partition_n=(
                        full_split_counts[
                            split_name
                        ]
                    ),
                    demo=rs,
                )
            )

            part = rs[
                [
                    "patient_id",
                    "stay_id",
                    "icu_los_days",
                    "icu_los_hours",
                    "remaining_los_days",
                    "remaining_los_hours",
                    "hospital_expire_flag",
                    "mortality_known",
                ]
            ].copy()
            part.insert(
                0,
                "landmark_hour",
                landmark,
            )
            part.insert(
                0,
                "split",
                split_name,
            )
            part.insert(
                0,
                "database",
                "mimic",
            )
            assignment_parts.append(
                part
            )

        external_ids = (
            eicu_demo[
                "patient_id"
            ]
            .astype(str)
            .tolist()
        )

        rs_ext = risk_set(
            eicu_demo,
            external_ids,
            landmark,
        )
        risk_sets[
            (
                "eicu",
                "external",
                landmark,
            )
        ] = rs_ext
        landmark_set_ids[
            (
                "eicu",
                "external",
                landmark,
            )
        ] = rs_ext[
            "patient_id"
        ].astype(str).tolist()

        risk_summary_rows.append(
            risk_set_summary_row(
                database="eicu",
                split_name="external",
                landmark_hour=landmark,
                full_partition_n=len(
                    eicu_demo
                ),
                demo=rs_ext,
            )
        )

        part = rs_ext[
            [
                "patient_id",
                "stay_id",
                "icu_los_days",
                "icu_los_hours",
                "remaining_los_days",
                "remaining_los_hours",
                "hospital_expire_flag",
                "mortality_known",
            ]
        ].copy()
        part.insert(
            0,
            "landmark_hour",
            landmark,
        )
        part.insert(
            0,
            "split",
            "external",
        )
        part.insert(
            0,
            "database",
            "eicu",
        )
        assignment_parts.append(
            part
        )

    risk_summary_df = pd.DataFrame(
        risk_summary_rows
    )
    risk_summary_df.to_csv(
        reports_dir
        / "landmark_risk_set_summary.csv",
        index=False,
    )

    landmark_assignment = pd.concat(
        assignment_parts,
        ignore_index=True,
    )
    landmark_assignment.to_csv(
        reports_dir
        / "landmark_split_assignments.csv",
        index=False,
    )

    nested_df = (
        nested_risk_set_audit(
            landmark_set_ids,
            landmarks,
        )
    )
    nested_df.to_csv(
        reports_dir
        / "nested_risk_set_audit.csv",
        index=False,
    )

    logger.info(
        "Dynamic risk sets are nested: PASS"
    )

    # --------------------------------------------------------------
    # Encode demographics and write each landmark/resolution dataset.
    # --------------------------------------------------------------
    demographic_audit_rows = []
    grid_summary_rows = []
    manifest_landmarks = {}

    for landmark in landmarks:
        landmark_key = (
            f"landmark_{landmark:03d}h"
        )
        landmark_dir = (
            output_dir
            / landmark_key
        )
        targets_dir = (
            landmark_dir
            / "targets"
        )
        targets_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_landmarks[
            str(landmark)
        ] = {
            "landmark_hour": int(
                landmark
            ),
            "eligibility": (
                f"total ICU LOS > {landmark} hours"
            ),
            "observation": (
                f"completed cumulative native endpoints <= {landmark}h"
            ),
            "los_target": (
                f"remaining ICU LOS = total ICU LOS - {landmark}h"
            ),
            "mortality_target": (
                "subsequent in-hospital mortality among patients "
                "still in ICU at the landmark"
            ),
            "splits": {},
        }

        split_specs = [
            (
                "mimic",
                "train",
            ),
            (
                "mimic",
                "test",
            ),
            (
                "eicu",
                "external",
            ),
        ]

        for database, split_name in split_specs:
            rs = risk_sets[
                (
                    database,
                    split_name,
                    landmark,
                )
            ]

            encoded, audit = (
                encode_demographics(
                    rs,
                    schema=(
                        demographic_schema
                    ),
                    database=database,
                    split_name=split_name,
                    landmark_hour=landmark,
                )
            )
            demographic_audit_rows.append(
                audit
            )

            target = make_targets(
                rs,
                database=database,
                split_name=split_name,
                landmark_hour=landmark,
            )

            target_path = (
                targets_dir
                / (
                    f"{database}_{split_name}"
                    "_targets.parquet"
                )
            )
            target.to_parquet(
                target_path,
                index=False,
                compression="zstd",
            )

            split_manifest = {
                "patients": int(
                    len(rs)
                ),
                "target_file": str(
                    target_path
                ),
                "resolutions": {},
            }

            for resolution_name, spec in RESOLUTIONS.items():
                source_path = (
                    window_dir
                    / database
                    / spec[
                        "filename"
                    ]
                )
                output_path = (
                    landmark_dir
                    / database
                    / split_name
                    / spec[
                        "filename"
                    ]
                )

                result = (
                    write_landmark_resolution(
                        source_path=source_path,
                        risk_demo=rs,
                        encoded_demo=encoded,
                        clinical_columns=clinical_columns,
                        model_features=model_features,
                        database=database,
                        split_name=split_name,
                        landmark_hour=landmark,
                        resolution_name=resolution_name,
                        resolution_hours=spec["hours"],
                        output_path=output_path,
                    )
                )

                split_manifest[
                    "resolutions"
                ][
                    resolution_name
                ] = result

                grid_summary_rows.append(
                    {
                        "database": database,
                        "split": split_name,
                        "landmark_hour": int(
                            landmark
                        ),
                        "resolution": resolution_name,
                        "resolution_hours": int(
                            spec["hours"]
                        ),
                        "available": bool(
                            result["available"]
                        ),
                        "patients": int(
                            result["patients"]
                        ),
                        "native_steps_per_patient": int(
                            result[
                                "native_steps_per_patient"
                            ]
                        ),
                        "latest_native_endpoint_hour": (
                            int(
                                result[
                                    "latest_native_endpoint_hour"
                                ]
                            )
                            if result[
                                "latest_native_endpoint_hour"
                            ] is not None
                            else np.nan
                        ),
                        "latest_native_endpoint_equals_landmark": bool(
                            result[
                                "latest_native_endpoint_equals_landmark"
                            ]
                        ),
                        "lag_to_landmark_hours": (
                            int(
                                result[
                                    "lag_to_landmark_hours"
                                ]
                            )
                            if result[
                                "lag_to_landmark_hours"
                            ] is not None
                            else np.nan
                        ),
                        "rows": int(
                            result["rows"]
                        ),
                        "clinical_nan_fraction": (
                            float(
                                result[
                                    "clinical_nan_fraction"
                                ]
                            )
                            if result[
                                "clinical_nan_fraction"
                            ] is not None
                            else np.nan
                        ),
                    }
                )

                if result["available"]:
                    logger.info(
                        "%dh %s/%s/%s | patients=%d rows=%d "
                        "steps=%d last_endpoint=%dh lag=%dh predictors=%d",
                        landmark,
                        database,
                        split_name,
                        resolution_name,
                        result[
                            "patients"
                        ],
                        result[
                            "rows"
                        ],
                        result[
                            "native_steps_per_patient"
                        ],
                        result[
                            "latest_native_endpoint_hour"
                        ],
                        result[
                            "lag_to_landmark_hours"
                        ],
                        result[
                            "predictor_count"
                        ],
                    )
                else:
                    logger.info(
                        "%dh %s/%s/%s | UNAVAILABLE | patients=%d "
                        "reason=%s",
                        landmark,
                        database,
                        split_name,
                        resolution_name,
                        result["patients"],
                        result["reason"],
                    )

            manifest_landmarks[
                str(landmark)
            ][
                "splits"
            ][
                f"{database}_{split_name}"
            ] = split_manifest

    pd.DataFrame(
        demographic_audit_rows
    ).to_csv(
        reports_dir
        / "demographic_encoding_audit.csv",
        index=False,
    )

    pd.DataFrame(
        grid_summary_rows
    ).to_csv(
        reports_dir
        / "landmark_grid_summary.csv",
        index=False,
    )

    # Landmark classification for the paper:
    # - fully_aligned_four_resolution: all o1-o4 available and end exactly at t
    # - partially_aligned_multi_resolution: >=2 resolutions available but at
    #   least one ends before t
    # - single_resolution_only: only one native resolution exists by t
    landmark_design_rows = []
    for landmark in landmarks:
        statuses = {
            name: resolution_landmark_status(
                landmark,
                spec["hours"],
            )
            for name, spec in RESOLUTIONS.items()
        }
        available_names = [
            name
            for name, status in statuses.items()
            if status["available"]
        ]
        all_four_aligned = (
            len(available_names) == 4
            and all(
                statuses[name][
                    "latest_native_endpoint_equals_landmark"
                ]
                for name in RESOLUTIONS
            )
        )

        if all_four_aligned:
            design_class = "fully_aligned_four_resolution"
        elif len(available_names) >= 2:
            design_class = "partially_aligned_multi_resolution"
        else:
            design_class = "single_resolution_only"

        landmark_design_rows.append(
            {
                "landmark_hour": int(landmark),
                "design_class": design_class,
                "available_resolutions": ",".join(
                    available_names
                ),
                "fully_aligned_four_resolution": bool(
                    all_four_aligned
                ),
                "o1_latest_hour": statuses["o1"][
                    "latest_native_endpoint_hour"
                ],
                "o2_latest_hour": statuses["o2"][
                    "latest_native_endpoint_hour"
                ],
                "o3_latest_hour": statuses["o3"][
                    "latest_native_endpoint_hour"
                ],
                "o4_latest_hour": statuses["o4"][
                    "latest_native_endpoint_hour"
                ],
            }
        )

    landmark_design_df = pd.DataFrame(
        landmark_design_rows
    )
    landmark_design_df.to_csv(
        reports_dir
        / "landmark_design_summary.csv",
        index=False,
    )

    manifest = {
        "script": "05_create_splits_and_encode.py",
        "design": "dynamic landmark risk-set prediction",
        "project_root": str(
            project_root
        ),
        "harmonized_dir": str(
            harmonized_dir
        ),
        "window_dir": str(
            window_dir
        ),
        "output_dir": str(
            output_dir
        ),
        "landmarks_hours": [
            int(x)
            for x in landmarks
        ],
        "risk_set_rule": (
            "at landmark t include only patients with total ICU LOS > t"
        ),
        "temporal_rule": (
            "include only completed native cumulative endpoints <= t; "
            "never use a future coarse-resolution endpoint"
        ),
        "landmark_design_policy": (
            "All requested prediction landmarks are retained. At very early "
            "landmarks, unavailable native resolutions are skipped. At "
            "non-aligned landmarks, a resolution may end before t but never "
            "after t. Fully aligned four-resolution landmarks are identified "
            "separately for strict multiresolution comparison."
        ),
        "los_primary_target": (
            "remaining ICU LOS after each landmark"
        ),
        "total_los_retained_in_target_files_for_reporting": True,
        "mortality_target": (
            "in-hospital mortality among patients still in ICU at t"
        ),
        "split_created_once_on_full_mimic_cohort": True,
        "split_strategy": (
            "patient-level 80/20 development/test split, stratified only "
            "on in-hospital mortality"
        ),
        "internal_validation_strategy": (
            "five-fold cross-validation within the MIMIC development set"
        ),
        "split_seed": int(
            args.seed
        ),
        "split_fractions": {
            "train": float(args.train_fraction),
            "test": float(args.test_fraction),
        },
        "full_mimic_split_counts": {
            key: int(value)
            for key, value in (
                full_split_counts.items()
            )
        },
        "demographic_schema_fit_source": (
            "MIMIC full-cohort training partition only"
        ),
        "eicu_used_for_split_or_schema_fit": False,
        "mortality_missingness_policy": (
            "Unknown mortality is never imputed. Patients with unknown "
            "mortality remain eligible for LOS; mortality modeling/evaluation "
            "uses only mortality_known==True."
        ),
        "clinical_feature_count": 304,
        "demographic_feature_count": int(
            demographic_schema["demographic_feature_count"]
        ),
        "base_feature_count": int(len(model_features)),
        "gender_columns": int(
            len(demographic_schema["gender_columns"])
        ),
        "race_columns": int(
            len(demographic_schema["race_columns"])
        ),
        "targets_separate_from_predictors": True,
        "icu_los_in_model_ready_predictors": False,
        "window_available_in_model_ready_predictors": False,
        "imputation_performed": False,
        "scaling_performed": False,
        "feature_selection_performed": False,
        "demographic_schema": demographic_schema,
        "landmarks": manifest_landmarks,
        "reports": {
            "full_cohort_split_assignments": str(
                reports_dir
                / "full_cohort_split_assignments.csv"
            ),
            "demographic_category_counts": str(
                reports_dir
                / "demographic_category_counts.csv"
            ),
            "demographic_encoding_audit": str(
                reports_dir
                / "demographic_encoding_audit.csv"
            ),
            "feature_order": str(
                reports_dir
                / "feature_order.csv"
            ),
            "landmark_risk_set_summary": str(
                reports_dir
                / "landmark_risk_set_summary.csv"
            ),
            "landmark_grid_summary": str(
                reports_dir
                / "landmark_grid_summary.csv"
            ),
            "nested_risk_set_audit": str(
                reports_dir
                / "nested_risk_set_audit.csv"
            ),
            "landmark_split_assignments": str(
                reports_dir
                / "landmark_split_assignments.csv"
            ),
            "landmark_design_summary": str(
                reports_dir
                / "landmark_design_summary.csv"
            ),
        },
    }

    manifest_path = (
        output_dir
        / "05_multilandmark_manifest.json"
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
        "Stage 05 complete | landmarks=%s | predictors=%d",
        landmarks,
        len(model_features),
    )
    logger.info(
        "All risk sets use LOS>t and all temporal files use endpoints<=t."
    )
    logger.info(
        "Manifest: %s",
        manifest_path,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
