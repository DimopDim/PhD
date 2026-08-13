from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config


@dataclass
class FrameSet:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    split_groups: np.ndarray
    sequence_groups: np.ndarray
    window_order: np.ndarray
    row_ids: np.ndarray

    def take(self, indices: np.ndarray, name: str) -> "FrameSet":
        return FrameSet(
            name=name,
            X=self.X.iloc[indices].reset_index(drop=True),
            y=self.y[indices],
            split_groups=self.split_groups[indices],
            sequence_groups=self.sequence_groups[indices],
            window_order=self.window_order[indices],
            row_ids=self.row_ids[indices],
        )


@dataclass
class DataBundle:
    train: FrameSet
    validation: FrameSet
    test: FrameSet
    external: FrameSet
    feature_columns: list[str]
    audit: dict


def _read_table(path: str) -> pd.DataFrame:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix in {".csv", ".gz", ".bz2", ".xz"} or resolved.name.endswith(".csv.gz"):
        return pd.read_csv(resolved, low_memory=False)
    raise ValueError(f"Unsupported input format: {resolved}. Use CSV or Parquet.")


def _hash_file(path: str, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _key(df: pd.DataFrame, columns: Iterable[str], label: str) -> np.ndarray:
    columns = list(columns)
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing key columns {missing}")
    if df[columns].isna().any(axis=None):
        raise ValueError(f"{label}: group identifiers contain missing values")
    return df[columns].astype(str).agg("\x1f".join, axis=1).to_numpy(dtype=str)


def _assert_constant_within_group(
    values: pd.Series, groups: np.ndarray, label: str, allow_multiple: bool = False
) -> None:
    if allow_multiple:
        return
    nunique = pd.DataFrame({"group": groups, "value": values}).groupby("group")["value"].nunique(dropna=False)
    bad = nunique[nunique != 1]
    if not bad.empty:
        raise ValueError(f"{label} is not constant within {len(bad)} sequence groups.")


def _resolve_features(mimic: pd.DataFrame, external: pd.DataFrame, config: Config) -> list[str]:
    dcfg = config.data
    if dcfg.feature_columns:
        features = list(dcfg.feature_columns)
    else:
        excluded = set(dcfg.exclude_columns)
        features = [column for column in mimic.columns if column not in excluded and column in external.columns]
    if not features:
        raise ValueError("No common predictor columns remain after exclusions.")
    missing_mimic = sorted(set(features) - set(mimic.columns))
    missing_external = sorted(set(features) - set(external.columns))
    if missing_mimic or missing_external:
        raise ValueError(
            f"Feature schema mismatch. Missing from MIMIC={missing_mimic}; "
            f"missing from eICU={missing_external}."
        )
    forbidden = set(dcfg.exclude_columns) | {dcfg.target_column}
    overlap = sorted(set(features) & forbidden)
    if overlap:
        raise ValueError(f"Excluded/outcome columns were selected as predictors: {overlap}")
    suspicious_tokens = ("los", "length_of_stay", "mortality", "expire", "death", "discharge")
    suspicious = [column for column in features if any(token in column.lower() for token in suspicious_tokens)]
    if suspicious:
        raise ValueError(
            "Potential outcome/post-outcome predictors detected. Add them to exclude_columns or "
            f"rename genuine predictors after review: {suspicious}"
        )
    return features


def _make_frameset(
    name: str,
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_group_columns: tuple[str, ...],
    sequence_group_columns: tuple[str, ...],
    window_order_column: str,
) -> FrameSet:
    required = [target_column, window_order_column, *split_group_columns, *sequence_group_columns]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {missing}")
    if df[target_column].isna().any():
        raise ValueError(f"{name}: target contains missing values")
    sequence_groups = _key(df, sequence_group_columns, name)
    split_groups = _key(df, split_group_columns, name)
    _assert_constant_within_group(df[target_column], sequence_groups, f"{name} target")
    if df[window_order_column].isna().any():
        raise ValueError(f"{name}: window-order column contains missing values")
    row_ids = np.asarray([f"{name}:{index}" for index in range(len(df))], dtype=str)
    return FrameSet(
        name=name,
        X=df[feature_columns].copy(),
        y=pd.to_numeric(df[target_column], errors="raise").to_numpy(dtype=float),
        split_groups=split_groups,
        sequence_groups=sequence_groups,
        window_order=pd.to_numeric(df[window_order_column], errors="raise").to_numpy(dtype=float),
        row_ids=row_ids,
    )


def _split_group_ids(mimic: pd.DataFrame, config: Config) -> tuple[set[str], set[str], set[str]]:
    dcfg, scfg = config.data, config.split
    split_key = _key(mimic, dcfg.split_group_columns, "mimic")
    group_table = pd.DataFrame({"group": split_key})
    if dcfg.stratify_column:
        if dcfg.stratify_column not in mimic.columns:
            raise ValueError(f"Missing stratification column: {dcfg.stratify_column}")
        if mimic[dcfg.stratify_column].isna().any():
            raise ValueError("Stratification column contains missing values.")
        group_table["stratum"] = mimic[dcfg.stratify_column].to_numpy()
        # A patient with multiple admissions is assigned the highest observed mortality indicator.
        group_table = group_table.groupby("group", as_index=False)["stratum"].max()
    else:
        sequence_key = _key(mimic, dcfg.sequence_group_columns, "mimic")
        admission_targets = pd.DataFrame(
            {"split_group": split_key, "sequence_group": sequence_key, "target": mimic[dcfg.target_column]}
        ).drop_duplicates("sequence_group")
        group_table = admission_targets.groupby("split_group", as_index=False)["target"].mean()
        group_table.columns = ["group", "target"]
        group_table["stratum"] = pd.qcut(group_table["target"], q=min(10, len(group_table)), duplicates="drop")

    groups = group_table["group"].to_numpy(dtype=str)
    strata = group_table["stratum"].to_numpy()
    train_groups, rest_groups = train_test_split(
        groups,
        test_size=scfg.validation_fraction + scfg.test_fraction,
        random_state=scfg.seed,
        shuffle=True,
        stratify=strata,
    )
    rest_table = group_table.set_index("group").loc[rest_groups]
    relative_test = scfg.test_fraction / (scfg.validation_fraction + scfg.test_fraction)
    validation_groups, test_groups = train_test_split(
        rest_groups,
        test_size=relative_test,
        random_state=scfg.seed + 1,
        shuffle=True,
        stratify=rest_table["stratum"].to_numpy(),
    )
    return set(train_groups), set(validation_groups), set(test_groups)


def _indices_for_groups(groups: np.ndarray, selected: set[str]) -> np.ndarray:
    return np.flatnonzero(np.fromiter((group in selected for group in groups), dtype=bool))


def _validate_windows(frame: FrameSet, expected: int) -> dict[str, int]:
    counts = pd.Series(frame.sequence_groups).value_counts()
    duplicates = pd.DataFrame(
        {"sequence": frame.sequence_groups, "window": frame.window_order}
    ).duplicated().sum()
    if duplicates:
        raise ValueError(f"{frame.name}: found {duplicates} duplicate sequence/window pairs")
    bad = counts[counts != expected]
    if not bad.empty:
        sample = bad.head(10).to_dict()
        raise ValueError(
            f"{frame.name}: expected exactly {expected} rows per admission; "
            f"{len(bad)} admissions differ. Examples: {sample}"
        )
    return {"admissions": int(len(counts)), "rows": int(len(frame.y)), "windows_per_admission": expected}


def _filter_target(df: pd.DataFrame, config: Config, label: str) -> tuple[pd.DataFrame, dict]:
    target = config.data.target_column
    if target not in df.columns:
        raise ValueError(f"{label}: missing target column {target}")

    numeric = pd.to_numeric(df[target], errors="raise")
    if numeric.isna().any():
        raise ValueError(f"{label}: target contains missing values")

    # Cohort eligibility is defined on the admission-level LOS target.
    # Non-positive LOS values are invalid for ICU LOS prediction and are
    # excluded together with admissions outside the configured upper bound.
    # Filtering is performed before splitting/preprocessing, so every window
    # from an ineligible admission is removed consistently.
    initial_rows = len(df)
    nonpositive = numeric <= 0
    keep = ~nonpositive

    upper_excluded = pd.Series(False, index=df.index)
    if config.data.max_los_inclusive is not None:
        upper_excluded = numeric > config.data.max_los_inclusive
        keep &= ~upper_excluded

    df = df.loc[keep].reset_index(drop=True)

    return df, {
        "initial_rows": initial_rows,
        "retained_rows": len(df),
        "excluded_rows": initial_rows - len(df),
        "excluded_nonpositive_rows": int(nonpositive.sum()),
        "excluded_upper_bound_rows": int(upper_excluded.sum()),
        "rule": (
            f"0 < {target} <= {config.data.max_los_inclusive}"
            if config.data.max_los_inclusive is not None
            else f"{target} > 0"
        ),
    }


def load_and_split(config: Config) -> DataBundle:
    dcfg = config.data
    mimic_df = _read_table(dcfg.mimic_path)
    external_df = _read_table(dcfg.eicu_path)
    mimic_df, mimic_filter_audit = _filter_target(mimic_df, config, "mimic")
    external_df, external_filter_audit = _filter_target(external_df, config, "eicu")
    features = _resolve_features(mimic_df, external_df, config)

    mimic = _make_frameset(
        "mimic_all", mimic_df, features, dcfg.target_column,
        dcfg.split_group_columns, dcfg.sequence_group_columns, dcfg.window_order_column,
    )
    external = _make_frameset(
        "eicu_external", external_df, features, dcfg.target_column,
        dcfg.external_sequence_group_columns, dcfg.external_sequence_group_columns,
        dcfg.window_order_column,
    )
    train_groups, validation_groups, test_groups = _split_group_ids(mimic_df, config)
    train = mimic.take(_indices_for_groups(mimic.split_groups, train_groups), "mimic_train")
    validation = mimic.take(_indices_for_groups(mimic.split_groups, validation_groups), "mimic_validation")
    test = mimic.take(_indices_for_groups(mimic.split_groups, test_groups), "mimic_test")

    group_sets = [set(frame.split_groups) for frame in (train, validation, test)]
    overlaps = {
        "train_validation": len(group_sets[0] & group_sets[1]),
        "train_test": len(group_sets[0] & group_sets[2]),
        "validation_test": len(group_sets[1] & group_sets[2]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient-level split leakage detected: {overlaps}")

    window_audit = {
        frame.name: _validate_windows(frame, dcfg.expected_windows_per_admission)
        for frame in (train, validation, test, external)
    }
    expected_counts = {
        "mimic": dcfg.expected_mimic_admissions,
        "eicu": dcfg.expected_eicu_admissions,
    }
    actual_counts = {
        "mimic": int(pd.Series(mimic.sequence_groups).nunique()),
        "eicu": int(pd.Series(external.sequence_groups).nunique()),
    }
    mismatches = {
        name: {"expected": expected_counts[name], "actual": actual_counts[name]}
        for name in expected_counts
        if expected_counts[name] is not None and expected_counts[name] != actual_counts[name]
    }
    if mismatches:
        raise ValueError(
            "Cohort admission counts do not match the configured reproducibility guards: "
            f"{mismatches}. Set the expected count to the verified value only after reviewing the input."
        )
    audit = {
        "input_sha256": {
            "mimic": _hash_file(dcfg.mimic_path),
            "eicu": _hash_file(dcfg.eicu_path),
        },
        "feature_count": len(features),
        "features": features,
        "split_group_overlap": overlaps,
        "sets": window_audit,
        "los_filter": {"mimic": mimic_filter_audit, "eicu": external_filter_audit},
        "cohort_admissions": actual_counts,
    }
    return DataBundle(train, validation, test, external, features, audit)


def save_split_manifest(bundle: DataBundle, path: Path) -> None:
    rows: list[dict[str, str]] = []
    for frame in (bundle.train, bundle.validation, bundle.test, bundle.external):
        for split_group, sequence_group, window, row_id in zip(
            frame.split_groups, frame.sequence_groups, frame.window_order, frame.row_ids
        ):
            rows.append(
                {
                    "dataset": frame.name,
                    "split_group": split_group,
                    "sequence_group": sequence_group,
                    "window_order": str(window),
                    "row_id": row_id,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def audit_json(bundle: DataBundle) -> str:
    return json.dumps(bundle.audit, indent=2, sort_keys=True)
