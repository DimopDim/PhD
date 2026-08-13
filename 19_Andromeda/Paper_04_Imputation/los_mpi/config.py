from __future__ import annotations

import dataclasses
import json
import tomllib
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class DataConfig:
    mimic_path: str
    eicu_path: str
    target_column: str
    split_group_columns: tuple[str, ...]
    sequence_group_columns: tuple[str, ...]
    external_sequence_group_columns: tuple[str, ...]
    window_order_column: str
    stratify_column: str | None
    exclude_columns: tuple[str, ...]
    feature_columns: tuple[str, ...] = ()
    expected_windows_per_admission: int = 16
    max_los_inclusive: float | None = 10.0
    expected_mimic_admissions: int | None = 3190
    expected_eicu_admissions: int | None = 4886


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 42


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str
    search_iterations: int = 50
    bayesian_folds: int = 3
    threads_per_rank: int = 1
    device: str = "cpu"
    bootstrap_iterations: int = 1000
    resume: bool = True


@dataclasses.dataclass(frozen=True)
class ImputationConfig:
    iterative_max_iter: int = 20
    iterative_trees: int = 10
    neural_epochs: int = 50
    neural_batch_size: int = 32
    neural_patience: int = 5
    neural_validation_fraction: float = 0.1
    neural_corruption_fraction: float = 0.2


@dataclasses.dataclass(frozen=True)
class Config:
    data: DataConfig
    split: SplitConfig
    experiment: ExperimentConfig
    imputation: ImputationConfig

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def _tuple_fields(cls: type, values: dict[str, Any], names: tuple[str, ...]) -> Any:
    copied = dict(values)
    for name in names:
        if name in copied:
            copied[name] = tuple(copied[name])
    return cls(**copied)


def _resolve_input_path(value: str, base_dir: Path) -> str:
    # Accept either Windows-style or POSIX-style separators in TOML values.
    normalized = str(value).replace("\\", "/")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return str(candidate.resolve())


def load_config(path: str | Path) -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    data_values = dict(raw["data"])
    # v1.0.4+: LOS cohort rule is 0 < LOS <= max_los_inclusive; eICU guard updated after audited exclusion of LOS=0 admissions.
    # Accept the legacy v1.0.2 key so existing server configs continue to work.
    if "max_los_inclusive" not in data_values and "max_los_exclusive" in data_values:
        data_values["max_los_inclusive"] = data_values.pop("max_los_exclusive")
    elif "max_los_inclusive" in data_values and "max_los_exclusive" in data_values:
        raise ValueError(
            "Specify only one of max_los_inclusive or legacy max_los_exclusive."
        )
    data_values["mimic_path"] = _resolve_input_path(data_values["mimic_path"], config_path.parent)
    data_values["eicu_path"] = _resolve_input_path(data_values["eicu_path"], config_path.parent)

    data = _tuple_fields(
        DataConfig,
        data_values,
        (
            "split_group_columns",
            "sequence_group_columns",
            "external_sequence_group_columns",
            "exclude_columns",
            "feature_columns",
        ),
    )
    split = SplitConfig(**raw.get("split", {}))
    experiment = ExperimentConfig(**raw["experiment"])
    imputation = ImputationConfig(**raw.get("imputation", {}))
    config = Config(data=data, split=split, experiment=experiment, imputation=imputation)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    fractions = (
        config.split.train_fraction,
        config.split.validation_fraction,
        config.split.test_fraction,
    )
    if any(value <= 0 for value in fractions):
        raise ValueError("Train, validation, and test fractions must all be positive.")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0; received {sum(fractions):.12f}.")
    if config.experiment.search_iterations < 1:
        raise ValueError("search_iterations must be positive.")
    if config.experiment.bayesian_folds < 2:
        raise ValueError("bayesian_folds must be at least 2.")
    if config.experiment.threads_per_rank < 1:
        raise ValueError("threads_per_rank must be positive.")
    if config.data.expected_windows_per_admission < 1:
        raise ValueError("expected_windows_per_admission must be positive.")
    if config.data.max_los_inclusive is not None and config.data.max_los_inclusive <= 0:
        raise ValueError("max_los_inclusive must be positive or omitted.")
