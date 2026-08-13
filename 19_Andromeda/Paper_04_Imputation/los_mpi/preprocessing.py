from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from .config import Config
from .data import FrameSet


class NumericImputer(Protocol):
    def fit(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> "NumericImputer": ...
    def transform(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> np.ndarray: ...


class NativeImputer:
    def fit(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> "NativeImputer":
        return self

    def transform(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float).copy()


class SklearnImputerAdapter:
    def __init__(self, estimator: object):
        self.estimator = estimator

    def fit(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> "SklearnImputerAdapter":
        self.estimator.fit(X)
        return self

    def transform(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.transform(X), dtype=float)


class CausalInterpolationImputer:
    """Within-admission interpolation/extrapolation using current and earlier windows only."""

    def fit(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> "CausalInterpolationImputer":
        X = np.asarray(X, dtype=float)
        self.fallback_ = np.nanmedian(X, axis=0)
        self.fallback_ = np.where(np.isfinite(self.fallback_), self.fallback_, 0.0)
        self.lower_ = np.nanquantile(X, 0.005, axis=0)
        self.upper_ = np.nanquantile(X, 0.995, axis=0)
        self.lower_ = np.where(np.isfinite(self.lower_), self.lower_, self.fallback_)
        self.upper_ = np.where(np.isfinite(self.upper_), self.upper_, self.fallback_)
        return self

    def transform(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> np.ndarray:
        result = np.asarray(X, dtype=float).copy()
        group_series = pd.Series(groups)
        for group in group_series.drop_duplicates().tolist():
            indices = np.flatnonzero(groups == group)
            indices = indices[np.argsort(order[indices], kind="stable")]
            for feature in range(result.shape[1]):
                observed_positions: list[float] = []
                observed_values: list[float] = []
                for index in indices:
                    value = result[index, feature]
                    position = float(order[index])
                    if np.isfinite(value):
                        observed_positions.append(position)
                        observed_values.append(value)
                        continue
                    if len(observed_values) >= 2:
                        x1, x2 = observed_positions[-2:]
                        y1, y2 = observed_values[-2:]
                        slope = 0.0 if x2 == x1 else (y2 - y1) / (x2 - x1)
                        value = y2 + slope * (position - x2)
                        value = float(np.clip(value, self.lower_[feature], self.upper_[feature]))
                    elif len(observed_values) == 1:
                        value = observed_values[-1]
                    else:
                        value = self.fallback_[feature]
                    result[index, feature] = value
        return result


def _seed_everything(seed: int, threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        pass


class _MLPNetwork:
    @staticmethod
    def build(n_features: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(n_features * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_features),
        )


def _build_rnn_network(n_features: int):
    import torch.nn as nn

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(n_features * 2, 64, batch_first=True, bidirectional=False)
            self.hidden = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3))
            self.output = nn.Linear(32, n_features)

        def forward(self, values):
            encoded, _ = self.lstm(values)
            return self.output(self.hidden(encoded))

    return Network()


@dataclass
class NeuralOptions:
    epochs: int
    batch_size: int
    patience: int
    validation_fraction: float
    corruption_fraction: float
    seed: int
    threads: int
    device: str


class NeuralDenoisingImputer:
    def __init__(self, architecture: str, options: NeuralOptions):
        self.architecture = architecture
        self.options = options

    def _device(self) -> str:
        import torch

        requested = self.options.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested for neural imputation but is unavailable: {requested}")
        return requested

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.median_ = np.nanmedian(X, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        self.mean_ = np.nanmean(X, axis=0)
        self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, self.median_)
        self.std_ = np.nanstd(X, axis=0)
        self.std_ = np.where(np.isfinite(self.std_) & (self.std_ > 1e-8), self.std_, 1.0)
        return (X - self.mean_) / self.std_

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def _row_train_validation(self, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        unique = np.unique(groups)
        rng = np.random.default_rng(self.options.seed)
        shuffled = rng.permutation(unique)
        n_val = max(1, int(round(len(unique) * self.options.validation_fraction)))
        val_groups = set(shuffled[:n_val])
        is_val = np.fromiter((group in val_groups for group in groups), dtype=bool)
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

    @staticmethod
    def _corrupt(values, observed, fraction: float, generator):
        import torch

        artificial = (torch.rand(values.shape, generator=generator, device=values.device) < fraction) & observed
        # Guarantee a usable loss mask for very sparse mini-batches.
        if not artificial.any() and observed.any():
            location = torch.nonzero(observed, as_tuple=False)[0]
            artificial[tuple(location)] = True
        visible = observed & ~artificial
        inputs = torch.where(visible, values, torch.zeros_like(values))
        return torch.cat([inputs, visible.float()], dim=-1), artificial

    def fit(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> "NeuralDenoisingImputer":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        _seed_everything(self.options.seed, self.options.threads)
        raw = np.asarray(X, dtype=float)
        standardized = self._standardize_fit(raw)
        observed = np.isfinite(standardized)
        filled = np.where(observed, standardized, 0.0).astype(np.float32)
        device = self._device()
        n_features = filled.shape[1]
        self.model_ = (
            _MLPNetwork.build(n_features)
            if self.architecture == "mlp"
            else _build_rnn_network(n_features)
        ).to(device)
        optimizer = torch.optim.Adam(self.model_.parameters())
        generator = torch.Generator(device=device).manual_seed(self.options.seed)

        if self.architecture == "mlp":
            train_idx, validation_idx = self._row_train_validation(groups)
            train_dataset = TensorDataset(
                torch.from_numpy(filled[train_idx]), torch.from_numpy(observed[train_idx])
            )
            validation_data = (
                torch.from_numpy(filled[validation_idx]).to(device),
                torch.from_numpy(observed[validation_idx]).to(device),
            )
            loader = DataLoader(
                train_dataset,
                batch_size=self.options.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.options.seed),
            )
        else:
            train_data, validation_data = self._sequence_tensors(filled, observed, groups, order, fit=True)
            train_values, train_observed, train_valid = train_data
            train_dataset = TensorDataset(train_values, train_observed, train_valid)
            loader = DataLoader(
                train_dataset,
                batch_size=self.options.batch_size,
                shuffle=True,
                generator=torch.Generator().manual_seed(self.options.seed),
            )

        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale = 0
        for _epoch in range(self.options.epochs):
            self.model_.train()
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                values = batch[0].to(device)
                obs = batch[1].to(device).bool()
                valid = batch[2].to(device).bool() if self.architecture == "rnn" else None
                network_input, loss_mask = self._corrupt(
                    values, obs, self.options.corruption_fraction, generator
                )
                if valid is not None:
                    loss_mask &= valid.unsqueeze(-1)
                prediction = self.model_(network_input)
                loss = ((prediction - values) ** 2)[loss_mask].mean()
                loss.backward()
                optimizer.step()

            val_loss = self._validation_loss(validation_data, generator, device)
            if val_loss < best_loss - 1e-7:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.options.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.model_.to("cpu")
        self.n_features_ = n_features
        self.best_validation_loss_ = best_loss
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        model = state.pop("model_", None)
        if model is not None:
            state["model_state_"] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model_state = self.__dict__.pop("model_state_", None)
        if model_state is not None:
            self.model_ = (
                _MLPNetwork.build(self.n_features_)
                if self.architecture == "mlp"
                else _build_rnn_network(self.n_features_)
            )
            self.model_.load_state_dict(model_state)
            self.model_.to("cpu")
            self.model_.eval()

    def _sequence_tensors(self, values, observed, groups, order, fit: bool):
        import torch

        unique = np.unique(groups)
        if fit:
            rng = np.random.default_rng(self.options.seed)
            shuffled = rng.permutation(unique)
            n_val = max(1, int(round(len(unique) * self.options.validation_fraction)))
            validation_groups = set(shuffled[:n_val])
            partitions = [
                [group for group in unique if group not in validation_groups],
                [group for group in unique if group in validation_groups],
            ]
        else:
            partitions = [list(unique)]
        output = []
        for selected in partitions:
            max_length = max(np.sum(groups == group) for group in selected)
            value_tensor = np.zeros((len(selected), max_length, values.shape[1]), dtype=np.float32)
            observed_tensor = np.zeros_like(value_tensor, dtype=bool)
            valid_tensor = np.zeros((len(selected), max_length), dtype=bool)
            for row, group in enumerate(selected):
                indices = np.flatnonzero(groups == group)
                indices = indices[np.argsort(order[indices], kind="stable")]
                length = len(indices)
                value_tensor[row, :length] = values[indices]
                observed_tensor[row, :length] = observed[indices]
                valid_tensor[row, :length] = True
            output.append(
                (
                    torch.from_numpy(value_tensor),
                    torch.from_numpy(observed_tensor),
                    torch.from_numpy(valid_tensor),
                )
            )
        return output if fit else output[0]

    def _validation_loss(self, data, generator, device: str) -> float:
        import torch

        self.model_.eval()
        with torch.no_grad():
            values = data[0].to(device)
            observed = data[1].to(device).bool()
            valid = data[2].to(device).bool() if self.architecture == "rnn" else None
            network_input, loss_mask = self._corrupt(
                values, observed, self.options.corruption_fraction, generator
            )
            if valid is not None:
                loss_mask &= valid.unsqueeze(-1)
            prediction = self.model_(network_input)
            return float(((prediction - values) ** 2)[loss_mask].mean().cpu())

    def transform(self, X: np.ndarray, groups: np.ndarray, order: np.ndarray) -> np.ndarray:
        import torch

        raw = np.asarray(X, dtype=float)
        standardized = self._standardize(raw)
        observed = np.isfinite(standardized)
        filled = np.where(observed, standardized, 0.0).astype(np.float32)
        self.model_.eval()
        with torch.no_grad():
            if self.architecture == "mlp":
                values = torch.from_numpy(filled)
                obs = torch.from_numpy(observed)
                network_input = torch.cat([values, obs.float()], dim=-1)
                predictions = self.model_(network_input).numpy()
            else:
                predictions = np.zeros_like(filled)
                for group in np.unique(groups):
                    indices = np.flatnonzero(groups == group)
                    sorted_indices = indices[np.argsort(order[indices], kind="stable")]
                    values = torch.from_numpy(filled[sorted_indices]).unsqueeze(0)
                    obs = torch.from_numpy(observed[sorted_indices]).unsqueeze(0)
                    network_input = torch.cat([values, obs.float()], dim=-1)
                    predictions[sorted_indices] = self.model_(network_input).squeeze(0).numpy()
        completed = np.where(observed, standardized, predictions)
        return completed * self.std_ + self.mean_


def _numeric_imputer(method: str, config: Config, seed: int, threads: int) -> NumericImputer:
    if method == "native":
        return NativeImputer()
    if method == "mean":
        return SklearnImputerAdapter(SimpleImputer(strategy="mean", keep_empty_features=True))
    if method == "regression":
        estimator = ExtraTreesRegressor(
            n_estimators=config.imputation.iterative_trees,
            random_state=seed,
            n_jobs=threads,
        )
        return SklearnImputerAdapter(
            IterativeImputer(
                estimator=estimator,
                max_iter=config.imputation.iterative_max_iter,
                random_state=seed,
                skip_complete=True,
                keep_empty_features=True,
            )
        )
    if method == "interpolation":
        return CausalInterpolationImputer()
    if method in {"mlp", "rnn"}:
        options = NeuralOptions(
            epochs=config.imputation.neural_epochs,
            batch_size=config.imputation.neural_batch_size,
            patience=config.imputation.neural_patience,
            validation_fraction=config.imputation.neural_validation_fraction,
            corruption_fraction=config.imputation.neural_corruption_fraction,
            seed=seed,
            threads=threads,
            device=config.experiment.device,
        )
        return NeuralDenoisingImputer(method, options)
    raise ValueError(f"Unknown imputation method: {method}")


class LeakageSafePreprocessor:
    def __init__(self, method: str, scale: bool, config: Config, seed: int, threads: int):
        self.method = method
        self.scale = scale
        self.config = config
        self.seed = seed
        self.threads = threads

    def fit(self, frame: FrameSet) -> "LeakageSafePreprocessor":
        self.raw_columns_ = list(frame.X.columns)
        self.numeric_columns_ = [
            column
            for column in self.raw_columns_
            if pd.api.types.is_numeric_dtype(frame.X[column]) or pd.api.types.is_bool_dtype(frame.X[column])
        ]
        self.categorical_columns_ = [column for column in self.raw_columns_ if column not in self.numeric_columns_]
        numeric = frame.X[self.numeric_columns_].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        self.numeric_imputer_ = _numeric_imputer(self.method, self.config, self.seed, self.threads)
        self.numeric_imputer_.fit(numeric, frame.sequence_groups, frame.window_order)
        numeric_transformed = self.numeric_imputer_.transform(
            numeric, frame.sequence_groups, frame.window_order
        )

        if self.categorical_columns_:
            self.categorical_ = Pipeline(
                [
                    ("missing", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                    ),
                ]
            )
            categorical = self.categorical_.fit_transform(frame.X[self.categorical_columns_])
            cat_names = list(
                self.categorical_.named_steps["onehot"].get_feature_names_out(self.categorical_columns_)
            )
        else:
            self.categorical_ = None
            categorical = np.empty((len(frame.X), 0), dtype=float)
            cat_names = []
        combined = np.column_stack([numeric_transformed, categorical])
        self.feature_names_out_ = [*self.numeric_columns_, *cat_names]

        if self.scale:
            # This exact order reproduces the requested Scale/Norm=Yes definition.
            self.standard_scaler_ = StandardScaler()
            standardized = self.standard_scaler_.fit_transform(combined)
            self.minmax_scaler_ = MinMaxScaler()
            self.minmax_scaler_.fit(standardized)
        else:
            self.standard_scaler_ = None
            self.minmax_scaler_ = None
        self.fit_scope_ = {
            "dataset": frame.name,
            "row_count": len(frame.X),
            "split_group_count": int(pd.Series(frame.split_groups).nunique()),
            "sequence_group_count": int(pd.Series(frame.sequence_groups).nunique()),
            "row_id_sha256": _hash_strings(frame.row_ids),
        }
        return self

    def transform(self, frame: FrameSet) -> np.ndarray:
        if list(frame.X.columns) != self.raw_columns_:
            raise ValueError(f"{frame.name}: raw feature order differs from fitted schema")
        numeric = frame.X[self.numeric_columns_].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        numeric = self.numeric_imputer_.transform(numeric, frame.sequence_groups, frame.window_order)
        if self.categorical_ is not None:
            categorical = self.categorical_.transform(frame.X[self.categorical_columns_])
        else:
            categorical = np.empty((len(frame.X), 0), dtype=float)
        combined = np.column_stack([numeric, categorical])
        if self.scale:
            combined = self.standard_scaler_.transform(combined)
            combined = self.minmax_scaler_.transform(combined)
        return np.asarray(combined, dtype=np.float32)

    def fit_transform(self, frame: FrameSet) -> np.ndarray:
        self.fit(frame)
        return self.transform(frame)


def _hash_strings(values: np.ndarray) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
