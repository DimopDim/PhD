from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
        "n_rows": int(len(y_true)),
    }


def cluster_bootstrap_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if iterations <= 0:
        return {"mae_ci_low": float("nan"), "mae_ci_high": float("nan")}
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    scores = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        scores[iteration] = mean_absolute_error(y_true[selected], y_pred[selected])
    low, high = np.quantile(scores, [0.025, 0.975])
    return {"mae_ci_low": float(low), "mae_ci_high": float(high)}


def prediction_frame(
    dataset: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sequence_groups: np.ndarray,
    split_groups: np.ndarray,
    window_order: np.ndarray,
    row_ids: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "row_id": row_ids,
            "split_group": split_groups,
            "sequence_group": sequence_groups,
            "window_order": window_order,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual_true_minus_predicted": y_true - y_pred,
        }
    )


def admission_aggregations(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    ordered = predictions.sort_values(["sequence_group", "window_order"], kind="stable")
    mean_frame = ordered.groupby("sequence_group", as_index=False).agg(
        y_true=("y_true", "first"), y_pred=("y_pred", "mean")
    )
    final_frame = ordered.groupby("sequence_group", as_index=False).tail(1)
    return {
        "admission_mean_prediction": regression_metrics(
            mean_frame.y_true.to_numpy(), mean_frame.y_pred.to_numpy()
        ),
        "final_window": regression_metrics(
            final_frame.y_true.to_numpy(), final_frame.y_pred.to_numpy()
        ),
    }

