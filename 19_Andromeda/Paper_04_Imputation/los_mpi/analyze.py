from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from xgboost import XGBRegressor

from .config import load_config
from .data import load_and_split
from .experiment import collect_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create provenance-linked figures from a completed run.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--configuration", required=True, help="Exact configuration id.")
    parser.add_argument(
        "--shap-dataset",
        choices=["mimic_test", "eicu_external"],
        default="mimic_test",
    )
    parser.add_argument("--shap-max-rows", type=int, default=2000)
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--skip-shap", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _configuration_parts(configuration: str) -> tuple[str, bool]:
    match = re.fullmatch(
        r"(native|mean|interpolation|mlp|rnn|regression)__scale-(yes|no)__(none|randomized|bayesian|hyperopt)",
        configuration,
    )
    if not match:
        raise ValueError(f"Invalid configuration id: {configuration}")
    return match.group(1), match.group(2) == "yes"


def residual_figure(configuration_dir: Path, output: Path) -> None:
    datasets = ["mimic_test", "eicu_external"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for axis, dataset in zip(axes, datasets):
        frame = pd.read_parquet(configuration_dir / f"predictions_{dataset}.parquet")
        mae = float(np.mean(np.abs(frame.residual_true_minus_predicted)))
        axis.scatter(
            frame.y_true,
            frame.residual_true_minus_predicted,
            s=12,
            alpha=0.28,
            color="#1764d8",
            edgecolors="none",
        )
        axis.axhline(0, color="#c62828", linestyle="--", linewidth=1.4, label="Zero residual")
        axis.axhline(mae, color="#188038", linestyle="--", linewidth=1.2, label=f"±MAE ({mae:.2f} d)")
        axis.axhline(-mae, color="#188038", linestyle="--", linewidth=1.2)
        axis.set(title=dataset.replace("_", " "), xlabel="True ICU LOS (days)", ylabel="True − predicted LOS (days)")
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
    fig.suptitle(f"Residuals from the same fitted configuration\n{configuration_dir.name}")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def metrics_figure(metrics: pd.DataFrame, output: Path) -> None:
    selected = metrics[metrics.dataset.isin(["mimic_test", "eicu_external"])].copy()
    selected["label"] = (
        selected.imputation.str.slice(0, 4)
        + " | S=" + selected.scaling.map({True: "Y", False: "N"})
        + " | " + selected.tuning.str.slice(0, 4)
    )
    order = (
        selected[selected.dataset == "mimic_test"]
        .sort_values("mae", kind="stable").label.tolist()
    )
    pivot = selected.pivot(index="label", columns="dataset", values="mae").loc[order]
    fig, axis = plt.subplots(figsize=(8.5, 12), constrained_layout=True)
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", linewidths=0.4, ax=axis)
    axis.set(title="MAE across all 48 leakage-safe configurations", xlabel="Evaluation dataset", ylabel="Imputation | Scaling | Tuning")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


SHAP_STATISTICS = ("min", "median", "mean", "max")


def _statistic_suffix(name: str) -> tuple[str, str] | None:
    """Return (base_feature, statistic) for final Min/Median/Mean/Max suffixes.

    Supports the naming used by the current LOS exports, e.g. ``pH_(Mean)`` and
    ``Albumin_(Max)``, as well as legacy ``pH_mean`` / ``pH_max`` names.
    Only the final statistic token is removed, so parentheses inside the
    clinical concept itself are preserved.
    """
    value = str(name).strip()
    patterns = (
        r"(?i)^(?P<base>.+?)(?:_)?\((?P<stat>min|median|mean|max)\)$",
        r"(?i)^(?P<base>.+?)_(?P<stat>min|median|mean|max)$",
    )
    for pattern in patterns:
        match = re.match(pattern, value)
        if match:
            base = match.group("base").rstrip("_ ")
            if base:
                return base, match.group("stat").lower()
    return None


def aggregate_shap_statistic_features(
    shap_values: np.ndarray,
    feature_data: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[list[str]]]:
    """Group Min/Median/Mean/Max siblings for a SHAP beeswarm.

    Signed SHAP values are summed row-by-row across sibling statistics.  This
    preserves SHAP local additivity.  The row-wise finite mean of the sibling
    feature values is used only for the beeswarm colour scale.

    A suffix is removed only if at least two distinct statistic siblings are
    present.  This avoids accidentally renaming a legitimate standalone feature
    whose clinical name happens to end in ``_mean`` or similar.
    """
    shap_values = np.asarray(shap_values)
    feature_data = np.asarray(feature_data)
    names = [str(name) for name in feature_names]

    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP matrix, got {shap_values.shape}")
    if shap_values.shape != feature_data.shape:
        raise ValueError(
            "SHAP values and feature data must have identical shapes: "
            f"{shap_values.shape} vs {feature_data.shape}"
        )
    if shap_values.shape[1] != len(names):
        raise ValueError(
            "Feature-name count does not match SHAP columns: "
            f"{len(names)} vs {shap_values.shape[1]}"
        )

    candidates: dict[str, list[int]] = {}
    statistic_by_index: dict[int, tuple[str, str]] = {}
    for index, name in enumerate(names):
        parsed = _statistic_suffix(name)
        if parsed is None:
            continue
        base, statistic = parsed
        candidates.setdefault(base, []).append(index)
        statistic_by_index[index] = (base, statistic)

    grouped_bases = {
        base
        for base, indices in candidates.items()
        if len({statistic_by_index[index][1] for index in indices}) >= 2
    }

    statistic_order = {name: i for i, name in enumerate(SHAP_STATISTICS)}
    group_indices: list[list[int]] = []
    group_names: list[str] = []
    emitted: set[str] = set()

    for index, name in enumerate(names):
        parsed = statistic_by_index.get(index)
        if parsed is None or parsed[0] not in grouped_bases:
            group_indices.append([index])
            group_names.append(name)
            continue

        base = parsed[0]
        if base in emitted:
            continue
        emitted.add(base)
        members = sorted(
            candidates[base],
            key=lambda i: statistic_order[statistic_by_index[i][1]],
        )
        group_indices.append(members)
        group_names.append(base)

    # Float64 accumulation minimises rounding drift when contributions are summed.
    raw_shap = np.asarray(shap_values, dtype=np.float64)
    grouped_shap = np.empty((raw_shap.shape[0], len(group_indices)), dtype=np.float64)
    grouped_data = np.empty((feature_data.shape[0], len(group_indices)), dtype=np.float64)
    components: list[list[str]] = []

    for out_index, members in enumerate(group_indices):
        grouped_shap[:, out_index] = raw_shap[:, members].sum(axis=1)

        values = np.asarray(feature_data[:, members], dtype=np.float64)
        finite = np.isfinite(values)
        count = finite.sum(axis=1)
        total = np.where(finite, values, 0.0).sum(axis=1)
        grouped_data[:, out_index] = np.divide(
            total,
            count,
            out=np.full(values.shape[0], np.nan, dtype=np.float64),
            where=count > 0,
        )
        components.append([names[i] for i in members])

    # Grouping must preserve the sum of signed SHAP contributions for every row.
    if not np.allclose(
        grouped_shap.sum(axis=1),
        raw_shap.sum(axis=1),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError("Grouped SHAP additivity check failed")

    return grouped_shap, grouped_data, group_names, components


def _pretty_feature_name(name: str) -> str:
    """Make a clinical feature label readable without changing its meaning."""
    return str(name).replace("_", " ").strip()


def grouped_shap_beeswarm(
    grouped_shap: np.ndarray,
    grouped_data: np.ndarray,
    grouped_names: list[str],
    output: Path,
    title: str,
    top_features: int,
) -> None:
    """Render a true SHAP beeswarm after statistic-level aggregation."""
    mean_abs = np.mean(np.abs(grouped_shap), axis=0)
    display_count = min(top_features, grouped_shap.shape[1])
    top_indices = np.argsort(mean_abs)[::-1][:display_count]

    # Pass only the selected concepts to SHAP.  Because the Explanation contains
    # exactly max_display columns, SHAP does not create a 'Sum of other features'
    # row in the beeswarm.
    explanation = shap.Explanation(
        values=grouped_shap[:, top_indices],
        data=grouped_data[:, top_indices],
        feature_names=[_pretty_feature_name(grouped_names[i]) for i in top_indices],
    )

    colour_map = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
        "shap_publication", ["#2171b5", "#d3d3d3", "#cb181d"]
    )

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    ):
        fig, _ = plt.subplots(figsize=(8.8, max(5.2, 0.36 * display_count + 1.4)))
        shap.plots.beeswarm(
            explanation,
            max_display=display_count,
            show=False,
            color=colour_map,
            plot_size=None,
        )
        axis = plt.gca()
        axis.set_xlabel("SHAP value (impact on model output)")
        axis.set_ylabel("")
        axis.set_title(title, pad=10)
        for candidate in fig.axes:
            if candidate is not axis:
                candidate.set_ylabel("Feature value")
        fig.tight_layout()
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)

def shap_figures(
    config,
    configuration: str,
    configuration_dir: Path,
    figure_dir: Path,
    dataset_name: str,
    max_rows: int,
    top_features: int,
) -> dict:
    method, scale = _configuration_parts(configuration)
    bundle_id = f"preprocess__{method}__scale-{'yes' if scale else 'no'}"
    root = Path(config.experiment.output_dir).expanduser().resolve()
    preprocessor = joblib.load(root / "bundles" / bundle_id / "preprocessor.joblib")

    # Use exactly the same split and fitted preprocessing bundle as training.
    data = load_and_split(config)
    frame = data.test if dataset_name == "mimic_test" else data.external
    X = preprocessor.transform(frame)

    model = XGBRegressor()
    model.load_model(configuration_dir / "model.json")

    # Refuse SHAP output if the reconstructed matrix does not replay the saved
    # predictions from the exact fitted configuration.
    saved = pd.read_parquet(configuration_dir / f"predictions_{dataset_name}.parquet")
    replay = model.predict(X)
    max_abs_error = float(np.max(np.abs(replay - saved.y_pred.to_numpy())))
    if max_abs_error > 1e-5:
        raise RuntimeError(f"Saved-prediction replay failed: max_abs_error={max_abs_error}")

    rng = np.random.default_rng(config.split.seed)
    if len(X) > max_rows:
        indices = np.sort(rng.choice(len(X), size=max_rows, replace=False))
    else:
        indices = np.arange(len(X))

    sample = np.asarray(X[indices], dtype=float)
    feature_names = [str(name) for name in preprocessor.feature_names_out_]

    # Exact TreeSHAP for the selected rows and the fitted XGBoost model.
    explainer = shap.TreeExplainer(model)
    explanation = explainer(sample)
    raw_shap = np.asarray(explanation.values, dtype=float)

    # Collapse e.g. pH_(Min), pH_(Median), pH_(Mean), pH_(Max) into one pH row.
    # Signed contributions are summed per observation, not averaged.
    (
        grouped_shap,
        grouped_data,
        grouped_names,
        grouped_components,
    ) = aggregate_shap_statistic_features(
        raw_shap,
        sample,
        feature_names,
    )

    grouped_shap_beeswarm(
        grouped_shap,
        grouped_data,
        grouped_names,
        figure_dir / "shap_beeswarm.png",
        title=f"{configuration} | {dataset_name} | n={len(indices)}",
        top_features=top_features,
    )

    # Save both original and grouped importance tables for complete auditability.
    individual_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": np.abs(raw_shap).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    individual_importance.to_csv(figure_dir / "shap_importance.csv", index=False)

    grouped_importance = pd.DataFrame(
        {
            "feature_concept": grouped_names,
            "mean_abs_shap": np.abs(grouped_shap).mean(axis=0),
            "component_count": [len(items) for items in grouped_components],
            "components": [" | ".join(items) for items in grouped_components],
        }
    ).sort_values("mean_abs_shap", ascending=False)
    grouped_importance.to_csv(
        figure_dir / "shap_grouped_importance.csv", index=False
    )

    # Publication-style grouped mean-|SHAP| bar plot using the same concepts.
    bar = grouped_importance.head(top_features).sort_values("mean_abs_shap")
    fig, axis = plt.subplots(
        figsize=(8.5, max(5.0, 0.30 * len(bar))), constrained_layout=True
    )
    axis.barh(
        [_pretty_feature_name(value) for value in bar.feature_concept],
        bar.mean_abs_shap,
        color="#1764d8",
    )
    axis.set(
        xlabel="Mean |grouped SHAP value|",
        ylabel="Feature concept",
        title=f"Grouped SHAP importance | {dataset_name}",
    )
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(
        figure_dir / "shap_grouped_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    grouped_count = sum(len(items) > 1 for items in grouped_components)
    return {
        "dataset": dataset_name,
        "sample_rows": int(len(indices)),
        "original_feature_count": int(raw_shap.shape[1]),
        "grouped_feature_count": int(grouped_shap.shape[1]),
        "statistic_groups_collapsed": int(grouped_count),
        "aggregation_rule": "row-wise sum of signed SHAP across Min/Median/Mean/Max siblings",
        "colour_value_rule": "row-wise finite mean of sibling feature values",
        "sample_row_ids_sha256": hashlib.sha256(
            "\0".join(frame.row_ids[indices]).encode()
        ).hexdigest(),
        "prediction_replay_max_abs_error": max_abs_error,
    }

def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = Path(config.experiment.output_dir).expanduser().resolve()
    configuration_dir = root / "configurations" / args.configuration
    if not (configuration_dir / "DONE.json").exists():
        raise FileNotFoundError(f"Configuration is incomplete: {configuration_dir}")
    figure_dir = root / "figures" / args.configuration
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics = collect_results(config)
    residual_figure(configuration_dir, figure_dir / "residuals_internal_external.png")
    metrics_figure(metrics, figure_dir / "all_48_mae_heatmap.png")
    shap_audit = None
    if not args.skip_shap:
        shap_audit = shap_figures(
            config,
            args.configuration,
            configuration_dir,
            figure_dir,
            args.shap_dataset,
            args.shap_max_rows,
            args.top_features,
        )
    manifest = {
        "configuration": args.configuration,
        "model_sha256": _sha256(configuration_dir / "model.json"),
        "residual_datasets": ["mimic_test", "eicu_external"],
        "shap": shap_audit,
    }
    (figure_dir / "figure_provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(figure_dir)


if __name__ == "__main__":
    main()

