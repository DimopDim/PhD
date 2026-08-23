#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_shap_concepts.py

Fold-ensemble concept-level SHAP for XGBoost.

Key properties
--------------
- For MIMIC test/eICU external, SHAP is averaged across the five fold models.
- SHAP is computed on the model's raw margin, preserving additivity before the
  mortality sigmoid/Platt calibration.
- Signed SHAP values are summed across:
    * resolution channels
    * within-window aggregation variants (mean/median/min/max)
    * temporal trajectory descriptors (last/mean/std/min/max/slope)
  to obtain one value per clinical concept.
- Demographics are grouped as Age, Gender, and Race.
- The beeswarm color score is the mean percentile rank of constituent model
  features within the explained cohort. It is used only for Low/High visual
  context; SHAP magnitude/sign remains exact after grouping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modeling_common import (
    group_feature_indices,
    load_json,
    load_model_matrix,
    matrix_cache_paths,
    task_dir,
)


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)


def get_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError(
            "xgboost is required."
        ) from exc
    return xgb


def rank_percentile_columns(
    X: np.ndarray,
) -> np.ndarray:
    """
    Column-wise percentile ranks, NaN preserving.
    """
    n, p = X.shape
    out = np.full(
        (n, p),
        np.nan,
        dtype=np.float32,
    )

    for j in range(p):
        x = X[:, j]
        finite = np.isfinite(
            x
        )
        if not finite.any():
            continue

        values = x[
            finite
        ]
        order = np.argsort(
            values,
            kind="mergesort",
        )
        ranks = np.empty(
            len(values),
            dtype=np.float64,
        )
        ranks[
            order
        ] = np.arange(
            len(values),
            dtype=np.float64,
        )

        if len(values) == 1:
            percentile = np.full(
                1,
                0.5,
            )
        else:
            percentile = (
                ranks
                / (
                    len(values)
                    - 1
                )
            )

        out[
            finite,
            j,
        ] = percentile.astype(
            np.float32
        )

    return out


def grouped_shap(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[str],
]:
    groups = group_feature_indices(
        feature_names
    )
    concepts = list(
        groups.keys()
    )

    percentile = rank_percentile_columns(
        X
    )

    grouped_phi = np.zeros(
        (
            len(X),
            len(concepts),
        ),
        dtype=np.float32,
    )
    grouped_color = np.full(
        (
            len(X),
            len(concepts),
        ),
        np.nan,
        dtype=np.float32,
    )

    for k, concept in enumerate(
        concepts
    ):
        idx = groups[
            concept
        ]
        grouped_phi[
            :,
            k,
        ] = np.sum(
            shap_values[
                :,
                idx,
            ],
            axis=1,
        )

        with np.errstate(
            invalid="ignore"
        ):
            grouped_color[
                :,
                k,
            ] = np.nanmean(
                percentile[
                    :,
                    idx,
                ],
                axis=1,
            )

    return (
        grouped_phi,
        grouped_color,
        concepts,
    )


def plot_grouped_beeswarm(
    grouped_phi: np.ndarray,
    grouped_color: np.ndarray,
    concepts: Sequence[str],
    *,
    top_n: int,
    title: str,
    output_path: Path,
    seed: int,
):
    importance = np.mean(
        np.abs(
            grouped_phi
        ),
        axis=0,
    )

    top_idx = np.argsort(
        importance
    )[
        -top_n:
    ]

    # Lowest displayed importance at top; highest at bottom is less intuitive
    # for SHAP. Reverse so most important appears at top.
    top_idx = top_idx[
        ::-1
    ]

    rng = np.random.default_rng(
        seed
    )

    fig_height = max(
        5.5,
        0.42 * len(
            top_idx
        ) + 1.5,
    )
    fig, ax = plt.subplots(
        figsize=(8, fig_height)
    )

    last_scatter = None

    for row, idx in enumerate(
        top_idx
    ):
        phi = grouped_phi[
            :,
            idx,
        ]
        color = grouped_color[
            :,
            idx,
        ]

        finite = np.isfinite(
            phi
        )
        phi = phi[
            finite
        ]
        color = color[
            finite
        ]

        jitter = rng.normal(
            loc=0.0,
            scale=0.08,
            size=len(
                phi
            ),
        )

        last_scatter = ax.scatter(
            phi,
            np.full(
                len(phi),
                row,
                dtype=float,
            ) + jitter,
            c=color,
            s=13,
            alpha=0.70,
        )

    ax.axvline(
        0.0,
        linewidth=1.0,
    )
    ax.set_yticks(
        np.arange(
            len(top_idx)
        )
    )
    ax.set_yticklabels(
        [
            concepts[i]
            for i in top_idx
        ]
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Grouped SHAP value"
    )
    ax.set_title(
        title
    )
    ax.grid(
        axis="x",
        alpha=0.20,
    )

    if last_scatter is not None:
        cbar = fig.colorbar(
            last_scatter,
            ax=ax,
            pad=0.02,
        )
        cbar.set_label(
            "Relative feature value (Low → High)"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_grouped_bar(
    grouped_phi: np.ndarray,
    concepts: Sequence[str],
    *,
    top_n: int,
    title: str,
    output_path: Path,
):
    importance = np.mean(
        np.abs(
            grouped_phi
        ),
        axis=0,
    )

    idx = np.argsort(
        importance
    )[
        -top_n:
    ]

    fig_height = max(
        5.0,
        0.38 * len(
            idx
        ) + 1.5,
    )
    fig, ax = plt.subplots(
        figsize=(7, fig_height)
    )

    ax.barh(
        np.arange(
            len(idx)
        ),
        importance[
            idx
        ],
    )
    ax.set_yticks(
        np.arange(
            len(idx)
        )
    )
    ax.set_yticklabels(
        [
            concepts[i]
            for i in idx
        ]
    )
    ax.set_xlabel(
        "Mean |grouped SHAP value|"
    )
    ax.set_title(
        title
    )
    ax.grid(
        axis="x",
        alpha=0.20,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def explain_task(
    *,
    modeling_root: Path,
    landmark: int,
    outcome: str,
    variant: str,
    cohort: str,
    max_rows: int,
    top_n: int,
    seed: int,
):
    xgb = get_xgboost()

    if cohort == "mimic_test":
        database = "mimic"
        split_name = "test"
    elif cohort == "eicu_external":
        database = "eicu"
        split_name = "external"
    else:
        raise ValueError(
            "SHAP cohort must be mimic_test or eicu_external."
        )

    matrix_path, feature_path = (
        matrix_cache_paths(
            modeling_root,
            landmark,
            variant,
            database,
            split_name,
        )
    )
    matrix = load_model_matrix(
        matrix_path,
        feature_path,
    )

    if outcome == "mortality":
        mask = matrix.mortality_known
    else:
        mask = np.ones(
            len(
                matrix.patient_id
            ),
            dtype=bool,
        )

    X = matrix.X[
        mask
    ]
    patient_id = matrix.patient_id[
        mask
    ]

    if len(X) > max_rows:
        rng = np.random.default_rng(
            seed
        )
        selected = np.sort(
            rng.choice(
                len(X),
                size=max_rows,
                replace=False,
            )
        )
        X = X[
            selected
        ]
        patient_id = patient_id[
            selected
        ]

    task_output = task_dir(
        modeling_root,
        landmark,
        outcome,
        variant,
    )
    model_paths = sorted(
        (
            task_output
            / "models"
        ).glob(
            "fold_*.json"
        )
    )
    if len(model_paths) != 5:
        raise FileNotFoundError(
            f"Expected five fold models under {task_output / 'models'}"
        )

    dmatrix = xgb.DMatrix(
        X,
        feature_names=(
            matrix.feature_names
        ),
        missing=np.nan,
    )

    fold_phi = []

    for model_path in model_paths:
        model = xgb.Booster()
        model.load_model(
            model_path
        )

        contrib = model.predict(
            dmatrix,
            pred_contribs=True,
            approx_contribs=False,
        )

        if contrib.shape[1] != (
            X.shape[1] + 1
        ):
            raise RuntimeError(
                f"Unexpected SHAP contribution width from {model_path}."
            )

        fold_phi.append(
            contrib[
                :,
                :-1,
            ].astype(
                np.float32
            )
        )

    shap_values = np.mean(
        np.stack(
            fold_phi,
            axis=0,
        ),
        axis=0,
    ).astype(
        np.float32
    )

    (
        grouped_phi,
        grouped_color,
        concepts,
    ) = grouped_shap(
        shap_values,
        X,
        matrix.feature_names,
    )

    figure_dir = (
        modeling_root
        / "figures"
        / "shap"
        / f"landmark_{landmark:03d}h"
        / outcome
        / variant
        / cohort
    )

    title = (
        f"{outcome.upper()} | {landmark}h | "
        f"{variant} | {cohort.replace('_', ' ')}"
    )

    plot_grouped_beeswarm(
        grouped_phi,
        grouped_color,
        concepts,
        top_n=top_n,
        title=title,
        output_path=(
            figure_dir
            / "shap_concept_beeswarm.png"
        ),
        seed=seed,
    )

    plot_grouped_bar(
        grouped_phi,
        concepts,
        top_n=top_n,
        title=title,
        output_path=(
            figure_dir
            / "shap_concept_bar.png"
        ),
    )

    importance = np.mean(
        np.abs(
            grouped_phi
        ),
        axis=0,
    )
    importance_df = pd.DataFrame(
        {
            "concept": concepts,
            "mean_abs_shap": (
                importance
            ),
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    )
    importance_df.to_csv(
        figure_dir
        / "shap_concept_importance.csv",
        index=False,
    )

    np.savez_compressed(
        figure_dir
        / "grouped_shap_values.npz",
        grouped_shap=grouped_phi,
        grouped_color_score=(
            grouped_color
        ),
        patient_id=patient_id,
        concept=np.asarray(
            concepts,
            dtype=str,
        ),
    )

    print(
        f"SHAP PASS | {title} | rows={len(X)}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--modeling-root",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--landmarks",
        type=str,
        default="4,8,12,16,20,24,36,48",
    )
    p.add_argument(
        "--outcomes",
        type=str,
        default="los,mortality",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="full",
    )
    p.add_argument(
        "--cohorts",
        type=str,
        default="mimic_test,eicu_external",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=2000,
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=20,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = p.parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    modeling_root = (
        args.modeling_root
        .expanduser()
        .resolve()
        if args.modeling_root is not None
        else project_root
        / "01_Modeling"
    )

    landmarks = [
        int(x.strip())
        for x in args.landmarks.split(
            ","
        )
        if x.strip()
    ]
    outcomes = [
        x.strip()
        for x in args.outcomes.split(
            ","
        )
        if x.strip()
    ]
    variants = [
        x.strip()
        for x in args.variants.split(
            ","
        )
        if x.strip()
    ]
    cohorts = [
        x.strip()
        for x in args.cohorts.split(
            ","
        )
        if x.strip()
    ]

    for landmark in landmarks:
        for outcome in outcomes:
            for variant in variants:
                task_output = task_dir(
                    modeling_root,
                    landmark,
                    outcome,
                    variant,
                )
                if not (
                    task_output
                    / "task_manifest.json"
                ).is_file():
                    print(
                        f"SKIP missing task: "
                        f"{landmark}h/{outcome}/{variant}"
                    )
                    continue

                for cohort in cohorts:
                    explain_task(
                        modeling_root=(
                            modeling_root
                        ),
                        landmark=landmark,
                        outcome=outcome,
                        variant=variant,
                        cohort=cohort,
                        max_rows=args.max_rows,
                        top_n=args.top_n,
                        seed=args.seed,
                    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
