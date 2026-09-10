#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_shap_concepts.py

Fold-ensemble concept-level SHAP for XGBoost.

Key properties
--------------
- For MIMIC test/eICU external, SHAP is averaged across the five fold models.
- SHAP is computed on the model's raw margin, preserving additivity before the
  mortality sigmoid/Platt calibration.
- Signed SHAP values are summed across all model features that map to the
  same clinical concept, including resolution channels, within-window
  aggregations, and temporal descriptors defined by the current model matrix.
  The grouping is therefore schema-driven rather than hard-coded to a legacy
  feature count or descriptor inventory.
- Demographics are grouped as Age, Gender, and Race.
- The beeswarm color score is the mean percentile rank of constituent model
  features within the explained cohort. It is used only for Low/High visual
  context; SHAP magnitude/sign remains exact after grouping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
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


SHAP_PROTOCOL_VERSION = "P5_SHAP_CONCEPTS_FINAL_PROVENANCE_V1"
SHAP_IDENTITY_VERSION = "P5_SHAP_CONCEPT_IDENTITY_V1"
SHAP_STAGE_IDENTITY_VERSION = "P5_SHAP_STAGE_IDENTITY_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_payload(payload) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def software_versions(xgb=None):
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
    }
    if xgb is not None:
        versions["xgboost"] = xgb.__version__
    return versions


def verify_stage03_binding(
    *,
    task_output: Path,
    task_manifest: dict,
    matrix_path: Path,
    feature_path: Path,
    cohort: str,
):
    """
    Bind the SHAP input exactly to the canonical Stage-03 training artifacts.

    SHAP explains the matrix itself, not a prediction parquet, so the required
    provenance chain is:
      Stage-03 training identity -> exact fold models -> exact cohort matrix
      and feature-name file -> SHAP outputs.
    """
    training_identity_path = task_output / "training_identity.json"
    if not training_identity_path.is_file():
        raise FileNotFoundError(training_identity_path)

    training_identity = load_json(training_identity_path)

    manifest_training_id = task_manifest.get("training_identity_sha256")
    identity_training_id = training_identity.get("training_identity_sha256")

    if not manifest_training_id or not identity_training_id:
        raise RuntimeError(
            f"Missing Stage-03 training identity under {task_output}."
        )
    if manifest_training_id != identity_training_id:
        raise RuntimeError(
            f"Stage-03 training identity mismatch under {task_output}."
        )

    input_hashes = task_manifest.get("input_hashes", {})
    if cohort == "mimic_test":
        matrix_key = "test_matrix_file_sha256"
        feature_key = "test_feature_file_sha256"
    elif cohort == "eicu_external":
        matrix_key = "external_matrix_file_sha256"
        feature_key = "external_feature_file_sha256"
    else:
        raise ValueError(f"Unsupported SHAP cohort: {cohort}")

    expected_matrix_sha = input_hashes.get(matrix_key)
    expected_feature_sha = input_hashes.get(feature_key)
    if not expected_matrix_sha or not expected_feature_sha:
        raise RuntimeError(
            f"Stage-03 input hashes missing for {cohort} under {task_output}."
        )

    actual_matrix_sha = sha256_file(matrix_path)
    actual_feature_sha = sha256_file(feature_path)

    if actual_matrix_sha != expected_matrix_sha:
        raise RuntimeError(
            f"Matrix SHA256 mismatch for {cohort} under {task_output}: "
            f"{actual_matrix_sha} != {expected_matrix_sha}"
        )
    if actual_feature_sha != expected_feature_sha:
        raise RuntimeError(
            f"Feature-file SHA256 mismatch for {cohort} under {task_output}: "
            f"{actual_feature_sha} != {expected_feature_sha}"
        )

    model_records = []
    manifest_models = task_manifest.get("model_files", {})
    if set(manifest_models.keys()) != {"1", "2", "3", "4", "5"}:
        raise RuntimeError(
            f"Stage-03 model provenance is incomplete under {task_output}."
        )

    for fold in range(1, 6):
        model_path = task_output / "models" / f"fold_{fold}.json"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        expected_model_sha = manifest_models[str(fold)].get("sha256")
        if not expected_model_sha:
            raise RuntimeError(
                f"Missing Stage-03 SHA256 for fold {fold} under {task_output}."
            )

        actual_model_sha = sha256_file(model_path)
        if actual_model_sha != expected_model_sha:
            raise RuntimeError(
                f"Model SHA256 mismatch for fold {fold} under {task_output}: "
                f"{actual_model_sha} != {expected_model_sha}"
            )

        model_records.append(
            {
                "fold": fold,
                "file": str(model_path),
                "sha256": actual_model_sha,
                "best_rounds": manifest_models[str(fold)].get("best_rounds"),
            }
        )

    return {
        "training_identity_sha256": identity_training_id,
        "training_identity_file": str(training_identity_path),
        "training_identity_file_sha256": sha256_file(training_identity_path),
        "task_manifest_file": str(task_output / "task_manifest.json"),
        "task_manifest_file_sha256": sha256_file(
            task_output / "task_manifest.json"
        ),
        "matrix_file": str(matrix_path),
        "matrix_file_sha256": actual_matrix_sha,
        "feature_file": str(feature_path),
        "feature_file_sha256": actual_feature_sha,
        "model_files": model_records,
    }


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

    assigned = [
        int(i)
        for idx in groups.values()
        for i in idx
    ]
    if len(assigned) != len(feature_names):
        raise RuntimeError(
            "Concept grouping does not assign exactly one group per model feature."
        )
    if sorted(assigned) != list(range(len(feature_names))):
        raise RuntimeError(
            "Concept grouping has missing or duplicate feature assignments."
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
    if output_path.suffix.lower() == ".png":
        fig.savefig(
            output_path.with_suffix(".pdf"),
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
    if output_path.suffix.lower() == ".png":
        fig.savefig(
            output_path.with_suffix(".pdf"),
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

    if len(matrix.feature_names) != matrix.X.shape[1]:
        raise RuntimeError(
            f"Feature-name width mismatch for {landmark}h/{variant}/{cohort}."
        )
    if len(set(matrix.feature_names)) != len(matrix.feature_names):
        raise RuntimeError(
            f"Duplicate model feature names for {landmark}h/{variant}/{cohort}."
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

    if len(X) == 0:
        raise RuntimeError(
            f"No eligible rows for SHAP: {landmark}h/{outcome}/{variant}/{cohort}."
        )
    if max_rows <= 0:
        raise ValueError("--max-rows must be positive.")

    n_eligible = int(len(X))
    sampled = False

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
        sampled = True

    task_output = task_dir(
        modeling_root,
        landmark,
        outcome,
        variant,
    )
    task_manifest_path = task_output / "task_manifest.json"
    if not task_manifest_path.is_file():
        raise FileNotFoundError(task_manifest_path)
    task_manifest = load_json(task_manifest_path)

    for key, expected in (
        ("landmark_hour", landmark),
        ("outcome", outcome),
        ("variant", variant),
    ):
        if task_manifest.get(key) != expected:
            raise RuntimeError(
                f"Task-manifest mismatch for {key}: "
                f"{task_manifest.get(key)!r} != {expected!r}"
            )

    manifest_feature_count = int(task_manifest.get("feature_count", -1))
    if manifest_feature_count != X.shape[1]:
        raise RuntimeError(
            f"Feature-count mismatch: task manifest={manifest_feature_count}, "
            f"matrix={X.shape[1]}."
        )

    stage03_binding = verify_stage03_binding(
        task_output=task_output,
        task_manifest=task_manifest,
        matrix_path=matrix_path,
        feature_path=feature_path,
        cohort=cohort,
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
    max_additivity_error = 0.0

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

        margin = model.predict(
            dmatrix,
            output_margin=True,
        )
        reconstructed = np.sum(
            contrib,
            axis=1,
        )
        error = float(
            np.max(
                np.abs(
                    reconstructed - margin
                )
            )
        )
        max_additivity_error = max(
            max_additivity_error,
            error,
        )
        if not np.isfinite(error) or error > 1e-3:
            raise RuntimeError(
                f"SHAP additivity check failed for {model_path}: "
                f"max_abs_error={error:.6g}"
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
    importance_path = figure_dir / "shap_concept_importance.csv"
    grouped_npz_path = figure_dir / "grouped_shap_values.npz"
    beeswarm_png = figure_dir / "shap_concept_beeswarm.png"
    beeswarm_pdf = figure_dir / "shap_concept_beeswarm.pdf"
    bar_png = figure_dir / "shap_concept_bar.png"
    bar_pdf = figure_dir / "shap_concept_bar.pdf"

    importance_df.to_csv(
        importance_path,
        index=False,
    )

    np.savez_compressed(
        grouped_npz_path,
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

    output_paths = [
        importance_path,
        grouped_npz_path,
        beeswarm_png,
        beeswarm_pdf,
        bar_png,
        bar_pdf,
    ]
    for output_path in output_paths:
        if not output_path.is_file():
            raise FileNotFoundError(
                f"Expected SHAP output missing: {output_path}"
            )

    output_files = [
        {
            "file": str(output_path),
            "sha256": sha256_file(output_path),
            "bytes": int(output_path.stat().st_size),
        }
        for output_path in output_paths
    ]

    shap_protocol = {
        "protocol_version": SHAP_PROTOCOL_VERSION,
        "landmark_hour": int(landmark),
        "outcome": outcome,
        "variant": variant,
        "cohort": cohort,
        "max_rows": int(max_rows),
        "top_n": int(top_n),
        "sampling_seed": int(seed),
        "sampling_rule": (
            "all eligible rows if n<=max_rows; otherwise simple random "
            "sample without replacement using numpy default_rng(seed)"
        ),
        "shap_algorithm": "XGBoost exact pred_contribs",
        "approx_contribs": False,
        "shap_scale": "raw_margin",
        "fold_models": 5,
        "fold_aggregation": "arithmetic mean of signed SHAP across fold models",
        "concept_aggregation": (
            "sum of signed SHAP over schema-derived constituent features"
        ),
        "demographic_groups": ["Age", "Gender", "Race"],
        "beeswarm_color": (
            "mean percentile rank of constituent model features within "
            "the explained cohort; visual context only"
        ),
        "max_allowed_additivity_error": 1e-3,
    }
    shap_protocol_sha256 = sha256_json_payload(shap_protocol)

    source_path = Path(__file__).resolve()
    identity_payload = {
        "identity_version": SHAP_IDENTITY_VERSION,
        "stage03_binding": stage03_binding,
        "shap_protocol_sha256": shap_protocol_sha256,
        "shap_source_sha256": sha256_file(source_path),
        "modeling_common_source_sha256": task_manifest.get(
            "modeling_common_source_sha256"
        ),
        "software_versions": software_versions(xgb),
        "eligible_rows": int(n_eligible),
        "explained_rows": int(len(X)),
        "feature_count": int(X.shape[1]),
        "concept_group_count": int(len(concepts)),
        "output_files": output_files,
    }
    shap_identity_sha256 = sha256_json_payload(identity_payload)

    shap_manifest = {
        "status": "PASS",
        "landmark_hour": int(landmark),
        "outcome": outcome,
        "variant": variant,
        "cohort": cohort,
        "eligible_rows": n_eligible,
        "explained_rows": int(len(X)),
        "sampled": bool(sampled),
        "sampling_seed": int(seed),
        "max_rows": int(max_rows),
        "top_n": int(top_n),
        "feature_count": int(X.shape[1]),
        "concept_group_count": int(len(concepts)),
        "fold_models": int(len(model_paths)),
        "shap_scale": "raw_margin",
        "fold_aggregation": "mean_signed_shap",
        "concept_aggregation": "sum_signed_shap_over_constituent_features",
        "max_shap_additivity_error": float(max_additivity_error),
        "stage03_binding": stage03_binding,
        "training_identity_sha256": stage03_binding[
            "training_identity_sha256"
        ],
        "shap_protocol": shap_protocol,
        "shap_protocol_sha256": shap_protocol_sha256,
        "shap_identity_sha256": shap_identity_sha256,
        "shap_identity_short": shap_identity_sha256[:16],
        "shap_source_file": str(source_path),
        "shap_source_sha256": sha256_file(source_path),
        "software_versions": software_versions(xgb),
        "output_files": output_files,
    }

    shap_manifest_path = figure_dir / "shap_manifest.json"
    shap_manifest_path.write_text(
        json.dumps(shap_manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"SHAP PASS | {title} | rows={len(X)}"
        f"/{n_eligible} | sampled={sampled} | "
        f"identity={shap_identity_sha256[:16]}"
    )

    return {
        "landmark_hour": int(landmark),
        "outcome": outcome,
        "variant": variant,
        "cohort": cohort,
        "training_identity_sha256": stage03_binding[
            "training_identity_sha256"
        ],
        "shap_identity_sha256": shap_identity_sha256,
        "shap_manifest": str(shap_manifest_path),
        "shap_manifest_sha256": sha256_file(shap_manifest_path),
        "status": "PASS",
    }


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

    if args.max_rows <= 0:
        p.error("--max-rows must be positive.")
    if args.top_n <= 0:
        p.error("--top-n must be positive.")

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

    invalid_outcomes = sorted(set(outcomes) - {"los", "mortality"})
    if invalid_outcomes:
        p.error(f"Unsupported outcomes: {invalid_outcomes}")
    invalid_cohorts = sorted(
        set(cohorts) - {"mimic_test", "eicu_external"}
    )
    if invalid_cohorts:
        p.error(f"Unsupported SHAP cohorts: {invalid_cohorts}")

    run_records = []

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
                    record = explain_task(
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
                    run_records.append(record)

    if not run_records:
        raise RuntimeError("No SHAP tasks were executed.")

    shap_root = modeling_root / "figures" / "shap"
    stage_source = Path(__file__).resolve()
    stage_payload = {
        "identity_version": SHAP_STAGE_IDENTITY_VERSION,
        "requested_landmarks": landmarks,
        "requested_outcomes": outcomes,
        "requested_variants": variants,
        "requested_cohorts": cohorts,
        "max_rows": int(args.max_rows),
        "top_n": int(args.top_n),
        "seed": int(args.seed),
        "shap_source_sha256": sha256_file(stage_source),
        "software_versions": software_versions(get_xgboost()),
        "runs": run_records,
    }
    stage_identity_sha256 = sha256_json_payload(stage_payload)

    stage_manifest = {
        "status": "PASS",
        "executed_runs": int(len(run_records)),
        "shap_stage_identity_sha256": stage_identity_sha256,
        "shap_stage_identity_short": stage_identity_sha256[:16],
        "stage_payload": stage_payload,
    }
    stage_manifest_path = shap_root / "06_shap_stage_manifest.json"
    stage_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    stage_manifest_path.write_text(
        json.dumps(stage_manifest, indent=2),
        encoding="utf-8",
    )

    print(
        f"SHAP stage complete. PASS runs={len(run_records)} | "
        f"stage_identity={stage_identity_sha256}"
    )
    print(stage_manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
