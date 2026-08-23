#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_prepare_model_matrices.py

Create and cache model-ready XGBoost matrices from Stage-06 tensors.

This stage is deterministic and outcome-independent. It is performed once so
LOS and mortality training reuse exactly the same representation.

Default variants:
    full,o1,o2,o3,o4,static

At 1h, unavailable variants are skipped automatically.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from modeling_common import (
    VALID_VARIANTS,
    build_model_matrix,
    load_json,
    load_tensor,
    matrix_cache_paths,
    parse_csv_arg,
    parse_landmarks,
    save_model_matrix,
    variant_available,
)


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)
TENSOR_ROOT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "00_Create_Tensor"
    / "data"
    / "06_numpy_cubes"
)
MODELING_ROOT_DEFAULT = (
    PROJECT_ROOT_DEFAULT
    / "01_Modeling"
)


def configure_logging(
    modeling_root: Path,
) -> logging.Logger:
    log_dir = modeling_root / "logs"
    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = logging.getLogger(
        "prepare_model_matrices"
    )
    logger.setLevel(
        logging.INFO
    )
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    sh = logging.StreamHandler(
        sys.stdout
    )
    sh.setFormatter(
        formatter
    )
    logger.addHandler(
        sh
    )

    fh = logging.FileHandler(
        log_dir
        / "00_prepare_model_matrices.log",
        mode="w",
        encoding="utf-8",
    )
    fh.setFormatter(
        formatter
    )
    logger.addHandler(
        fh
    )

    return logger


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT_DEFAULT,
    )
    p.add_argument(
        "--tensor-root",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--modeling-root",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--landmarks",
        type=str,
        default=None,
    )
    p.add_argument(
        "--variants",
        type=str,
        default="full,o1,o2,o3,o4,static",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()

    project_root = (
        args.project_root
        .expanduser()
        .resolve()
    )
    tensor_root = (
        args.tensor_root
        .expanduser()
        .resolve()
        if args.tensor_root is not None
        else project_root
        / "00_Create_Tensor"
        / "data"
        / "06_numpy_cubes"
    )
    modeling_root = (
        args.modeling_root
        .expanduser()
        .resolve()
        if args.modeling_root is not None
        else project_root
        / "01_Modeling"
    )

    logger = configure_logging(
        modeling_root
    )

    manifest = load_json(
        tensor_root
        / "06_tensor_manifest.json"
    )
    landmarks = parse_landmarks(
        args.landmarks,
        manifest,
    )
    variants = parse_csv_arg(
        args.variants,
        VALID_VARIANTS,
    )

    logger.info(
        "Tensor root: %s",
        tensor_root,
    )
    logger.info(
        "Modeling root: %s",
        modeling_root,
    )
    logger.info(
        "Landmarks: %s",
        landmarks,
    )
    logger.info(
        "Variants: %s",
        variants,
    )

    rows: List[dict] = []

    split_specs = [
        ("mimic", "train"),
        ("mimic", "test"),
        ("eicu", "external"),
    ]

    for landmark in landmarks:
        reference = load_tensor(
            tensor_root,
            landmark,
            "mimic",
            "train",
        )

        for variant in variants:
            available = variant_available(
                reference,
                variant,
            )
            if not available:
                logger.info(
                    "%dh/%s unavailable; skipped.",
                    landmark,
                    variant,
                )
                rows.append(
                    {
                        "landmark_hour": landmark,
                        "variant": variant,
                        "available": False,
                        "feature_count": np.nan,
                    }
                )
                continue

            expected_features = None

            for database, split_name in split_specs:
                matrix_path, feature_path = (
                    matrix_cache_paths(
                        modeling_root,
                        landmark,
                        variant,
                        database,
                        split_name,
                    )
                )

                if (
                    matrix_path.is_file()
                    and feature_path.is_file()
                    and not args.overwrite
                ):
                    logger.info(
                        "%dh/%s/%s_%s exists; skipped.",
                        landmark,
                        variant,
                        database,
                        split_name,
                    )
                    continue

                tensor = load_tensor(
                    tensor_root,
                    landmark,
                    database,
                    split_name,
                )

                if not variant_available(
                    tensor,
                    variant,
                ):
                    raise RuntimeError(
                        f"Variant availability differs across cohorts: "
                        f"{landmark}h/{variant}/{database}/{split_name}"
                    )

                matrix = build_model_matrix(
                    tensor,
                    variant,
                )

                if expected_features is None:
                    expected_features = (
                        matrix.feature_names
                    )
                elif (
                    matrix.feature_names
                    != expected_features
                ):
                    raise RuntimeError(
                        f"Feature schema mismatch across cohorts: "
                        f"{landmark}h/{variant}"
                    )

                save_model_matrix(
                    matrix,
                    matrix_path,
                    feature_path,
                )

                logger.info(
                    "%dh/%s/%s_%s | X=%s | NaN=%.2f%%",
                    landmark,
                    variant,
                    database,
                    split_name,
                    matrix.X.shape,
                    float(
                        np.isnan(
                            matrix.X
                        ).mean()
                        * 100.0
                    ),
                )

            if expected_features is None:
                # Existing cache path. Read the feature file.
                _, feature_path = (
                    matrix_cache_paths(
                        modeling_root,
                        landmark,
                        variant,
                        "mimic",
                        "train",
                    )
                )
                expected_features = load_json(
                    feature_path
                )["feature_names"]

            rows.append(
                {
                    "landmark_hour": landmark,
                    "variant": variant,
                    "available": True,
                    "feature_count": len(
                        expected_features
                    ),
                }
            )

    report_dir = (
        modeling_root
        / "reports"
    )
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = pd.DataFrame(
        rows
    )
    report.to_csv(
        report_dir
        / "model_matrix_manifest.csv",
        index=False,
    )

    logger.info(
        "Model-matrix preparation complete."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
