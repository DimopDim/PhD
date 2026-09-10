#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_collect_metrics.py

Collect per-task XGBoost metrics into master CSV tables.

Provenance design
-----------------
This stage does not recompute model-performance metrics. It only aggregates the
per-task metrics.csv files produced by Stage 03. To preserve end-to-end
traceability, the collection manifest binds:
1. the Stage-03 training summary,
2. all Stage-03 training identities,
3. every input metrics.csv file,
4. the three aggregate CSV outputs,
5. this collector source file,
6. the runtime software versions,
into a deterministic collection identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT_DEFAULT = Path(
    "/home/ddimopoulos/Paper_05_Tensor"
)

COLLECTION_IDENTITY_VERSION = "P5_METRICS_COLLECTION_IDENTITY_V1"
COLLECTION_PROTOCOL_VERSION = "P5_METRICS_COLLECTION_FINAL_PROVENANCE_V1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json_payload(payload: Dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def software_versions() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
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

    report_dir = modeling_root / "reports"
    training_summary_path = report_dir / "training_task_summary.csv"

    if not training_summary_path.is_file():
        raise FileNotFoundError(
            f"{training_summary_path}. Run 03_train_xgboost.py first."
        )

    training = pd.read_csv(training_summary_path)

    required = {"landmark_hour", "outcome", "variant", "status"}
    missing = required - set(training.columns)
    if missing:
        raise RuntimeError(
            f"Training summary missing columns: {sorted(missing)}"
        )

    if training.empty or not training["status"].eq("PASS").all():
        raise RuntimeError(
            "Training summary is empty or contains non-PASS tasks."
        )

    keys = ["landmark_hour", "outcome", "variant"]
    if training.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate tasks in training_task_summary.csv."
        )

    metric_files = sorted(
        (
            modeling_root
            / "results"
        ).glob(
            "landmark_*h/*/*/metrics.csv"
        )
    )

    if not metric_files:
        raise FileNotFoundError(
            "No task metrics.csv files found."
        )

    if len(metric_files) != len(training):
        raise RuntimeError(
            f"Expected {len(training)} metrics.csv files from Stage 03, "
            f"found {len(metric_files)}."
        )

    frames: List[pd.DataFrame] = []
    metric_file_records = []
    training_identity_records = []

    for path in metric_files:
        frame = pd.read_csv(path)

        required_metric = {
            "landmark_hour",
            "outcome",
            "variant",
            "cohort",
            "n",
        }
        missing = required_metric - set(frame.columns)
        if missing:
            raise RuntimeError(
                f"{path} missing required columns: {sorted(missing)}"
            )

        if len(frame) != 3:
            raise RuntimeError(
                f"{path} must contain exactly 3 evaluated cohort rows; "
                f"found {len(frame)}."
            )

        expected_cohorts = {
            "mimic_train_oof",
            "mimic_test",
            "eicu_external",
        }
        observed = set(frame["cohort"].astype(str))
        if observed != expected_cohorts:
            raise RuntimeError(
                f"{path} cohort rows mismatch: {sorted(observed)}"
            )

        if frame.duplicated(
            ["landmark_hour", "outcome", "variant", "cohort"]
        ).any():
            raise RuntimeError(
                f"Duplicate metric rows in {path}."
            )

        task_dir = path.parent
        manifest_path = task_dir / "task_manifest.json"
        identity_path = task_dir / "training_identity.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing Stage-03 task manifest: {manifest_path}"
            )
        if not identity_path.is_file():
            raise FileNotFoundError(
                f"Missing Stage-03 training identity: {identity_path}"
            )

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        identity = json.loads(
            identity_path.read_text(encoding="utf-8")
        )

        row0 = frame.iloc[0]
        lm = int(row0["landmark_hour"])
        outcome = str(row0["outcome"])
        variant = str(row0["variant"])

        if int(manifest.get("landmark_hour", -1)) != lm:
            raise RuntimeError(
                f"Landmark mismatch between metrics and task manifest: {path}"
            )
        if str(manifest.get("outcome")) != outcome:
            raise RuntimeError(
                f"Outcome mismatch between metrics and task manifest: {path}"
            )
        if str(manifest.get("variant")) != variant:
            raise RuntimeError(
                f"Variant mismatch between metrics and task manifest: {path}"
            )

        manifest_training_id = manifest.get(
            "training_identity_sha256"
        )
        identity_training_id = identity.get(
            "training_identity_sha256"
        )

        if not manifest_training_id or not identity_training_id:
            raise RuntimeError(
                f"Missing Stage-03 training identity SHA256: {task_dir}"
            )

        if str(manifest_training_id) != str(identity_training_id):
            raise RuntimeError(
                f"Training identity mismatch between Stage-03 files: {task_dir}"
            )

        metric_file_records.append(
            {
                "landmark_hour": lm,
                "outcome": outcome,
                "variant": variant,
                "file": str(path),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
            }
        )

        training_identity_records.append(
            {
                "landmark_hour": lm,
                "outcome": outcome,
                "variant": variant,
                "training_identity_sha256": str(identity_training_id),
                "training_identity_file": str(identity_path),
                "training_identity_file_sha256": sha256_file(identity_path),
                "task_manifest_file": str(manifest_path),
                "task_manifest_file_sha256": sha256_file(manifest_path),
            }
        )

        frames.append(frame)

    all_metrics = pd.concat(
        frames,
        ignore_index=True,
    )

    all_metrics = all_metrics.sort_values(
        [
            "outcome",
            "landmark_hour",
            "variant",
            "cohort",
        ]
    ).reset_index(drop=True)

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics_path = report_dir / "all_metrics.csv"
    los_metrics_path = report_dir / "los_metrics.csv"
    mortality_metrics_path = report_dir / "mortality_metrics.csv"

    all_metrics.to_csv(
        all_metrics_path,
        index=False,
    )

    all_metrics.loc[
        all_metrics["outcome"].eq("los")
    ].to_csv(
        los_metrics_path,
        index=False,
    )

    all_metrics.loc[
        all_metrics["outcome"].eq("mortality")
    ].to_csv(
        mortality_metrics_path,
        index=False,
    )

    observed_tasks = (
        all_metrics[
            ["landmark_hour", "outcome", "variant"]
        ]
        .drop_duplicates()
        .sort_values(
            ["landmark_hour", "outcome", "variant"]
        )
        .reset_index(drop=True)
    )

    expected_tasks = (
        training[
            ["landmark_hour", "outcome", "variant"]
        ]
        .sort_values(
            ["landmark_hour", "outcome", "variant"]
        )
        .reset_index(drop=True)
    )

    if not observed_tasks.equals(expected_tasks):
        raise RuntimeError(
            "Collected metric task coverage does not match "
            "Stage-03 training summary."
        )

    training_ids = [
        r["training_identity_sha256"]
        for r in training_identity_records
    ]
    if len(set(training_ids)) != len(training_ids):
        raise RuntimeError(
            "Duplicate Stage-03 training identities detected."
        )

    collector_source = Path(__file__).resolve()

    collection_protocol = {
        "protocol_version": COLLECTION_PROTOCOL_VERSION,
        "operation": (
            "aggregate Stage-03 per-task metrics.csv files without "
            "recomputing performance metrics"
        ),
        "expected_cohorts_per_task": [
            "mimic_train_oof",
            "mimic_test",
            "eicu_external",
        ],
        "rows_per_task": 3,
        "sort_order": [
            "outcome",
            "landmark_hour",
            "variant",
            "cohort",
        ],
        "metric_recomputation": False,
    }
    collection_protocol_sha256 = sha256_json_payload(
        collection_protocol
    )

    identity_payload = {
        "identity_version": COLLECTION_IDENTITY_VERSION,
        "training_summary_file_sha256": sha256_file(
            training_summary_path
        ),
        "training_identities": sorted(
            training_identity_records,
            key=lambda r: (
                r["landmark_hour"],
                r["outcome"],
                r["variant"],
            ),
        ),
        "input_metric_files": sorted(
            metric_file_records,
            key=lambda r: (
                r["landmark_hour"],
                r["outcome"],
                r["variant"],
            ),
        ),
        "collection_protocol_sha256": collection_protocol_sha256,
        "collector_source_sha256": sha256_file(
            collector_source
        ),
        "software_versions": software_versions(),
    }

    metrics_collection_identity_sha256 = (
        sha256_json_payload(identity_payload)
    )

    output_files = {
        "all_metrics": {
            "file": str(all_metrics_path),
            "sha256": sha256_file(all_metrics_path),
            "rows": int(len(all_metrics)),
        },
        "los_metrics": {
            "file": str(los_metrics_path),
            "sha256": sha256_file(los_metrics_path),
            "rows": int(
                all_metrics["outcome"].eq("los").sum()
            ),
        },
        "mortality_metrics": {
            "file": str(mortality_metrics_path),
            "sha256": sha256_file(mortality_metrics_path),
            "rows": int(
                all_metrics["outcome"].eq("mortality").sum()
            ),
        },
    }

    audit = {
        "status": "PASS",
        "training_tasks": int(len(training)),
        "metric_files": int(len(metric_files)),
        "aggregate_metric_rows": int(len(all_metrics)),
        "expected_metric_rows": int(3 * len(training)),
        "all_training_tasks_pass": True,
        "task_coverage_matches_training_summary": True,
        "unique_training_identities": int(len(set(training_ids))),
        "training_summary_file": str(training_summary_path),
        "training_summary_file_sha256": sha256_file(
            training_summary_path
        ),
        "collection_protocol": collection_protocol,
        "collection_protocol_sha256": collection_protocol_sha256,
        "metrics_collection_identity_sha256": (
            metrics_collection_identity_sha256
        ),
        "metrics_collection_identity_short": (
            metrics_collection_identity_sha256[:16]
        ),
        "identity_payload": identity_payload,
        "output_files": output_files,
        "collector_source_file": str(collector_source),
        "collector_source_sha256": sha256_file(
            collector_source
        ),
        "software_versions": software_versions(),
    }

    manifest_path = (
        report_dir
        / "metrics_collection_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"PASS: collected {len(metric_files)} task metric files "
        f"into {len(all_metrics)} rows."
    )
    print(
        f"Unique Stage-03 training identities: "
        f"{len(set(training_ids))}"
    )
    print(
        "Metrics collection identity: "
        f"{metrics_collection_identity_sha256}"
    )
    print(all_metrics_path)
    print(manifest_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
