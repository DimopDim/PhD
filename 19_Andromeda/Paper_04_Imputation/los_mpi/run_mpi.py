from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mpi4py import MPI

from .config import load_config
from .data import audit_json, load_and_split, save_split_manifest
from .experiment import (
    all_bundle_tasks,
    collect_results,
    environment_provenance,
    safe_task,
)


WORK_TAG = 11
RESULT_TAG = 12
STOP_TAG = 13


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all 48 leakage-safe MIMIC-IV to eICU LOS configurations with MPI."
    )
    parser.add_argument("--config", required=True, help="Path to the TOML configuration file.")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs, split integrity, and schema without fitting models.",
    )
    return parser.parse_args()


def _prepare(config, mpi_size: int):
    output = Path(config.experiment.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_and_split(config)
    (output / "data_audit.json").write_text(audit_json(bundle), encoding="utf-8")
    save_split_manifest(bundle, output / "split_manifest.csv")
    (output / "environment.json").write_text(
        json.dumps(environment_provenance(config, mpi_size), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return bundle


def _serial(config, bundle) -> list[dict]:
    return [safe_task(task, bundle, config, rank=0) for task in all_bundle_tasks()]


def _master(config, size: int) -> list[dict]:
    tasks = all_bundle_tasks()
    next_task = 0
    active = 0
    results: list[dict] = []
    for worker in range(1, size):
        if next_task < len(tasks):
            MPI.COMM_WORLD.send(tasks[next_task], dest=worker, tag=WORK_TAG)
            next_task += 1
            active += 1
        else:
            MPI.COMM_WORLD.send(None, dest=worker, tag=STOP_TAG)
    while active:
        status = MPI.Status()
        result = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=RESULT_TAG, status=status)
        worker = status.Get_source()
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if next_task < len(tasks):
            MPI.COMM_WORLD.send(tasks[next_task], dest=worker, tag=WORK_TAG)
            next_task += 1
        else:
            MPI.COMM_WORLD.send(None, dest=worker, tag=STOP_TAG)
            active -= 1
    return results


def _worker(config, rank: int) -> None:
    bundle = load_and_split(config)
    while True:
        status = MPI.Status()
        task = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == STOP_TAG:
            return
        result = safe_task(task, bundle, config, rank)
        MPI.COMM_WORLD.send(result, dest=0, tag=RESULT_TAG)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    os.environ.setdefault("OMP_NUM_THREADS", str(config.experiment.threads_per_rank))
    os.environ.setdefault("MKL_NUM_THREADS", str(config.experiment.threads_per_rank))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(config.experiment.threads_per_rank))

    bundle = None
    preflight_error = None
    if rank == 0:
        try:
            bundle = _prepare(config, size)
            print(
                f"Preflight passed: {len(bundle.feature_columns)} features, "
                f"{len(bundle.train.y)} train rows, MPI size={size}.",
                flush=True,
            )
        except Exception as exc:
            preflight_error = f"{type(exc).__name__}: {exc}"
    preflight_error = comm.bcast(preflight_error, root=0)
    if preflight_error:
        if rank == 0:
            print(f"Preflight failed: {preflight_error}", file=sys.stderr, flush=True)
        comm.Abort(2)
    if args.preflight_only:
        return

    if size == 1:
        results = _serial(config, bundle)
    elif rank == 0:
        results = _master(config, size)
    else:
        _worker(config, rank)
        return

    if rank == 0:
        table = collect_results(config)
        output = Path(config.experiment.output_dir).expanduser().resolve()
        (output / "mpi_task_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True), encoding="utf-8"
        )
        failures = [result for result in results if result["status"] == "failed"]
        configuration_count = len(table.config_id.unique()) if not table.empty else 0
        if configuration_count != 48:
            print("Warning: results do not yet contain all 48 configurations.", file=sys.stderr)
        if failures:
            print(f"{len(failures)} MPI bundle task(s) failed. See mpi_task_results.json.", file=sys.stderr)
            raise SystemExit(1)
        print(f"Completed. Consolidated metrics: {output / 'all_metrics_internal_external.csv'}")


if __name__ == "__main__":
    main()
