# Leakage-Safe ICU LOS Prediction

Code accompanying the study:

**Cross-database transportability of time-windowed ICU length-of-stay prediction in stroke: a comparative evaluation of missing-data handling strategies**

The pipeline evaluates ICU length-of-stay prediction models developed on **MIMIC-IV** and independently evaluated on **eICU**.

## Experiment

The pipeline evaluates **48 configurations**:

* 6 missing-data strategies: `native`, `mean`, `interpolation`, `regression`, `mlp`, `rnn`
* Scaling: `yes` / `no`
* Hyperparameter tuning: `none`, `randomized`, `bayesian`, `hyperopt`

All preprocessing is fitted only on the MIMIC-IV training set.

## Requirements

* Python ≥ 3.11
* MPI implementation (OpenMPI or equivalent)
* `mpi4py`

Install the Python package and dependencies:

```bash
pip install -e .
```

## Configuration

Edit:

```text
config.andromeda.toml
```

At minimum, set the paths to the MIMIC-IV and eICU input files and the output directory:

```toml
[data]
mimic_path = "path/to/mimic.csv"
eicu_path = "path/to/eicu.csv"

[experiment]
output_dir = "path/to/outputs"
```

## Preflight check

Validate the datasets, schema, patient-level split, and configuration before training:

```bash
python -m los_mpi.run_mpi \
  --config config.andromeda.toml \
  --preflight-only
```

## Run the experiment

Example MPI execution:

```bash
mpirun -np 12 python -m los_mpi.run_mpi \
  --config config.andromeda.toml
```

The pipeline automatically evaluates all **48 configurations**.

Results are written to the configured output directory, including:

```text
all_metrics_internal_external.csv
all_metrics_long.csv
split_manifest.csv
data_audit.json
environment.json
```

## Analyze a configuration

Generate residual and SHAP analyses for a selected configuration:

```bash
python -m los_mpi.analyze \
  --config config.andromeda.toml \
  --configuration mean__scale-no__hyperopt \
  --shap-dataset mimic_test \
  --shap-max-rows 2000
```

For external SHAP analysis:

```bash
python -m los_mpi.analyze \
  --config config.andromeda.toml \
  --configuration mean__scale-no__hyperopt \
  --shap-dataset eicu_external \
  --shap-max-rows 2000
```

Configuration IDs follow:

```text
<imputation>__scale-<yes|no>__<tuning>
```

Example:

```text
interpolation__scale-no__bayesian
```

## Notes

MIMIC-IV is used for development, validation, and internal testing.
eICU is used only for independent external evaluation and is not used for preprocessing fitting, model training, or hyperparameter selection.
