# Paper 05 Tensor — 01_Modeling with Optuna

This is the revised XGBoost modeling pipeline using leakage-safe Optuna
hyperparameter optimization.

## Pipeline

```text
00_prepare_model_matrices.py
        ↓
01_optuna_tune_xgboost.py
        ↓
02_train_xgboost.py
        ↓
03_collect_metrics.py
        ↓
04_plot_results.py
        ↓
05_shap_concepts.py
```

## Why Optuna is anchored at 24 h

Hyperparameters are tuned once per `outcome × representation` at **24 h**.

24 h is a fully aligned landmark for all four native resolutions:

```text
o1 -> 24 h
o2 -> 24 h
o3 -> 24 h
o4 -> 24 h
```

The resulting hyperparameters are then **frozen across all landmarks**.

This prevents the landmark comparison from becoming a comparison of nine
independently optimized models. The main scientific contrast remains how
predictive information changes as the observation horizon increases.

The tuner never loads MIMIC test or eICU.

## Patient-level cross-validation and identifier exclusion

All five-fold cross-validation is explicitly performed at **patient level**.

```text
LOS       -> GroupKFold(groups=patient_id)
Mortality -> StratifiedGroupKFold(groups=patient_id)
```

The inner early-stopping split is also checked for patient-level disjointness.

The current model matrices have exactly **one row per patient**, but the code
still uses `patient_id` as an explicit grouping key and aborts if duplicate
patient rows or train/validation patient overlap are detected.

Identifiers are metadata only. They are never model predictors. The pipeline
fails if the feature list contains identifiers/admin fields such as:

```text
subject_id
hadm_id
stay_id
patient_id
patientunitstayid
patienthealthsystemstayid
uniquepid
window_index
window_start_hour
window_end_hour
window_label
landmark_hour
ICU LOS / target columns
```

`patient_id` and `stay_id` are retained only for grouping, auditing, and
linking predictions back to patients.

## Optuna objectives

```text
LOS:
    minimize mean 5-fold RMSE

Mortality:
    maximize mean 5-fold Average Precision
```

For every Optuna fold, the outer validation fold is not used for early
stopping. Early stopping occurs on a further split inside the outer-training
subset.

## Search space

```text
eta                log 0.01 – 0.15
max_depth          3 – 10
min_child_weight   log 0.5 – 20
subsample          0.60 – 1.00
colsample_bytree   0.50 – 1.00
gamma              0 – 5
alpha              log 1e-5 – 10
lambda             log 1e-3 – 30
max_bin            128 / 256 / 512
tree_method        hist (fixed)
```

`scale_pos_weight` is deliberately not tuned because mortality prevalence
changes across dynamic risk sets. Mortality ranking is optimized with Average
Precision and probability calibration is handled later using MIMIC OOF
predictions only.

## Persistent studies

Each `outcome × representation` Optuna study is stored in SQLite:

```text
hyperparameters/
└── landmark_024h/
    ├── los/
    │   ├── full/
    │   │   ├── study.db
    │   │   ├── trials.csv
    │   │   └── best_params.json
    │   └── ...
    └── mortality/
        └── ...
```

Interrupted studies can be resumed. `--trials 40` means a target total of
40 trials per study; already existing trials are counted.

## Recommended sequence

### 1. Prepare matrices

```bash
cd /home/ddimopoulos/Paper_05_Tensor/01_Modeling

python 00_prepare_model_matrices.py
```

### 2. Optuna smoke test

Before spending compute on all studies:

```bash
python 01_optuna_tune_xgboost.py \
  --tuning-landmark 24 \
  --variants full,o1,static \
  --outcomes los,mortality \
  --trials 5 \
  --workers 1 \
  --xgb-threads 8
```

If this passes, continue the same persistent studies to 40 total trials:

```bash
python 01_optuna_tune_xgboost.py \
  --tuning-landmark 24 \
  --trials 40 \
  --workers 4 \
  --xgb-threads 8
```

### 3. Final training with frozen Optuna parameters

```bash
python 02_train_xgboost.py \
  --params-source optuna \
  --tuning-landmark 24 \
  --workers 6 \
  --xgb-threads 8
```

Every landmark for a given `outcome × representation` uses the same tuned
parameter file from 24 h.

The number of boosting rounds is still selected inside each training fold by
inner early stopping; test and external data never influence it.

### 4. Collect metrics

```bash
python 03_collect_metrics.py
```

### 5. Evaluation plots

```bash
python 04_plot_results.py --task-plots
```

### 6. SHAP

```bash
python 05_shap_concepts.py \
  --landmarks 4,8,12,16,20,24,36,48 \
  --variants full \
  --outcomes los,mortality \
  --cohorts mimic_test,eicu_external \
  --max-rows 2000 \
  --top-n 20
```

For the 1 h landmark:

```bash
python 05_shap_concepts.py \
  --landmarks 1 \
  --variants o1 \
  --outcomes los,mortality \
  --cohorts mimic_test,eicu_external
```

## Full one-command execution

After the smoke test:

```bash
chmod +x run_modeling.sh
./run_modeling.sh
```

The runner is configured for a 60-core node:

```text
Optuna:  4 studies × 8 XGBoost threads = 32 threads
Training: 6 tasks × 8 threads = 48 threads
```

## Important outputs to inspect after Optuna

```text
reports/optuna_tuning_summary.csv
hyperparameters/optuna_manifest.json
hyperparameters/landmark_024h/*/*/best_params.json
logs/01_optuna_tune_xgboost.log
```

Do not launch final training until all 12 tuning studies report `PASS`.
