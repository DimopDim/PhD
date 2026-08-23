# Multi-Resolution Tensorization of ICU EHRs

Reproducible data-preparation and tensor-construction pipeline for the study:

> **Multi-Resolution Tensorization of Irregularly Sampled ICU Electronic Health Records: External Validation of Length-of-Stay and Mortality Prediction in Stroke Patients**

This repository implements a deterministic, leakage-aware pipeline for constructing multi-resolution temporal representations from **MIMIC-IV v3.1** and **eICU-CRD v2.0** stroke ICU cohorts.

The pipeline performs cohort construction, raw-event extraction, cross-database harmonization, cumulative temporal aggregation, leakage auditing, patient-level split creation, dynamic landmark construction, demographic encoding, and causal alignment into four-channel NumPy tensors.

No raw MIMIC-IV or eICU data are distributed with this repository. Users must obtain authorized access to the source databases separately.

---

## 1. Pipeline overview

The scripts must be executed in the following order:

| Stage | Script | Purpose |
|---|---|---|
| 01 | `01_build_stroke_cohorts.py` | Identify eligible stroke ICU cohorts in MIMIC-IV and eICU |
| 02 | `02_extract_raw_events.py` | Extract numeric clinical/laboratory observations from the first 48 ICU hours |
| 02b | `02b_audit_raw_features.py` | Audit raw feature coverage, units, source variables, and mapping candidates |
| 03 | `03_harmonize_events.py` | Harmonize MIMIC-IV/eICU variables, units, demographics, and canonical concepts |
| 04 | `04_build_cumulative_windows.py` | Construct cumulative 1 h, 2 h, 3 h, and 4 h temporal summaries |
| 04b | `04b_audit_temporal_leakage.py` | Quantify potential ICU-LOS leakage through temporal missingness patterns |
| 05 | `05_create_splits_and_encode.py` | Create fixed MIMIC patient split, dynamic landmark risk sets, targets, and frozen demographic encoding |
| 06 | `06_build_multilandmark_tensor_cubes.py` | Causally align the four resolutions and construct patient-level tensor cubes |

The output of each stage becomes the input of the next stage. Each major stage also produces a JSON manifest and/or CSV audit files so that the run can be inspected after execution.

---

## 2. Data sources

The pipeline expects the following database versions:

- **MIMIC-IV v3.1** — development and internal evaluation cohort.
- **eICU Collaborative Research Database v2.0** — external evaluation cohort.

These databases are available through PhysioNet subject to their respective access requirements and data-use agreements.

### Expected raw-data layout

The commands below assume the following server paths:

```text
/home/ddimopoulos/Datasets/00_Datasets/
├── mimic-iv-3_1/
│   ├── derived/
│   ├── hosp/
│   ├── icu/
│   └── note/
└── eicu-2_0/
```

You may use different paths by changing the shell variables shown below or by passing the corresponding command-line arguments.

---

## 3. Repository layout

A recommended repository structure is:

```text
00_Create_Tensor/
├── README.md
├── 01_build_stroke_cohorts.py
├── 02_extract_raw_events.py
├── 02b_audit_raw_features.py
├── 03_harmonize_events.py
├── 04_build_cumulative_windows.py
├── 04b_audit_temporal_leakage.py
├── 05_create_splits_and_encode.py
├── 06_build_multilandmark_tensor_cubes.py
│
├── mimic_icd_stroke.csv
├── eicu_icd_stroke.csv
│
├── imports/
│   ├── concept_schema_76.csv
│   ├── feature_mapping_76.csv
│   └── pipeline_config.json
│
└── data/
    ├── 01_cohorts/
    ├── 02_raw_events/
    ├── 03_harmonized/
    ├── 04_cumulative_windows/
    ├── 05_splits/
    └── 06_numpy_cubes/
```

The `data/` directory is generated automatically and should normally be excluded from Git because it may contain derived patient-level data.

---

## 4. Required project files

Before running the pipeline, place the following controlled inputs in the repository.

### Stroke ICD code files

At the project root:

```text
mimic_icd_stroke.csv
eicu_icd_stroke.csv
```

`01_build_stroke_cohorts.py` accepts either a database-specific `icd_code` column or a single-column CSV file.

### Harmonization files

Inside `imports/`:

```text
concept_schema_76.csv
feature_mapping_76.csv
pipeline_config.json
```

The current implementation uses a locked schema of **76 time-varying clinical concepts**. With the primary four summary operators (`mean`, `median`, `min`, `max`), Stage 04 produces:

```text
76 concepts × 4 aggregations = 304 clinical descriptors per temporal endpoint
```

Demographic predictors are appended later in Stage 05 using a categorical schema fitted only on the MIMIC training partition.

---

## 5. Software requirements

The Python scripts require:

- Python 3
- NumPy
- pandas
- PyArrow
- scikit-learn

Install the minimum Python dependencies in a clean virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas pyarrow scikit-learn
```

For stricter reproducibility, generate and commit a frozen dependency file from the environment used for the final experiment:

```bash
python -m pip freeze > requirements-lock.txt
```

Then another user can recreate that software environment with:

```bash
python -m pip install -r requirements-lock.txt
```

### Optional: `pigz`

Stage 02 can use `pigz` for multi-threaded decompression of large `.csv.gz` files. If `pigz` is unavailable, the code automatically falls back to pandas/gzip decompression.

On Debian/Ubuntu systems:

```bash
sudo apt-get install pigz
```

---

## 6. Define paths once

The scripts from Stage 02 onward use the following directory as the canonical project root:

```bash
export PROJECT_ROOT=/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor
export MIMIC_ROOT=/home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1
export EICU_ROOT=/home/ddimopoulos/Datasets/00_Datasets/eicu-2_0

cd "$PROJECT_ROOT"
```

> **Important:** `01_build_stroke_cohorts.py` contains an older built-in project-root default. For reproducibility, the commands in this README always pass `--project-root "$PROJECT_ROOT"` explicitly so that every stage writes under the same pipeline directory.

---

# 7. Execution

## Stage 01 — Build stroke cohorts

This stage identifies stroke-related ICU admissions using the supplied ICD code lists.

Key rules include:

- MIMIC-IV diagnosis matching against `hosp/diagnoses_icd`.
- eICU diagnosis matching against `diagnosis.icd9code`.
- ICD codes are normalized before matching.
- Only one eligible ICU stay per patient is retained.
- ICU stays longer than 10 days are excluded by default.
- All eligible stays are also retained for auditing.

Run:

```bash
python 01_build_stroke_cohorts.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database all \
  --chunksize 500000 \
  --max-icu-los-days 10
```

Main output directory:

```text
data/01_cohorts/
```

Run manifest:

```text
data/01_cohorts/01_cohort_manifest.json
```

---

## Stage 02 — Extract first-48-hour raw events

Stage 02 extracts cohort-relevant numeric events and anchors them to ICU admission.

It does **not** perform harmonization, temporal aggregation, imputation, splitting, or tensor construction.

Because the source files can be large, running the two databases separately is recommended.

### MIMIC-IV

```bash
python 02_extract_raw_events.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database mimic \
  --horizon-hours 48 \
  --workers 10 \
  --chunksize 500000 \
  --max-in-flight 20 \
  --pigz-workers 4
```

### eICU

```bash
python 02_extract_raw_events.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database eicu \
  --horizon-hours 48 \
  --workers 10 \
  --chunksize 500000 \
  --max-in-flight 20 \
  --pigz-workers 4
```

Main output directory:

```text
data/02_raw_events/
```

Run manifest:

```text
data/02_raw_events/02_raw_event_manifest.json
```

---

## Stage 02b — Audit raw features

This stage audits feature availability **before harmonization**. It preserves source-table names, raw variable names, source item IDs, observed units, event counts, patient coverage, and numerical ranges.

Run the strict audit:

```bash
python 02b_audit_raw_features.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --batch-rows 250000 \
  --workers 4 \
  --strict
```

Important reports include:

```text
data/02_raw_events/reports/
├── mimic_feature_inventory.csv
├── eicu_feature_inventory.csv
├── unit_inventory.csv
├── source_summary.csv
├── patient_event_coverage.csv
├── cohort_coverage_summary.csv
├── demographics_summary.csv
├── mapping_review_candidates.csv
├── feature_mapping_template.csv
└── 02b_audit_manifest.json
```

Review these outputs before proceeding to Stage 03, especially if the raw database release or mapping files have changed.

---

## Stage 03 — Harmonize events and demographics

Stage 03 maps raw database-specific measurements onto the controlled 76-concept schema.

Key rules include:

- no temporal aggregation;
- no imputation;
- missing values remain missing;
- deterministic source mapping;
- exact duplicate observations are removed;
- sentinel `999999` is treated as invalid;
- FiO2 is standardized to percentage;
- selected eICU unit conversions are applied explicitly;
- MIMIC GCS is reconstructed from Eye + Motor + Verbal only when all three components are available at the same timestamp;
- demographic variables are harmonized but not yet one-hot encoded.

Run:

```bash
python 03_harmonize_events.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --batch-rows 250000
```

For a finalized mapping file with no unresolved review rows, the stricter form can be used:

```bash
python 03_harmonize_events.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --batch-rows 250000 \
  --fail-on-review
```

Main outputs:

```text
data/03_harmonized/
├── mimic/
│   ├── harmonized_events.parquet
│   └── harmonized_demographics.parquet
├── eicu/
│   ├── harmonized_events.parquet
│   └── harmonized_demographics.parquet
├── reports/
│   ├── concept_coverage.csv
│   ├── conversion_audit.csv
│   └── review_flags.csv
└── 03_harmonization_manifest.json
```

---

## Stage 04 — Build cumulative multi-resolution windows

The primary temporal representation uses cumulative summaries at four native resolutions:

| Channel | Resolution | Native endpoints through 48 h |
|---|---:|---|
| `o1` | 1 h | 1, 2, ..., 48 |
| `o2` | 2 h | 2, 4, ..., 48 |
| `o3` | 3 h | 3, 6, ..., 48 |
| `o4` | 4 h | 4, 8, ..., 48 |

The primary aggregation operators are:

```text
mean, median, min, max
```

An event exactly at a window endpoint is included. No clinical observation is carried beyond ICU discharge. If a temporal endpoint occurs after ICU discharge, all clinical descriptors for that endpoint are set to `NaN`.

### Run the built-in unit test

```bash
python 04_build_cumulative_windows.py \
  --project-root "$PROJECT_ROOT" \
  --self-test
```

Expected result:

```text
Stage 04 self-test: PASS
```

### Build the windows

```bash
python 04_build_cumulative_windows.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --workers 10 \
  --patients-per-task 64
```

Main output directory:

```text
data/04_cumulative_windows/
```

Key audit files:

```text
data/04_cumulative_windows/
├── feature_order.json
├── 04_cumulative_windows_manifest.json
└── reports/
    ├── window_summary.csv
    ├── final_48h_consistency.csv
    └── feature_order.csv
```

`final_48h_consistency.csv` verifies that the cumulative 48-hour clinical summary is consistent across the four native resolutions.

---

## Stage 04b — Audit temporal leakage

Stage 04 deliberately makes post-discharge clinical windows unavailable. For patients discharged before 48 hours, the resulting all-`NaN` tail can encode discharge timing and therefore potentially leak total ICU LOS.

Stage 04b quantifies this issue without modifying Stage 04 outputs.

It evaluates:

1. a structural availability oracle used for diagnosis only;
2. an implicit boundary inferred only from the clinical `NaN` pattern;
3. a missingness-only ExtraTrees out-of-fold model using 48 endpoint-wise observed fractions.

### Run the self-test

```bash
python 04b_audit_temporal_leakage.py \
  --project-root "$PROJECT_ROOT" \
  --self-test
```

### Run the full leakage audit

```bash
python 04b_audit_temporal_leakage.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --cv-folds 5 \
  --trees 300 \
  --model-workers 10 \
  --seed 42
```

Outputs:

```text
data/04_cumulative_windows/leakage_audit/
├── leakage_patient_summary.csv
├── leakage_metrics.csv
├── missingness_oof_predictions.csv
├── endpoint_missingness_summary.csv
├── cohort_leakage_summary.csv
└── 04b_temporal_leakage_manifest.json
```

This audit motivates the dynamic landmark formulation used in Stage 05.

---

## Stage 05 — Create split, landmark risk sets, targets, and demographic encoding

Stage 05 creates one fixed patient-level MIMIC development/test assignment and reuses that assignment at every landmark.

Default landmarks:

```text
1, 4, 8, 12, 16, 20, 24, 36, 48 hours
```

For a landmark `t`:

```text
eligibility: total ICU LOS > t
observation: only cumulative endpoints <= t
LOS target: total ICU LOS - t
mortality target: subsequent in-hospital mortality
```

Important leakage barriers:

- patients discharged at or before the landmark are excluded;
- future temporal endpoints are never included;
- LOS, remaining LOS, mortality, and window-availability variables do not enter predictor matrices;
- targets are stored separately from predictors;
- clinical values are not imputed or scaled;
- eICU is never used to create the MIMIC split or fit the demographic schema;
- race vocabulary is frozen from the MIMIC training partition and applied unchanged to test and eICU.

Run:

```bash
python 05_create_splits_and_encode.py \
  --project-root "$PROJECT_ROOT" \
  --landmarks 1,4,8,12,16,20,24,36,48 \
  --seed 42 \
  --train-fraction 0.80 \
  --test-fraction 0.20
```

Main outputs:

```text
data/05_splits/
├── demographic_schema.json
├── feature_order.json
├── 05_multilandmark_manifest.json
├── reports/
│   ├── full_cohort_split_assignments.csv
│   ├── demographic_category_counts.csv
│   ├── demographic_encoding_audit.csv
│   ├── feature_order.csv
│   ├── landmark_risk_set_summary.csv
│   ├── landmark_grid_summary.csv
│   ├── nested_risk_set_audit.csv
│   └── landmark_split_assignments.csv
└── landmark_XXXh/
    ├── mimic/train/
    ├── mimic/test/
    ├── eicu/external/
    └── targets/
```

The MIMIC **training/development partition** is the only source used for fitting data-dependent preprocessing schemas. Five-fold model cross-validation is intended to be performed later within that development partition.

---

## Stage 06 — Build causal multi-resolution tensor cubes

Stage 06 converts the Stage-05 native-resolution files into causally aligned four-channel tensors.

For landmark `t` and `F` frozen predictors:

```text
X.shape = (N, t, F, 4)
```

Channel order:

```text
[o1, o2, o3, o4] = [1 h, 2 h, 3 h, 4 h]
```

For native resolution `w`, aligned hour `h` uses only the latest **completed** endpoint:

```text
e(h, w) = floor(h / w) × w
```

A future native endpoint is never copied backward.

For example, at 8 hours:

```text
hour:  1   2   3   4   5   6   7   8
o1:    X1  X2  X3  X4  X5  X6  X7  X8
o2:    NA  X2  X2  X4  X4  X6  X6  X8
o3:    NA  NA  X3  X3  X3  X6  X6  X6
o4:    NA  NA  NA  X4  X4  X4  X4  X8
```

Static demographic variables are validated as time-invariant and repeated across the temporal axis and all four channels. No imputation or scaling is performed.

Run all landmarks present in the Stage-05 manifest:

```bash
python 06_build_multilandmark_tensor_cubes.py \
  --project-root "$PROJECT_ROOT" \
  --atol 1e-6 \
  --rtol 1e-6
```

To build only selected landmarks:

```bash
python 06_build_multilandmark_tensor_cubes.py \
  --project-root "$PROJECT_ROOT" \
  --landmarks 12,24,48 \
  --atol 1e-6 \
  --rtol 1e-6
```

Main output directory:

```text
data/06_numpy_cubes/
```

Each `.npz` file contains arrays including:

```text
X
y_los_remaining_days
y_los_total_days
y_mortality
mortality_known
patient_id
stay_id
time_hours
channel_available
source_endpoint_hour
```

Key audit files include:

```text
data/06_numpy_cubes/reports/
├── tensor_diagnostics.csv
├── causal_alignment_audit.csv
├── final_channel_consistency.csv
├── target_alignment_audit.csv
└── variant_availability.csv
```

The overall run is summarized in:

```text
data/06_numpy_cubes/06_tensor_manifest.json
```

---

# 8. Complete execution sequence

For convenience, the full pipeline can be executed manually in this order:

```bash
export PROJECT_ROOT=/home/ddimopoulos/Paper_05_Tensor/00_Create_Tensor
export MIMIC_ROOT=/home/ddimopoulos/Datasets/00_Datasets/mimic-iv-3_1
export EICU_ROOT=/home/ddimopoulos/Datasets/00_Datasets/eicu-2_0

cd "$PROJECT_ROOT"

# Stage 01
python 01_build_stroke_cohorts.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database all \
  --chunksize 500000 \
  --max-icu-los-days 10

# Stage 02 - MIMIC
python 02_extract_raw_events.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database mimic \
  --horizon-hours 48 \
  --workers 10 \
  --chunksize 500000 \
  --max-in-flight 20 \
  --pigz-workers 4

# Stage 02 - eICU
python 02_extract_raw_events.py \
  --project-root "$PROJECT_ROOT" \
  --mimic-root "$MIMIC_ROOT" \
  --eicu-root "$EICU_ROOT" \
  --database eicu \
  --horizon-hours 48 \
  --workers 10 \
  --chunksize 500000 \
  --max-in-flight 20 \
  --pigz-workers 4

# Stage 02b
python 02b_audit_raw_features.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --batch-rows 250000 \
  --workers 4 \
  --strict

# Stage 03
python 03_harmonize_events.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --batch-rows 250000

# Stage 04 unit test
python 04_build_cumulative_windows.py \
  --project-root "$PROJECT_ROOT" \
  --self-test

# Stage 04
python 04_build_cumulative_windows.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --workers 10 \
  --patients-per-task 64

# Stage 04b self-test
python 04b_audit_temporal_leakage.py \
  --project-root "$PROJECT_ROOT" \
  --self-test

# Stage 04b
python 04b_audit_temporal_leakage.py \
  --project-root "$PROJECT_ROOT" \
  --database all \
  --cv-folds 5 \
  --trees 300 \
  --model-workers 10 \
  --seed 42

# Stage 05
python 05_create_splits_and_encode.py \
  --project-root "$PROJECT_ROOT" \
  --landmarks 1,4,8,12,16,20,24,36,48 \
  --seed 42 \
  --train-fraction 0.80 \
  --test-fraction 0.20

# Stage 06
python 06_build_multilandmark_tensor_cubes.py \
  --project-root "$PROJECT_ROOT" \
  --atol 1e-6 \
  --rtol 1e-6
```

A shell wrapper may be added later, but running the stages explicitly is useful during replication because each audit can be inspected before the next transformation is performed.

---

# 9. Reproducibility checks

A successful run should not be judged only by whether the scripts terminate. The following artifacts should also be reviewed.

### Cohort construction

Check:

```text
data/01_cohorts/01_cohort_manifest.json
```

Confirm the database versions, ICD files, LOS exclusion rule, and final cohort sizes.

### Raw extraction

Check:

```text
data/02_raw_events/02_raw_event_manifest.json
```

Confirm the 48-hour extraction horizon and source-file row counts.

### Harmonization

Check:

```text
data/03_harmonized/reports/concept_coverage.csv
data/03_harmonized/reports/conversion_audit.csv
data/03_harmonized/reports/review_flags.csv
```

No unresolved variable/unit mapping should be silently accepted.

### Cumulative windows

Check:

```text
data/04_cumulative_windows/reports/final_48h_consistency.csv
```

The final cumulative summary should be equivalent across aligned resolutions within the configured tolerances.

### Temporal leakage

Check:

```text
data/04_cumulative_windows/leakage_audit/leakage_metrics.csv
```

The complete-cohort missingness audit should be considered when interpreting any model that includes patients discharged within the observation horizon.

### Landmark risk sets

Check:

```text
data/05_splits/reports/nested_risk_set_audit.csv
data/05_splits/reports/landmark_risk_set_summary.csv
```

Later landmark risk sets must be nested subsets of earlier landmark risk sets.

### Final tensor construction

Check:

```text
data/06_numpy_cubes/reports/causal_alignment_audit.csv
data/06_numpy_cubes/reports/final_channel_consistency.csv
data/06_numpy_cubes/reports/target_alignment_audit.csv
data/06_numpy_cubes/reports/tensor_diagnostics.csv
```

These audits verify that no future native endpoint is copied backward, target rows remain aligned with patient tensors, and channel consistency is preserved where expected.

---

# 10. Determinism and leakage controls

The implementation includes several explicit reproducibility and leakage protections:

- deterministic cohort-selection rules;
- patient-level rather than row-level splitting;
- fixed random seed (`42`) for the MIMIC split and leakage-audit model;
- eICU reserved for external evaluation;
- no eICU-based fitting of demographic categories;
- no clinical imputation or scaling during tensor construction;
- no future landmark endpoints in predictors;
- LOS and mortality targets stored separately from predictors;
- causal multi-resolution alignment;
- explicit tensor/target alignment audits;
- run manifests containing configuration and output metadata;
- built-in self-tests for cumulative-window construction and leakage auditing.

---

# 11. Computational notes

The default commands above were designed for a multi-core Linux server.

The most computationally intensive steps are generally:

- Stage 02: scanning and filtering large compressed MIMIC/eICU source tables;
- Stage 04: patient-level cumulative temporal aggregation;
- Stage 04b: the optional missingness-only ExtraTrees audit.

If fewer CPU cores are available, reduce `--workers`, `--model-workers`, and `--pigz-workers`. Lower worker counts affect runtime but should not change the deterministic data transformation itself.

For serial debugging, Stage 02 and Stage 04 support a single worker.

---

# 12. Git and data governance

Do **not** commit raw or patient-level derived MIMIC-IV/eICU data to a public repository.

A suitable `.gitignore` should include at least:

```gitignore
# Python
__pycache__/
*.py[cod]
.venv/

# Generated patient-level data
data/

# Logs
*.log

# Local environment files
.env

# IDE / OS
.vscode/
.idea/
.DS_Store
```

The controlled, non-patient-level configuration files required to reproduce the pipeline (`imports/`, ICD code lists, source code, and dependency lock file) should be version controlled.

---

# 13. Versioning recommendation

For a scientific repository, create an immutable Git tag or GitHub release corresponding to each manuscript/submission version, for example:

```bash
git tag -a tensor-pipeline-v1.0 -m "Reproducible tensor-construction pipeline"
git push origin tensor-pipeline-v1.0
```

Record the corresponding Git commit hash in the final experimental manifest or manuscript repository notes. This prevents later code changes from silently altering the implementation associated with reported results.

---

# 14. Important implementation note

The current codebase implements the corrected **76-concept / 304-clinical-descriptor** pipeline and a leakage-safe dynamic landmark design.

If numerical results from an earlier manuscript draft were generated with a different feature count, demographic width, split definition, or complete-cohort formulation, that earlier implementation should be preserved under a separate Git tag/release rather than mixed with this corrected pipeline. Reproducibility requires the code release, configuration files, and reported manuscript results to correspond to the same analytical version.

---

# 15. Citation

If you use this pipeline, please cite the associated manuscript:

```text
Dimopoulos D, Danilatou V, Kostoulas T.
Multi-Resolution Tensorization of Irregularly Sampled ICU Electronic Health Records:
External Validation of Length-of-Stay and Mortality Prediction in Stroke Patients.
```

The final DOI and bibliographic metadata should be added here once available.

---

## Contact

For questions regarding the implementation or reproduction of the analysis, please use the contact information provided in the associated manuscript or the GitHub repository's issue tracker.
