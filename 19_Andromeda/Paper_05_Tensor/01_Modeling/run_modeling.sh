#!/usr/bin/env bash
set -euo pipefail

cd /home/ddimopoulos/Paper_05_Tensor/01_Modeling

echo "[1/7] Preparing deterministic model matrices..."
python 00_prepare_model_matrices.py

echo "[2/7] Optuna tuning at the fully aligned 24h anchor..."
python 01_optuna_tune_xgboost.py \
  --tuning-landmark 24 \
  --trials 40 \
  --workers 4 \
  --xgb-threads 8

echo "[3/7] Training all landmark models with frozen Optuna parameters..."
python 02_train_xgboost.py \
  --params-source optuna \
  --tuning-landmark 24 \
  --workers 6 \
  --xgb-threads 8

echo "[4/7] Collecting metrics..."
python 03_collect_metrics.py

echo "[5/7] Running patient-level bootstrap and paired landmark analysis..."
python 06_bootstrap_landmark_analysis.py \
  --bootstrap 2000 \
  --workers 10 \
  --common-risk-landmarks 16,20,24 \
  --common-risk-pairs all

echo "[6/7] Generating evaluation plots..."
python 04_plot_results.py --task-plots

echo "[7/7] Generating concept-level SHAP for full multi-resolution models..."
python 05_shap_concepts.py \
  --landmarks 4,8,12,16,20,24,36,48 \
  --variants full \
  --outcomes los,mortality \
  --cohorts mimic_test,eicu_external \
  --max-rows 2000 \
  --top-n 20

echo "Modeling pipeline complete."