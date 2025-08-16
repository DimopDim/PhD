# CIR-03: Hierarchical Imputation Framework (MPI-Parallelized)

This repository contains an **MPI-parallelized hierarchical imputation framework** for clinical datasets with missing values.  
It was developed for large ICU datasets (e.g., **MIMIC-IV**, **eICU**) where missingness is often **not random** and requires robust, multi-stage strategies.

---

## Key Features

- **MPI parallelization** using [`mpi4py`](https://mpi4py.readthedocs.io/)  
  Distributes imputation jobs across multiple CPU cores/nodes (tested up to 64 cores).
- **Dynamic hierarchical grouping**  
  Rows are grouped by percentage of missing values (e.g., 0–5%, 5–10%, …, 90–100%).
- **Multiple imputation methods supported**:
  - **Simple**: Mean, Median, KNN
  - **Model-based**: Iterative (ExtraTrees, Ridge, BayesianRidge, HistGradientBoosting), XGBoost
  - **Deep Learning**: LSTM autoencoder, GRU autoencoder (RNN)
  - **GAN-based**: GAIN-style imputation
- **Checkpointing and caching**:
  - Per-group CSV checkpoints
  - Shared-prefix cache to resume incomplete runs without recomputing earlier groups
- **Robust logging**:
  - Per-rank log files in `logs/`
  - Shared log file for global progress
- **Diagnostics & plots**:
  - Cumulative bar plots of rows imputed per group and method
  - QA reports on final outputs (NaN counts, dataset shape, source file)

---


- **Run with MPI:**
mpirun -np 8 python mpi_groups_thresholds.py
