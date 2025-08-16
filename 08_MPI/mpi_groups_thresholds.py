"""
======================================================================================================
CIR-03: Hierarchical Imputation Framework (MPI-Parallelized)
======================================================================================================

This script implements a **dynamic hierarchical imputation framework** for clinical datasets with 
missing values. The goal is to fill in missing values row-wise based on the percentage of missingness 
per row, applying increasingly sophisticated methods as missingness increases.

Key Features:
------------------------------------------------------------------------------------------------------
 - Utilizes MPI (via `mpi4py`) to parallelize imputation across multiple CPU cores/nodes.
 - Groups rows into imputation buckets based on % of missing values (e.g., 0–5%, 5–10%, ..., 90–100%).
 - Applies different imputation techniques per group:
 -   - Simple (mean, median, KNN)
 -   - Model-based (Iterative, XGBoost)
 -   - Deep Learning (LSTM, GRU)
 -   - GAN-based (GAIN-style)
 - Caches intermediate results to avoid recomputation.
 - Logs detailed progress, runtime, and method usage per row.
 - Generates diagnostic plots showing cumulative rows imputed per method and group.
 - Saves results and logs per MPI rank in organized folders (`CSV/exports/CIR-16/impute/...`).

Usage Context:
------------------------------------------------------------------------------------------------------
This framework is used in medical AI research, especially for ICU time-series datasets (e.g., MIMIC-IV, 
eICU), where missingness is not random and requires methodologically robust imputation strategies.

Developed by Dimitrios Dimopoulos
PhD Candidate — University of the Aegean
Research Focus: AI in Medical Data, Missing Value Imputation, Predictive Modeling
======================================================================================================
"""

# ---- Environment (set BEFORE importing TensorFlow) ----
import os
# Silence most TF C++ warnings (0=all, 1=INFO, 2=WARNING, 3=ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# keep TF from grabbing too many CPU threads when using MPI
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# ---- Standard libs ----
import sys
import io
import glob
import time
import logging
import copy
import errno
import gc
import contextlib
import re
import hashlib
import json



# ---- Third-party ----
from mpi4py import MPI
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Machine learning / DL
import tensorflow as tf  # keep ONLY tf.keras (no standalone 'keras')
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Concatenate, GRU, Dropout
from tensorflow.keras.optimizers import Adam

from xgboost import XGBRegressor

from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (enables IterativeImputer)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from collections import defaultdict

# Quiet absl logs that TF emits
logging.getLogger('absl').setLevel(logging.ERROR)


"""
----------------------------------------------------------
"""

@contextlib.contextmanager
def file_lock(path):
    """
    Atomic file lock using O_EXCL. Yields True if acquired, False if not.
    Lock is released by deleting the file on exit.
    """
    got = False
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        got = True
        yield True
    except FileExistsError:
        yield False
    except OSError as e:
        if e.errno == errno.EEXIST:
            yield False
        else:
            raise
    finally:
        if got and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass



# === Shared-prefix cache helpers ============================================
def _prefix_hash(parts):
    """Stable short key for a prefix signature."""
    s = "|".join(parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]

def _shared_prefix_dir(dataset_name, gidx, method_names, thresholds, columns):
    """
    Directory to store/load the cumulative 'previous_imputed' snapshot
    up to and including group index gidx (0-based).
    Cache key = dataset + thresholds + methods[:gidx+1] + column schema.
    """
    prefix_methods = method_names[:gidx+1]
    key_parts = [
        str(dataset_name or ""),
        json.dumps(thresholds, sort_keys=True),
        json.dumps(prefix_methods, sort_keys=True),
        json.dumps(sorted(list(columns)), sort_keys=True),
    ]
    h = _prefix_hash(key_parts)
    return os.path.join(
        "CSV/exports/CIR-16/impute/shared_prefix_cache",
        f"g{gidx+1:02d}",
        str(dataset_name or "unknown"),
        h
    )

def _shared_prefix_csv_path(shared_dir):
    """Canonical CSV path inside a shared-prefix cache directory."""
    os.makedirs(shared_dir, exist_ok=True)
    return os.path.join(shared_dir, "imputed.csv")
# ============================================================================



"""
----------------------------------------------------------
"""

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rank = MPI.COMM_WORLD.Get_rank()
print(f"[Rank {rank}] Initialized successfully.")



# ------------------------------
# MPI Task Farm (Master–Worker)
# ------------------------------
TAG_READY  = 1
TAG_JOB    = 2
TAG_RESULT = 3
TAG_STOP   = 4
TAG_ERROR  = 5



""" Logging Config"""
# === Initial logger setup ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variable to hold the active file handler
current_file_handler = None

# Stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[Rank %(mpi_rank)d] %(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


# Add custom attribute for MPI rank
class MPIRankFilter(logging.Filter):
    def filter(self, record):
        record.mpi_rank = rank
        return True

logger.addFilter(MPIRankFilter())

# === Optional: Per-rank log file ===
per_rank_log_path = f'logs/rank_{rank}.log'
os.makedirs(os.path.dirname(per_rank_log_path), exist_ok=True)

rank_file_handler = logging.FileHandler(per_rank_log_path, mode='w')
rank_file_handler.setFormatter(formatter)
logger.addHandler(rank_file_handler)

# === Function to switch to a shared log file ===
def switch_log_file(filename_base):
    global current_file_handler

    # Remove old handler if it exists
    if current_file_handler:
        logger.removeHandler(current_file_handler)
        current_file_handler.close()

    # Ensure directory exists
    log_dir = os.path.dirname(filename_base)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Single shared log file (same for all ranks)
    filename = f"{filename_base}"

    # Create new file handler
    current_file_handler = logging.FileHandler(filename, mode='a')  # append mode
    current_file_handler.setFormatter(formatter)
    logger.addHandler(current_file_handler)

    logger.info(f"Switched logging to shared file {filename}")


"""
----------------------------------------------------------
"""

""" Rank-0 dataset discovery (map) + broadcast to workers """

""" dataset_map is now available everywhere and holds the path for any dataset by name (e.g., "o1_X_test"). """

# Build log file
switch_log_file('logs/CIR-2.log')
logger.info("This is being logged to CIR-2.log")

# Rank 0 discovers files and builds the dataset map
if rank == 0:
    data_path = "CSV/imports/split_set/without_multiple_rows"
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
    # Map: "o1_X_test" -> "/.../o1_X_test.csv"
    dataset_map = {
        f.replace(".csv", "").replace("-", "_"): os.path.join(data_path, f)
        for f in all_files
    }
    logging.info("+++++++++++++++++CIR-2+++++++++++++++++++++++++")
    logging.info(f"[Start] Rank 0 built dataset map with {len(dataset_map)} entries.")

else:
    dataset_map = None

# Broadcast the map to all ranks
dataset_map = comm.bcast(dataset_map, root=0)
logging.info(f"[Rank {rank}] Received dataset map with {len(dataset_map)} entries.")

# Optional: list what this rank can see (small log)
if rank == 0:
    logging.info(f"[Complete] Rank 0 broadcast dataset map to all ranks.")
logging.info("++++++++++++++++++++++++++++++++++++++++++")

# === MPI Synchronization Barrier ===
comm.Barrier()
logging.info(f"[Rank {rank}] All ranks reached synchronization barrier after dataset map broadcast.")



"""
----------------------------------------------------------
"""

"""
Impute missing values using XGBoost regression for each column independently.
"""
def xgboost_imputer(df, random_state=0):
    df_imputed = df.copy()

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # Skip fully observed columns

        # Split rows with and without missing values in this column
        not_null_idx = df[col].notnull()
        null_idx = df[col].isnull()

        X_train = df.loc[not_null_idx].drop(columns=[col])
        y_train = df.loc[not_null_idx, col]
        X_pred = df.loc[null_idx].drop(columns=[col])

        # Skip if nothing to predict
        if X_pred.empty:
            continue

        # Drop columns that are completely NaN
        X_train = X_train.dropna(axis=1, how='all')
        X_pred = X_pred[X_train.columns]  # keep same columns

        # Fill remaining NaNs with column means (simple fallback)
        X_train = X_train.fillna(X_train.mean())
        X_pred = X_pred.fillna(X_train.mean())

        # Train XGBoost model
        model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
            n_jobs=1 # one thread per rank
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_pred)

        # Impute predicted values
        df_imputed.loc[null_idx, col] = y_pred

    return df_imputed


"""
----------------------------------------------------------
"""


"""
Impute missing values using an LSTM autoencoder.
Works best for dense rows (e.g., <40% missing).
"""

# Cache the model outside the function (top-level variable)
_lstm_model = None

def lstm_imputer(df, random_state=0, epochs=1000, batch_size=64):
    global _lstm_model

    df_copy = df.copy()
    idx = df_copy.index
    cols = df_copy.columns

    # Fill missing values and normalize
    df_filled = df_copy.fillna(df_copy.mean())
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_filled)
    X = df_scaled.reshape((df_scaled.shape[0], 1, df_scaled.shape[1]))
    input_dim = X.shape[2]

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Only build the model once
    if _lstm_model is None:
        input_layer = Input(shape=(1, input_dim))
        encoded = LSTM(64, activation="relu", return_sequences=False)(input_layer)
        repeated = RepeatVector(1)(encoded)
        decoded = LSTM(input_dim, activation="sigmoid", return_sequences=True)(repeated)
        _lstm_model = Model(inputs=input_layer, outputs=decoded)
        _lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Train and log loss
    for epoch in range(epochs):
        history = _lstm_model.fit(X, X, epochs=1, batch_size=batch_size, verbose=0)
        if epoch % 100 == 0 or epoch == epochs - 1:
            logging.info(f"[LSTM Epoch {epoch}] Loss: {history.history['loss'][0]:.4f}")

    # Predict and inverse transform
    X_imputed = _lstm_model.predict(X, verbose=0)
    df_imputed_array = scaler.inverse_transform(X_imputed[:, 0, :])
    df_imputed = pd.DataFrame(df_imputed_array, columns=cols, index=idx)

    # Only fill missing values
    for col in cols:
        missing_mask = df[col].isnull()
        df_copy.loc[missing_mask, col] = df_imputed.loc[missing_mask, col]

    return df_copy


"""
----------------------------------------------------------
"""

"""
GAN-style imputer for missing data based on GAIN.
Arguments:
    df (pd.DataFrame): Input dataframe with missing values.
    random_state (int): Seed for reproducibility.
    epochs (int): Number of training iterations.
    batch_size (int): Batch size for training.
Returns:
    pd.DataFrame: Imputed dataframe.
"""
def gan_imputer(df, random_state=0, epochs=1000, batch_size=128):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    df_copy = df.copy()
    cols = df_copy.columns
    idx = df_copy.index

    # ===== Step 1: Normalize & Create mask =====
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_copy.fillna(0))  # Fill NA with 0 for scaling
    mask = ~df_copy.isnull().values  # 1 where observed, 0 where missing

    data_dim = df_scaled.shape[1]
    
    # ===== Step 2: Generator =====
    def build_generator():
        inputs = Input(shape=(data_dim * 2,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(data_dim, activation='sigmoid')(x)
        return Model(inputs, x)

    # ===== Step 3: Discriminator =====
    def build_discriminator():
        inputs = Input(shape=(data_dim * 2,))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        x = Dense(data_dim, activation='sigmoid')(x)
        return Model(inputs, x)

    G = build_generator()
    D = build_discriminator()
    G.compile(loss='binary_crossentropy', optimizer=Adam(0.001))
    D.compile(loss='binary_crossentropy', optimizer=Adam(0.001))

    # ===== Step 4: Training =====
    for epoch in range(epochs):
        # === Consistent batch size to avoid retracing ===
        if df_scaled.shape[0] < batch_size:
            repeat_factor = int(np.ceil(batch_size / df_scaled.shape[0]))
            X_batch = np.tile(df_scaled, (repeat_factor, 1))[:batch_size]
            M_batch = np.tile(mask, (repeat_factor, 1))[:batch_size]
        else:
            batch_idx = np.random.choice(df_scaled.shape[0], batch_size, replace=False)
            X_batch = df_scaled[batch_idx]
            M_batch = mask[batch_idx]

        Z_batch = np.random.uniform(0, 0.01, size=X_batch.shape)
        X_hat = M_batch * X_batch + (1 - M_batch) * Z_batch
        G_input = np.concatenate([X_hat, M_batch], axis=1)

        G_sample = G.predict(G_input, verbose=0)
        X_fake = M_batch * X_batch + (1 - M_batch) * G_sample

        D_input_real = np.concatenate([X_batch, M_batch], axis=1)
        D_input_fake = np.concatenate([X_fake, M_batch], axis=1)

        D_loss_real = D.train_on_batch(D_input_real, M_batch)
        D_loss_fake = D.train_on_batch(D_input_fake, M_batch)

        # === Train Generator ===
        G_loss = G.train_on_batch(G_input, M_batch)

        if epoch % 100 == 0:
            logging.info(f"[{epoch}] D_loss: {(D_loss_real + D_loss_fake) / 2:.4f} | G_loss: {G_loss:.4f}")

    # ===== Step 5: Imputation =====
    Z_full = np.random.uniform(0, 0.01, size=df_scaled.shape)
    X_hat_full = mask * df_scaled + (1 - mask) * Z_full
    G_input_full = np.concatenate([X_hat_full, mask], axis=1)

    G_imputed = G.predict(G_input_full, verbose=0)
    X_final = mask * df_scaled + (1 - mask) * G_imputed

    df_imputed_array = scaler.inverse_transform(X_final)
    df_imputed = pd.DataFrame(df_imputed_array, columns=cols, index=idx)

    return df_imputed


"""
----------------------------------------------------------
"""


"""
Impute missing values using a GRU-based autoencoder.
"""

# Cache model to avoid retracing
_rnn_model = None

"""
Parameter patience enables the early stopping if the training doesn't produce
better MAE than the previews 10 or what number we put predictions.

The mask_ratio controls the percentage of the known values are hidden in order
to predict them.
    - Low (0.05–0.1) Only a few known values are hidden per row.
      Training is conservative but may not learn well.
    - Medium (0.2–0.3) Balanced — enough challenge for learning while
      preserving input context.
    - High (0.4–0.5+) Very challenging — model must infer much of the input,
      good for robustness, risky for small data.
"""
def rnn_imputer(df, random_state=0, epochs=1000, batch_size=64, mask_ratio=0.2, patience=10):
    global _rnn_model

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    df_copy = df.copy()
    idx = df_copy.index
    cols = df_copy.columns

    # Step 1: Fill initial NaNs with column mean
    df_filled = df_copy.fillna(df_copy.mean())

    # Step 2: Scale
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df_filled)

    # Save original
    X_original = df_scaled.copy()
    input_dim = X_original.shape[1]

    # Step 3: Build model once
    if _rnn_model is None:
        input_layer = Input(shape=(1, input_dim))
        encoded = GRU(64, activation='relu', return_sequences=False)(input_layer)
        encoded = Dropout(0.2)(encoded)  # Dropout added
        repeated = RepeatVector(1)(encoded)
        decoded = GRU(input_dim, activation='sigmoid', return_sequences=True)(repeated)
        _rnn_model = Model(inputs=input_layer, outputs=decoded)
        _rnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Step 4: Training loop with dynamic masking + early stopping
    best_mae = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(epochs):
        # === Dynamic masking each epoch ===
        mask = (np.random.rand(*X_original.shape) < mask_ratio)
        X_masked = X_original.copy()
        X_masked[mask] = 0

        X_input = X_masked.reshape((X_masked.shape[0], 1, X_masked.shape[1]))
        X_target = X_original.reshape((X_original.shape[0], 1, X_original.shape[1]))
        loss_mask = tf.convert_to_tensor(mask.reshape((mask.shape[0], 1, mask.shape[1])), dtype=tf.float32)

        # === Custom training step ===
        with tf.GradientTape() as tape:
            preds = _rnn_model(tf.convert_to_tensor(X_input, dtype=tf.float32))
            loss = tf.reduce_sum(tf.square((preds - X_target) * loss_mask)) / tf.reduce_sum(loss_mask)

        grads = tape.gradient(loss, _rnn_model.trainable_variables)
        _rnn_model.optimizer.apply_gradients(zip(grads, _rnn_model.trainable_variables))

        # === Benchmark MAE ===
        mae = tf.reduce_sum(tf.abs((preds - X_target) * loss_mask)) / tf.reduce_sum(loss_mask)

        # === Early stopping check ===
        if mae < best_mae:
            best_mae = mae
            best_weights = _rnn_model.get_weights()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"[RNN Epoch {epoch}] Early stopping triggered. Best MAE: {best_mae.numpy():.4f}")
                break

        if epoch % 100 == 0 or epoch == epochs - 1:
            logging.info(f"[RNN Epoch {epoch}] Masked Loss MSE: {loss.numpy():.4f} | MAE: {mae.numpy():.4f}")

    # Restore best weights
    if best_weights is not None:
        _rnn_model.set_weights(best_weights)

    # Step 5: Predict (impute)
    X_pred = _rnn_model.predict(X_original.reshape((X_original.shape[0], 1, input_dim)), verbose=0)
    X_imputed_array = scaler.inverse_transform(X_pred[:, 0, :])
    df_imputed = pd.DataFrame(X_imputed_array, columns=cols, index=idx)

    # Step 6: Replace only real missing values
    for col in cols:
        missing_mask = df[col].isnull()
        df_copy.loc[missing_mask, col] = df_imputed.loc[missing_mask, col]

    return df_copy



"""
----------------------------------------------------------
"""


# --- Tee class to redirect output to both stdout and logging ---
class Tee:
    def __init__(self, *files, use_logging=False):
        self.files = files
        self.use_logging = use_logging

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
        if self.use_logging:
            for line in obj.rstrip().splitlines():
                logging.info(line)

    def flush(self):
        for f in self.files:
            f.flush()

# --- Iterative Imputer Function ---
def impute_with_iterative(input_df, method, output_path, n_iter, log_verbose_file_path=None):
    logging.info(f"Starting Iterative Imputer with method={method} on input DataFrame of shape {input_df.shape}.")

    data_copy = input_df.copy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Estimator selection
    if method == "ExtraTrees":
        estimator = ExtraTreesRegressor(n_estimators=5, random_state=0, n_jobs=-1)
    elif method == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(random_state=0)
    elif method == "BayesianRidge":
        estimator = BayesianRidge()
    elif method == "Ridge":
        estimator = Ridge(alpha=1.0, random_state=0)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'ExtraTrees' or 'HistGradientBoosting'.")
    
    imputer = IterativeImputer(
        estimator=estimator,          
        max_iter=n_iter,              
        random_state=0,               
        verbose=2,                    
        sample_posterior=False,       
        tol=1e-3,                     
        initial_strategy='mean',      
        imputation_order='ascending'  
)

    start_time = time.time()

    if log_verbose_file_path is not None:
        os.makedirs(os.path.dirname(log_verbose_file_path), exist_ok=True)
        original_stdout = sys.stdout
        with open(log_verbose_file_path, "w") as log_file:
            sys.stdout = Tee(sys.__stdout__, log_file, use_logging=True)
            try:
                imputed_array = imputer.fit_transform(data_copy)
            finally:
                sys.stdout = original_stdout
    else:
        sys.stdout = Tee(sys.__stdout__, use_logging=True)
        try:
            imputed_array = imputer.fit_transform(data_copy)
        finally:
            sys.stdout = sys.__stdout__

    end_time = time.time()
    runtime = end_time - start_time

    # Retain original index to avoid downstream assignment errors
    imputed_df = pd.DataFrame(imputed_array, columns=data_copy.columns, index=data_copy.index)
    imputed_df.to_csv(output_path, index=False)

    logging.info(f"Imputation completed in {runtime:.2f} seconds.")
    logging.info(f"Number of NaNs after imputation: {np.isnan(imputed_df.values).sum()}")
    logging.info(f"Imputed dataset saved at {output_path}")

    #describe_output_path = output_path.replace(".csv", "_describe.csv")
    #imputed_df.describe().to_csv(describe_output_path)
    #logging.info(f"Basic statistics saved at {describe_output_path}")

    return imputed_df


"""
----------------------------------------------------------
"""


# --- Registry of Imputation Methods ---
imputer_registry = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "knn": KNNImputer(n_neighbors=5, weights="uniform"),
    "iterative_function": lambda df, random_state=0: impute_with_iterative(
        input_df=df,
        method="ExtraTrees", # BayesianRidge or ExtraTrees or Ridge
        output_path="imputed_outputs/tmp.csv",
        n_iter=8
    ),
    "iterative_simple": IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10, random_state=42),
                                   max_iter=10, random_state=42),
    "xgboost": xgboost_imputer,
    "gan": gan_imputer,
    "lstm": lstm_imputer,
    "rnn": rnn_imputer
}


"""
----------------------------------------------------------
"""

# Build log file
switch_log_file('logs/CIR-16.log')
logger.info("This is being logged to CIR-16.log")


"""
Dynamic hierarchical imputer using cumulative row-wise missingness and assigned methods.

Parameters:
    df (pd.DataFrame): Dataset with missing values.
    thresholds (list): List of group widths (must sum to ~1.0).
    method_names (list): List of method names (must match thresholds).
    method_registry (dict): Registered methods with keys as names and values as callables or sklearn objects.
    random_state (int): Random seed for reproducibility.
    return_method_log (bool): Return pd.Series logging method used per row.

Returns:
    imputed_df (pd.DataFrame)
    method_log (pd.Series) — only if return_method_log=True
"""

def hierarchical_impute_dynamic(
    df,
    thresholds,
    method_names,
    method_registry,
    random_state=0,
    return_method_log=False,
    dataset_name=None,
    plot_id=None
):
    if len(thresholds) != len(method_names):
        raise ValueError("The number of thresholds must match the number of methods.")

    df_copy = df.copy()
    df_copy["missing_pct"] = df_copy.isnull().mean(axis=1)
    cols = df_copy.columns.drop("missing_pct")
    col_list = list(cols)

    global_means = df_copy[cols].mean().fillna(0)
    global_min = df_copy[cols].min()
    global_max = df_copy[cols].max()

    imputed_df = pd.DataFrame(index=df_copy.index, columns=cols)
    method_log = pd.Series(index=df_copy.index, dtype="object")

    cum_thresholds = np.cumsum(thresholds)
    if not np.isclose(cum_thresholds[-1], 1.0):
        raise ValueError("Thresholds must sum to 1.0")

    n_groups = len(cum_thresholds)

    # ------------------------------------------------------------
    # Warm-start: try to load the longest cached shared prefix
    # ------------------------------------------------------------
    start_group = 0            # group index to start computing (0-based)
    previous_imputed = None    # cumulative df of all rows imputed so far

    try:
        for gidx in range(n_groups - 1, -1, -1):
            shared_dir = _shared_prefix_dir(
                dataset_name=dataset_name,
                gidx=gidx,
                method_names=method_names,
                thresholds=thresholds,
                columns=col_list
            )
            cached_path = _shared_prefix_csv_path(shared_dir)
            if os.path.exists(cached_path):
                warm = pd.read_csv(cached_path, index_col=0)
                # Ensure same columns / order
                if list(warm.columns) == col_list:
                    previous_imputed = warm
                    start_group = gidx + 1  # next group to compute
                    logging.info(f"[Rank {rank}] Warm-start {dataset_name}: using prefix through group {gidx+1:02d} -> {cached_path}")

                    # Pre-fill imputed_df & method_log for the cached part,
                    # so the final output is complete without recomputation.
                    for i_prefill in range(start_group):
                        upper_bound = cum_thresholds[i_prefill]
                        lower_bound = cum_thresholds[i_prefill - 1] if i_prefill > 0 else 0.0
                        idx_prefill = df_copy.index[
                            (df_copy["missing_pct"] > lower_bound) & (df_copy["missing_pct"] <= upper_bound)
                        ]
                        if len(idx_prefill) == 0:
                            continue
                        # Fill from cached cumulative
                        imputed_df.loc[idx_prefill] = previous_imputed.loc[idx_prefill, col_list]
                        method_log.loc[idx_prefill] = method_names[i_prefill]
                    break
                else:
                    logging.info(f"[Rank {rank}] Ignored cache (column mismatch): {cached_path}")
    except Exception as _e:
        logging.info(f"[Rank {rank}] Warm-start probing failed: {_e}")

    # ------------------------------------------------------------
    # Plot bookkeeping
    # ------------------------------------------------------------
    group_names = []
    cumulative_rows = []
    method_names_actual = []
    cumulative_total = 0

    # Per-sequence checkpoint directory (your existing per-group temps)
    checkpoint_dir = os.path.join(
        "CSV/exports/CIR-16/impute/threshold",
        f"seq_{plot_id:02d}_{'_'.join(method_names)}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # If we warmed up, also advance plot bookkeeping for already-cached groups
    for i_prefill in range(start_group):
        upper_bound = cum_thresholds[i_prefill]
        lower_bound = cum_thresholds[i_prefill - 1] if i_prefill > 0 else 0.0
        idx_prefill = df_copy.index[
            (df_copy["missing_pct"] > lower_bound) & (df_copy["missing_pct"] <= upper_bound)
        ]
        if len(idx_prefill) == 0:
            continue
        group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
        group_names.append(group_label)
        cumulative_total += len(idx_prefill)
        cumulative_rows.append(cumulative_total)
        method_names_actual.append(method_names[i_prefill])

    # ------------------------------------------------------------
    # Main loop — start from start_group (skip cached prefix)
    # ------------------------------------------------------------
    for i in range(start_group, n_groups):
        upper_bound = cum_thresholds[i]
        lower_bound = cum_thresholds[i - 1] if i > 0 else 0.0

        idx = df_copy.index[
            (df_copy["missing_pct"] > lower_bound) & (df_copy["missing_pct"] <= upper_bound)
        ]
        group_data = df_copy.loc[idx, cols].copy()

        # Stabilize all-NaN columns within this group
        for col in group_data.columns:
            if group_data[col].isnull().all():
                group_data[col] = global_means[col]

        if group_data.empty:
            # Still update plot bookkeeping for consistency
            group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
            group_names.append(group_label)
            cumulative_rows.append(cumulative_total)
            method_names_actual.append(method_names[i])
            continue

        method_name = method_names[i]
        imputer = get_imputer(method_name, method_registry)

        # Per-group checkpoint (sequence-local)
        temp_path = os.path.join(checkpoint_dir, f"{dataset_name}_group{i+1:02d}_temp.csv")
        if os.path.exists(temp_path):
            logging.info(f"[Rank {rank}] Skipping Group {i+1}/{n_groups} — checkpoint found: {temp_path}")
            group_imputed = pd.read_csv(temp_path, index_col=0)
            group_imputed = group_imputed.clip(lower=global_min, upper=global_max, axis=1)
            imputed_df.loc[idx] = group_imputed
            method_log.loc[idx] = method_name

            previous_imputed = (
                pd.concat([previous_imputed, group_imputed]).sort_index()
                if previous_imputed is not None else group_imputed.copy()
            )
            # Plot bookkeeping
            group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
            group_names.append(group_label)
            cumulative_total += len(group_data)
            cumulative_rows.append(cumulative_total)
            method_names_actual.append(method_name)

            # Persist updated shared-prefix cache (now includes this group)
            try:
                shared_dir = _shared_prefix_dir(dataset_name, i, method_names, thresholds, col_list)
                shared_csv = _shared_prefix_csv_path(shared_dir)
                lock_path = os.path.join(shared_dir, "cache.lock")
                with file_lock(lock_path) as got:
                    if got:
                        tmp_out = shared_csv + ".part"
                        previous_imputed.to_csv(tmp_out)  # keep index to preserve original row ids
                        os.replace(tmp_out, shared_csv)
                        logging.info(f"[Rank {rank}] Updated shared prefix cache: {shared_csv}")
            except Exception as _e:
                logging.info(f"[Rank {rank}] Could not update shared prefix cache for g{i+1}: {_e}")

            continue

        # Build combined rows to fit stats/models
        combined = group_data if previous_imputed is None else pd.concat([previous_imputed, group_data])

        logging.info(
            f"[{dataset_name}][Group {i+1}] "
            f"({lower_bound:.2f}, {upper_bound:.2f}] -> {method_name} | "
            f"{len(group_data)} group rows | {len(combined)} total used rows"
        )

        # Impute
        try:
            # If it is an sklearn-like imputer with fit/transform, do:
            #   1) fit on 'combined' (A..current)
            #   2) transform only 'group_data' (current group)
            if hasattr(imputer, "fit") and hasattr(imputer, "transform"):
                # Fit on all available rows so far
                imputer.fit(combined)
                # Transform only the target group rows
                group_imputed_arr = imputer.transform(group_data)
                group_imputed = pd.DataFrame(
                    group_imputed_arr, columns=combined.columns, index=group_data.index
                )[group_data.columns]  # enforce column order

            # Otherwise, fall back to function-style imputers that return a full DF
            else:
                try:
                    combined_imputed = imputer(combined, random_state=random_state)
                except TypeError:
                    combined_imputed = imputer(combined)
                group_imputed = combined_imputed.loc[idx]

        except Exception as e:
            logging.exception(
                f"[Rank {rank}] Error during method '{method_name}' in sequence #{plot_id} "
                f"on group ({lower_bound:.2f}, {upper_bound:.2f}]: {e}"
            )
            continue


        # Clip to global min/max
        group_imputed = group_imputed.clip(lower=global_min, upper=global_max, axis=1)

        # Save per-sequence group checkpoint
        try:
            group_imputed.to_csv(temp_path)  # keep index so we can re-align on load
        except Exception as _e:
            logging.info(f"[Rank {rank}] Could not write checkpoint {temp_path}: {_e}")

        # Write into final frame + log
        imputed_df.loc[idx] = group_imputed
        method_log.loc[idx] = method_name

        # Update cumulative
        previous_imputed = (
            pd.concat([previous_imputed, group_imputed]).sort_index()
            if previous_imputed is not None else group_imputed.copy()
        )

        # Persist/refresh shared-prefix cache (cumulative up to and including group i)
        try:
            shared_dir = _shared_prefix_dir(dataset_name, i, method_names, thresholds, col_list)
            shared_csv = _shared_prefix_csv_path(shared_dir)
            lock_path = os.path.join(shared_dir, "cache.lock")
            with file_lock(lock_path) as got:
                if got:
                    tmp_out = shared_csv + ".part"
                    previous_imputed.to_csv(tmp_out)  # keep index
                    os.replace(tmp_out, shared_csv)
                    logging.info(f"[Rank {rank}] Updated shared prefix cache: {shared_csv}")
                else:
                    logging.info(f"[Rank {rank}] Shared prefix cache busy; skipped write for g{i+1}")
        except Exception as _e:
            logging.info(f"[Rank {rank}] Could not update shared prefix cache for g{i+1}: {_e}")

        # Plot bookkeeping
        group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
        group_names.append(group_label)
        cumulative_total += len(group_data)
        cumulative_rows.append(cumulative_total)
        method_names_actual.append(method_name)

        logging.info(f"[Rank {rank}] Finished group {i+1} / {n_groups} for sequence #{plot_id}")

    # Final sanity
    if imputed_df.isnull().values.any():
        raise ValueError("NaNs remain after hierarchical imputation!")

    # === Plot cumulative bar ===
    output_dir = "figures/CIR-16"
    os.makedirs(output_dir, exist_ok=True)

    unique_methods = list(set(method_names_actual)) or ["(none)"]
    palette = sns.color_palette("tab10", n_colors=len(unique_methods))
    method_color_map = {method: palette[i] for i, method in enumerate(unique_methods)}
    colors = [method_color_map[m] for m in method_names_actual]

    plt.figure(figsize=(10, 10))
    plt.barh(
        y=range(1, len(cumulative_rows) + 1),
        width=cumulative_rows,
        color=colors,
        edgecolor='black'
    )
    plt.yticks(ticks=range(1, len(group_names) + 1), labels=group_names)
    plt.title(f"Cumulative Rows Used for Imputation by Group - {dataset_name}", fontsize=14, fontweight='bold')
    plt.ylabel("Missingness Range", fontsize=12)
    plt.xlabel("Cumulative Rows Used", fontsize=12)
    plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    legend_handles = [Patch(color=color, label=method) for method, color in method_color_map.items()]
    if legend_handles:
        plt.legend(handles=legend_handles, title="Imputation Method", loc="lower right")
    plt.tight_layout()

    filename = f"seq_{plot_id:02d}_{dataset_name}_cumulative_imputation_rows.png" if dataset_name and plot_id is not None else f"{dataset_name}_cumulative_imputation_rows.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    
    #---------------------------------|
    # Temporarily disabled.           |
    # Keep checkpoints for inspection |
    #---------------------------------|
    # === Cleanup group temp files ===
    #if dataset_name and plot_id is not None:
    #    for i in range(n_groups):
    #        temp_path = os.path.join(checkpoint_dir, f"{dataset_name}_group{i+1:02d}_temp.csv")
    #        if os.path.exists(temp_path):
    #            os.remove(temp_path)
    
    return (imputed_df, method_log) if return_method_log else imputed_df

"""
----------------------------------------------------------
"""



def get_imputer(method_name, registry):
    imputer = registry.get(method_name)
    if imputer is None:
        raise ValueError(f"Method '{method_name}' not found or not implemented.")

    if callable(imputer):
        return imputer  # Don't copy or modify lambdas — just return

    if hasattr(imputer, "fit") and hasattr(imputer, "transform"):
        return copy.deepcopy(imputer)

    return imputer


"""
----------------------------------------------------------
"""




methods_all = ["mean", "median", "knn", "iterative_simple", "iterative_function", "xgboost", "gan", "lstm", "rnn"]

# methods_all = ["mean", "median", "knn", "iterative_function", "xgboost", "gan", "lstm", "rnn"]

# Define all datasets
datasets = [
    "o1_X_test", "o1_X_validate", "o1_X_train"
    #"o1_X_train", "o1_X_validate", "o1_X_test", "o1_X_external",
    #"o2_X_train", "o2_X_validate", "o2_X_test", "o2_X_external",
    #"o3_X_train", "o3_X_validate", "o3_X_test", "o3_X_external",
    #"o4_X_train", "o4_X_validate", "o4_X_test", "o4_X_external"
]


"""
------------------Build Sequenses----------------------------------------
"""
# Define the fixed sequence builder


import itertools

def build_sequences():
    
    #------------------------0%------------------------------
    # Jerez et al., 2010; Batista & Monard, 2003; Che et al., 2018.
    
    group_A =  ["knn"] #["knn"] #["knn", "median", "mean"] #5 #1
    group_B =  ["knn"] #["knn"] #["knn", "median", "mean"] #5 #2
    group_C =  ["knn"] #["knn"] #["knn", "median", "mean"] #5 #3
    group_D =  ["iterative_function"] #["knn", "median", "mean"] #5 #4
    group_E =  ["iterative_function"] #["knn", "median", "mean"] #5 #5
    group_F =  ["iterative_function"] #["knn", "median", "mean"] #5 #6
    
    #------------------------30%------------------------------
    # Bertsimas et al., 2018 (data-driven MICE with tree-based models); Lin et al., 2020 (XGBoost outperforms for ICU datasets).
    
    group_G = ["iterative_function"] #["iterative_function", "xgboost"] #5 #7
    group_H = ["xgboost"] #["xgboost", "iterative_function"] #5 #8 
    group_I = ["xgboost"] #["xgboost", "iterative_function"] #5 #9
    group_J = ["xgboost"] #["xgboost", "iterative_function"] #5 #10
    group_K = ["xgboost"] #["xgboost", "iterative_function"] #5 #11
    group_L = ["xgboost"] #["xgboost", "iterative_function"] #5 #12
   
    #------------------------60%------------------------------
    # Yoon et al., 2019 (BRITS), Cao et al., 2018 (GRU-D).
    
    group_M = ["lstm", "rnn" ] #5 #13
    group_N = ["lstm", "rnn"] #5 #14
    group_O = ["lstm", "rnn"] #5 #15
    group_P = ["lstm", "rnn"] #5 #16
    
    #------------------------80%------------------------------
    # Yoon et al., 2018 (GAIN); Luo et al., 2020.
    
    group_Q = ["gan", "rnn"] #10 #17
    group_R = ["gan", "rnn"] #10 #18
    
    #------------------------60%------------------------------

    thresholds = [0.05]*16 + [0.10]*2  # groups = 100%
    work_items = []
    index_records = []

    idx = 0
    for a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r in itertools.product(group_A, group_B, group_C, group_D, group_E, group_F, group_G, group_H, group_I, group_J, group_K, group_L, group_M, group_N, group_O, group_P, group_Q, group_R):
        method_names = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r]
        folder_name = f"seq_{idx:03d}_{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h}_{i}_{j}_{k}_{l}_{m}_{n}_{o}_{p}_{q}_{r}"

        for dataset in datasets:  # use datasets from global scope
            work_items.append((idx, folder_name, dataset, thresholds, method_names))
            index_records.append({
                "sequence_id": folder_name,
                "methods": " | ".join(method_names),
                "num_methods": len(method_names),
                "dataset": dataset
            })

        idx += 1

    return work_items, index_records


"""
----------------------------------------------------------
"""

# Custom function to split work across ranks without numpy
def split_evenly(lst, n):
    """Split list `lst` into `n` nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# --- Build sequences and combinations ---

work_items = []
index_records = []
timing_records = {}



# Use the new rotational sequence builder
work_items, index_records = build_sequences()
print(f"Total sequences generated: {len(work_items)}")


# === Distribute work across ranks ===
items_per_rank = split_evenly(work_items, size)
local_work = items_per_rank[rank]
print(f"[Rank {rank}] Received {len(local_work)} work items out of {len(work_items)} total.")

# === Ensure base output path exists ===
base_output_root = "CSV/exports/CIR-16/impute/threshold"
os.makedirs(base_output_root, exist_ok=True)



"""
----------------------------------------------------------
"""

"""On-demand loading + timing gather"""


"""
Workers load only the dataset they are currently working on (_load_dataset_by_name).

A tiny per-rank cache _dataset_cache avoids repeated disk I/O if a rank works on the same dataset again.

We gather timing from all ranks to rank 0 so your summary file includes everyone’s timings (previously rank 0 only had its own).

"""



# --- Build sequences and combinations ---
work_items, index_records = build_sequences()
print(f"Total sequences generated: {len(work_items)}")

# === Ensure base output path exists ===
base_output_root = "CSV/exports/CIR-16/impute/threshold"
os.makedirs(base_output_root, exist_ok=True)

# ------------------------------
# On-demand dataset loader (per-rank cache)
# ------------------------------
_dataset_cache = {}

def _load_dataset_by_name(name: str) -> pd.DataFrame:
    """Load a dataset by name using the broadcasted dataset_map. Cached per rank."""
    if name not in dataset_map:
        logging.warning(f"[Rank {rank}] Dataset '{name}' not found in dataset_map. Skipping.")
        return None
    if name in _dataset_cache:
        return _dataset_cache[name]
    path = dataset_map[name]
    logging.info(f"[Rank {rank}] Loading dataset '{name}' from {path}")
    df = pd.read_csv(path).astype('float32')
    _dataset_cache[name] = df
    logging.info(f"[Rank {rank}] Loaded '{name}' with shape {df.shape}")
    return df


# ------------------------------


def _job_already_done(dataset_name: str, method_names: list, idx: int) -> bool:
    method_sequence_str = "_".join(method_names)
    folder_name = f"seq_{idx:02d}_{method_sequence_str}"
    subfolder_path = os.path.join(base_output_root, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # If canonical file exists, we’re done.
    canonical = os.path.join(subfolder_path, f"{dataset_name}.csv")
    if os.path.exists(canonical):
        return True

    # Or if any rank-specific output exists.
    imputed_path_pattern = os.path.join(subfolder_path, f"{dataset_name}_rank*.csv")
    existing_files = glob.glob(imputed_path_pattern)
    return len(existing_files) > 0


# ------------------------------

def acquire_job_lock(idx: int, dataset_name: str, method_names: list):
    """
    Returns a context manager. Use:
        with acquire_job_lock(...) as got:
            if not got: ...skip...
            # do ALL the work while holding the lock
    """
    method_sequence_str = "_".join(method_names)
    folder_name = f"seq_{idx:02d}_{method_sequence_str}"
    subfolder_path = os.path.join(base_output_root, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    lock_path = os.path.join(subfolder_path, f"{dataset_name}.lock")
    return file_lock(lock_path)



# ------------------------------


def _do_one_job_locally(job):
    idx = job["idx"]
    dataset_name = job["dataset_name"]
    thresholds = job["thresholds"]
    method_names = job["method_names"]

    method_sequence_str = "_".join(method_names)
    folder_name = f"seq_{idx:02d}_{method_sequence_str}"
    subfolder_path = os.path.join(base_output_root, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    # Acquire the job lock for the whole duration of the job
    with acquire_job_lock(idx, dataset_name, method_names) as got_lock:
        if not got_lock:
            logging.info(f"[Rank {rank}] Another rank holds lock for #{idx:02d} / {dataset_name}. Skipping.")
            return {"idx": idx, "dataset": dataset_name, "duration": 0.0, "skipped": True}

        # Double-check after acquiring the lock (prevents race with just-finished job)
        if _job_already_done(dataset_name, method_names, idx):
            logging.info(f"[Rank {rank}] Skipping sequence #{idx:02d} on {dataset_name} — outputs already exist.")
            return {"idx": idx, "dataset": dataset_name, "duration": 0.0, "skipped": True}

        logging.info(f"[Rank {rank}] Processing sequence #{idx:02d} on {dataset_name} using: {' | '.join(method_names)}")

        df = _load_dataset_by_name(dataset_name)
        if df is None or not isinstance(df, pd.DataFrame):
            msg = f"[Rank {rank}] Skipping {dataset_name} (not found or not a DataFrame)"
            logging.info(msg)
            return {"idx": idx, "dataset": dataset_name, "duration": 0.0, "error": msg}

        try:
            start_time = time.time()

            imputed_df, method_log = hierarchical_impute_dynamic(
                df=df,
                thresholds=thresholds,
                method_names=method_names,
                method_registry=imputer_registry,
                random_state=0,
                return_method_log=True,
                dataset_name=dataset_name,
                plot_id=idx
            )

            duration = time.time() - start_time

            imputed_path = os.path.join(subfolder_path, f"{dataset_name}_rank{rank}.csv")
            method_log_path = os.path.join(subfolder_path, f"{dataset_name}_method_log_rank{rank}.csv")

            imputed_df.to_csv(imputed_path, index=False)
            method_log.to_csv(method_log_path, index=False)

            logging.info(f"[Rank {rank}] Saved: {imputed_path}")

            # free memory between big jobs
            del imputed_df, method_log
            gc.collect()
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

            return {"idx": idx, "dataset": dataset_name, "duration": duration}

        except Exception as e:
            logging.exception(f"[Rank {rank}] ❌ Exception during step #{idx} on {dataset_name}")
            return {"idx": idx, "dataset": dataset_name, "duration": 0.0, "error": str(e)}



# ------------------------------


def _master_loop(all_jobs):
    """Rank 0: dispatch jobs to workers that announce READY. Collect timing & errors."""
    num_workers = size - 1
    if num_workers <= 0:
        # Fallback: single-process run
        results = []
        for j in all_jobs:
            res = _do_one_job_locally(j)
            results.append(res)
        return results

    # Convert work list to a queue we can pop from
    job_queue = list(all_jobs)
    in_flight = 0
    stopped = 0
    results = []

    logging.info(f"[Rank 0] Master starting task farm with {num_workers} workers and {len(job_queue)} jobs.")

    while stopped < num_workers:
        status = MPI.Status()
        # Receive any incoming control msg from any worker
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        src = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            # Worker is ready; give it a job or tell it to stop
            if job_queue:
                job = job_queue.pop(0)
                # Quick skip check: if already done, don't waste worker time
                if _job_already_done(job["dataset_name"], job["method_names"], job["idx"]):
                    results.append({"idx": job["idx"], "dataset": job["dataset_name"], "duration": 0.0, "skipped": True})
                    # Immediately send another job if available; otherwise STOP
                    if job_queue:
                        next_job = job_queue.pop(0)
                        comm.send(next_job, dest=src, tag=TAG_JOB)
                        in_flight += 1
                    else:
                        comm.send(None, dest=src, tag=TAG_STOP)
                        stopped += 1
                else:
                    comm.send(job, dest=src, tag=TAG_JOB)
                    in_flight += 1
            else:
                # No more jobs: stop this worker
                comm.send(None, dest=src, tag=TAG_STOP)
                stopped += 1

        elif tag == TAG_RESULT:
            # Worker finished one job
            results.append(msg)  # msg is result dict
            in_flight = max(0, in_flight - 1)

        elif tag == TAG_ERROR:
            # Worker reports an error but keeps going
            logging.error(f"[Rank 0] Worker {src} reported error: {msg}")
        else:
            logging.warning(f"[Rank 0] Received unknown tag {tag} from {src}")

    logging.info(f"[Rank 0] Master finished: collected {len(results)} results.")
    return results


# ------------------------------


def _worker_loop():
    """Ranks 1..N: announce READY, receive JOBs, run, send RESULT, repeat until STOP."""
    while True:
        comm.send(None, dest=0, tag=TAG_READY)
        status = MPI.Status()
        job = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            break
        elif tag == TAG_JOB:
            res = _do_one_job_locally(job)
            # Send result back
            comm.send(res, dest=0, tag=TAG_RESULT)
        else:
            # Unexpected message/tag
            comm.send(f"Unexpected tag {tag}", dest=0, tag=TAG_ERROR)


# ------------------------------
# Build the job list from work_items
# ------------------------------


all_jobs = []
for idx, _, dataset_name, thresholds, method_names in work_items:
    all_jobs.append({
        "idx": idx,
        "dataset_name": dataset_name,
        "thresholds": thresholds,
        "method_names": method_names,
    })


# ------------------------------
# Run the farm
# ------------------------------


if rank == 0:
    results = _master_loop(all_jobs)
    # Prepare timing map for later logging
    merged_timing = {}
    for r in results:
        if "duration" in r and r.get("duration", 0) > 0:
            merged_timing[(r["idx"], r["dataset"])] = r["duration"]
else:
    _worker_loop()
    merged_timing = None

# Broadcast/collect merged timing to rank 0 (in case you want it elsewhere later)
if rank == 0:
    all_timing_records = merged_timing
else:
    all_timing_records = None

comm.Barrier()


# ------------------------------


def _finalize_outputs(base_dir: str, datasets_local: list, work_items_local: list):
    """
    Rank-0: For each (sequence, dataset), pick the produced _rank*.csv,
    run final QA, and write a canonical {dataset}.csv next to them.
    """
    by_seq = {}
    for idx, _, dataset_name, thresholds, method_names in work_items_local:
        key = (idx, "_".join(method_names))
        by_seq.setdefault(key, set()).add(dataset_name)

    for (idx, seq_str), dsets in by_seq.items():
        seq_folder = f"seq_{idx:02d}_{seq_str}"
        seq_path = os.path.join(base_dir, seq_folder)
        if not os.path.isdir(seq_path):
            continue

        for dataset_name in sorted(dsets):
            # any rank output
            cand = sorted(glob.glob(os.path.join(seq_path, f"{dataset_name}_rank*.csv")))
            if not cand:
                logging.warning(f"[Rank 0] No outputs found for {seq_folder}/{dataset_name}")
                continue

            # choose the largest file (most rows/bytes) as best
            best = max(cand, key=lambda p: os.path.getsize(p))
            try:
                df = pd.read_csv(best)
                # sanity checks
                nan_ct = int(np.isnan(df.values).sum())
                if nan_ct > 0:
                    logging.warning(f"[Rank 0] {seq_folder}/{dataset_name}: {nan_ct} NaNs remain. Keeping anyway for traceability.")

                # write canonical file
                final_path = os.path.join(seq_path, f"{dataset_name}.csv")
                df.to_csv(final_path, index=False)

                # light QA report
                qa_path = os.path.join(seq_path, f"{dataset_name}_QA.txt")
                with open(qa_path, "w", encoding="utf-8") as f:
                    f.write(f"Sequence: {seq_folder}\nDataset: {dataset_name}\n")
                    f.write(f"Rows, Cols: {df.shape[0]}, {df.shape[1]}\n")
                    f.write(f"Total NaNs: {nan_ct}\n")
                    f.write(f"Source file: {os.path.basename(best)}\n")

                logging.info(f"[Rank 0] Finalized {seq_folder}/{dataset_name} → {os.path.basename(final_path)}")
            except Exception:
                logging.exception(f"[Rank 0] Finalize failed for {seq_folder}/{dataset_name}")




"""
----------------------------------------------------------
"""


# === Rank 0 saves the index file and description log ===


def _extract_seq_idx(sequence_id: str):
    # Accepts 'seq_000_knn_...'; returns int 0
    m = re.match(r"seq_(\d+)_", sequence_id)
    return int(m.group(1)) if m else None

comm.Barrier()
if rank == 0:
    # 1) Save the flat index CSV
    index_df = pd.DataFrame(index_records)
    index_df_path = os.path.join(base_output_root, "imputation_sequence_index.csv")
    index_df.to_csv(index_df_path, index=False)
    logging.info(f"[Rank 0] Saved index file: {index_df_path}")

    # 2) Group records by sequence_id for a cleaner text log
    by_seq = defaultdict(list)
    for rec in index_records:
        by_seq[rec["sequence_id"]].append(rec)

    # Choose which timing map we actually have
    timings = all_timing_records if 'all_timing_records' in globals() and all_timing_records else {}

    # 3) Human-readable description
    text_log_path = os.path.join(base_output_root, "sequence_description.txt")
    with open(text_log_path, "w", encoding="utf-8") as f:
        for sequence_id, recs in sorted(by_seq.items()):
            # Header line: sequence_id and its method string (all recs share the same methods)
            methods_line = recs[0]["methods"] if "methods" in recs[0] else ""
            f.write(f"{sequence_id}: {methods_line}\n")

            # Determine the numeric sequence index for timing lookups
            seq_idx = recs[0].get("seq_idx", None)
            if seq_idx is None:
                seq_idx = _extract_seq_idx(sequence_id)

            # Append dataset timing lines if available
            for rec in recs:
                ds = rec.get("dataset")
                if ds is None:
                    continue
                dur = timings.get((seq_idx, ds)) if seq_idx is not None else None
                if dur is not None:
                    f.write(f"    {ds}: {dur:.2f} sec\n")
    logging.info(f"[Rank 0] Saved sequence description log to {text_log_path}")

"""
----------------------------------------------------------
"""