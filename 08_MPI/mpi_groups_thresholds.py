"""
======================================================================================================
CIR-03: Hierarchical Imputation Framework (MPI-Parallelized)
======================================================================================================

This script implements a **dynamic hierarchical imputation framework** for clinical datasets with 
missing values. The goal is to fill in missing values row-wise based on the percentage of missingness 
per row, applying increasingly sophisticated methods as missingness increases.

Key Features:
------------------------------------------------------------------------------------------------------
 - Utilizes MPI (via mpi4py) to parallelize imputation across multiple CPU cores/nodes.
 - Groups rows into imputation buckets based on % of missing values (e.g., 0–5%, 5–10%, ..., 90–100%).
 - Applies different imputation techniques per group:
 -   - Simple (mean, median, KNN)
 -   - Model-based (Iterative, XGBoost)
 -   - Deep Learning (LSTM, GRU)
 -   - GAN-based (GAIN-style)
 - Caches intermediate results to avoid recomputation.
 - Logs detailed progress, runtime, and method usage per row.
 - Generates diagnostic plots showing cumulative rows imputed per method and group.
 - Saves results and logs per MPI rank in organized folders (CSV/exports/CIR-16/impute/...).

Usage Context:
------------------------------------------------------------------------------------------------------
This framework is used in medical AI research, especially for ICU time-series datasets (e.g., MIMIC-IV, 
eICU), where missingness is not random and requires methodologically robust imputation strategies.

Developed by: [Dimopoulos Dimitrios]
======================================================================================================
"""

import os
# Pin math libraries to 1 thread per MPI rank
# This block forces those libraries to run single-threaded per MPI rank.

# Sets the maximum threads (used by NumPy, SciPy, scikit-learn, XGBoost, etc.) to 1.
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Limits OpenBLAS (BLAS implementation used by NumPy) to 1 thread.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
# Limits Intel MKL (Math Kernel Library, used by some NumPy builds) to 1 thread.
os.environ.setdefault("MKL_NUM_THREADS", "1")
# 
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# Limits NumExpr (fast numerical expression evaluator) to 1 thread.
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Turns off OpenBLAS debug output (so it doesn’t print which threading mode it’s using).
os.environ.setdefault("OPENBLAS_VERBOSE", "0")
"""----------------------------------------------------------------------------------------------------------------------"""

import sys
import pandas as pd
import numpy as np
import io
import sys
import shutil
import glob
import time
import logging
import hashlib
import copy
import errno
import socket
import json
import itertools
from mpi4py import MPI

"""----------------------------------------------------------------------------------------------------------------------"""

import tensorflow as tf
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception as _e:
    logging.info(f"TF threading config skipped: {_e}")

"""----------------------------------------------------------------------------------------------------------------------"""

import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from xgboost import XGBRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Concatenate, GRU, Dropout
from tensorflow.keras.optimizers import Adam

"""----------------------------------------------------------------------------------------------------------------------"""
def try_claim(lock_path: str) -> bool:
    """
    Atomically create a lock file. Returns True if we acquired it, False if someone else did.
    Safe across ranks/nodes on a shared filesystem.
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        fd = os.open(lock_path, flags)
        with os.fdopen(fd, "w") as f:
            f.write(f"rank={rank}\npid={os.getpid()}\nhost={socket.gethostname()}\n")
        return True
    except OSError as e:
        if e.errno == errno.EEXIST:
            return False
        raise

def release_claim(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


"""----------------------------------------------------------------------------------------------------------------------"""

# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Inject mpi_rank into every record, no matter which logger produced it

_old_factory = logging.getLogRecordFactory()
def _record_factory(*args, **kwargs):
    record = _old_factory(*args, **kwargs)
    if not hasattr(record, "mpi_rank"):
        record.mpi_rank = rank
    return record
logging.setLogRecordFactory(_record_factory)

rank = MPI.COMM_WORLD.Get_rank()
print(f"[Rank {rank}] Initialized successfully.")



# --- MPI message tags for dispatcher/worker ---
TAG_WORK = 1
TAG_DONE = 2
TAG_STOP = 3

# --- Logging Config ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)

current_file_handler = None

# Named loggers
io_logger = logging.getLogger("cir.io")

# Console handler — only rank 0, and exclude cir.io
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('[Rank %(mpi_rank)d] %(asctime)s - %(levelname)s - %(message)s')
stream_handler.setFormatter(formatter)

# --- Makes sure logs know which MPI rank produced them ---
class MPIRankFilter(logging.Filter):
    def filter(self, record):
        record.mpi_rank = rank
        return True

# --- Hides noisy loggers from the console, but keeps them in file logs ---
class ExcludeNames(logging.Filter):
    def __init__(self, *names):
        self.names = names
    def filter(self, record):
        return not any(record.name.startswith(n) for n in self.names)

logger.addFilter(MPIRankFilter())


# --- Prevents the console from being flooded with output from all ranks ---
if rank == 0:
    stream_handler.setLevel(logging.INFO)
    stream_handler.addFilter(ExcludeNames("cir.io"))  # hide dataset-load logs on console
    logger.addHandler(stream_handler)

# Per-rank file log (keeps everything, including cir.io)
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


"""----------------------------------------------------------------------------------------------------------------------"""
""" Load Datasets """
# Build log file
switch_log_file('logs/CIR-2.log')
logger.info("This is being logged to CIR-2.log")

# Locate datasets (but **do not** load them yet)
data_path = "CSV/imports/split_set/without_multiple_rows"
all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])

logging.info("+++++++++++++++++CIR-2+++++++++++++++++++++++++")
logging.info(f"[Start] Rank {rank} is mapping dataset paths.")

# Map dataset variable names to file paths (lazy loading)
dataset_path_map = {}
for file in all_files:
    var_name = file.replace(".csv", "").replace("-", "_")
    file_path = os.path.join(data_path, file)
    dataset_path_map[var_name] = file_path

logging.info(f"[Rank {rank}] Mapped {len(dataset_path_map)} datasets to file paths.")
logging.info(f"[Complete] Rank {rank} finished mapping datasets.")
logging.info("++++++++++++++++++++++++++++++++++++++++++")

# === MPI Synchronization Barrier ===
comm.Barrier()
logging.info(f"[Rank {rank}] All ranks reached synchronization barrier after mapping.")

def load_dataset_by_name(name: str):
    """Load a dataset by name on demand (float32)."""
    path = dataset_path_map.get(name)
    if path is None:
        logging.warning(f"[Rank {rank}] Dataset '{name}' not found in dataset_path_map.")
        return None
    # goes to files; filtered out of console
    io_logger.info(f"[Rank {rank}] Loading on demand -> {name} from {path}")
    # or: io_logger.debug(...) if you want it at DEBUG level
    return pd.read_csv(path).astype('float32')




"""----------------------------------------------------------------------------------------------------------------------"""
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


"""----------------------------------------------------------------------------------------------------------------------"""

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

"""----------------------------------------------------------------------------------------------------------------------"""

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

"""----------------------------------------------------------------------------------------------------------------------"""


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


"""----------------------------------------------------------------------------------------------------------------------"""

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
        estimator = ExtraTreesRegressor(n_estimators=5, random_state=0, n_jobs=1)
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


"""----------------------------------------------------------------------------------------------------------------------"""
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

"""----------------------------------------------------------------------------------------------------------------------"""

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

    # Work on a copy; ensure unique index to avoid pandas reindex errors
    df_copy = df.copy()
    if not df_copy.index.is_unique:
        logging.info(f"[{dataset_name}] Input index not unique; resetting index.")
        df_copy = df_copy.reset_index(drop=True)

    df_copy["missing_pct"] = df_copy.isnull().mean(axis=1)
    cols = df_copy.columns.drop("missing_pct")

    global_means = df_copy[cols].mean().fillna(0)
    global_min = df_copy[cols].min()
    global_max = df_copy[cols].max()

    imputed_df = pd.DataFrame(index=df_copy.index, columns=cols)
    method_log = pd.Series(index=df_copy.index, dtype="object")

    cum_thresholds = np.cumsum(thresholds)
    if not np.isclose(cum_thresholds[-1], 1.0):
        raise ValueError("Thresholds must sum to 1.0")

    previous_imputed = None
    n_groups = len(cum_thresholds)

    group_names = []
    cumulative_rows = []
    method_names_actual = []
    cumulative_total = 0

    # Shared checkpoint folder for this sequence
    checkpoint_dir = os.path.join(
        "CSV/exports/CIR-16/impute/threshold",
        f"seq_{plot_id:02d}_{'_'.join(method_names)}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    for i, upper_bound in enumerate(cum_thresholds):
        lower_bound = cum_thresholds[i - 1] if i > 0 else 0.0
        idx = df_copy.index[
            (df_copy["missing_pct"] > lower_bound) & (df_copy["missing_pct"] <= upper_bound)
        ]
        group_data = df_copy.loc[idx, cols].copy()

        # Fill all-NaN columns with global means to keep estimators happy
        for col in group_data.columns:
            if group_data[col].isnull().all():
                group_data[col] = global_means[col]

        if group_data.empty:
            continue

        method_name = method_names[i]
        imputer = get_imputer(method_name, method_registry)

        # --- Group checkpoint path ---
        temp_path = os.path.join(checkpoint_dir, f"{dataset_name}_group{i+1:02d}_temp.csv")

        # --- If checkpoint exists, load and continue ---
        if os.path.exists(temp_path):
            logging.info(f"[Rank {rank}] Skipping Group {i+1}/{n_groups} — checkpoint found: {temp_path}")
            group_imputed = pd.read_csv(temp_path, index_col=0)

            # Defensive: drop any accidental duplicate index in the checkpoint
            if not group_imputed.index.is_unique:
                group_imputed = group_imputed[~group_imputed.index.duplicated(keep="last")]

            # Align back to expected order and clip to global bounds
            group_imputed = group_imputed.reindex(idx).clip(lower=global_min, upper=global_max, axis=1)

            # Assign
            imputed_df.loc[idx, cols] = group_imputed.values
            method_log.loc[idx] = method_name
            previous_imputed = (
                pd.concat([previous_imputed, group_imputed])
                if previous_imputed is not None else group_imputed.copy()
            )
            continue

        # --- Atomic group lock (prevents duplicate work across ranks) ---
        lock_path = os.path.join(checkpoint_dir, f"{dataset_name}_group{i+1:02d}.lock")
        if not try_claim(lock_path):
            logging.info(f"[Rank {rank}] Group {i+1}/{n_groups} already claimed by another worker. Skipping.")
            continue

        try:
            combined = group_data if previous_imputed is None else pd.concat([previous_imputed, group_data])

            logging.info(
                f"[{dataset_name}][Group {i+1}] "
                f"({lower_bound:.2f}, {upper_bound:.2f}] -> {method_name} | "
                f"{len(group_data)} group rows | {len(combined)} total used rows"
            )

            # Run the chosen imputer
            try:
                if hasattr(imputer, "fit_transform"):
                    combined_imputed = imputer.fit_transform(combined)
                    combined_imputed = pd.DataFrame(combined_imputed, columns=combined.columns, index=combined.index)
                else:
                    try:
                        combined_imputed = imputer(combined, random_state=random_state)
                    except TypeError:
                        combined_imputed = imputer(combined)
            except Exception as e:
                logging.exception(
                    f"[Rank {rank}] Error during method '{method_name}' in sequence #{plot_id} on group "
                    f"({lower_bound:.2f}, {upper_bound:.2f}]: {e}"
                )
                # Skip this group but release the lock
                continue

            # Extract current group's rows, clip, save checkpoint
            group_imputed = combined_imputed.loc[idx].clip(lower=global_min, upper=global_max, axis=1)

            # Defensive: ensure unique index before assignment
            if not group_imputed.index.is_unique:
                group_imputed = group_imputed[~group_imputed.index.duplicated(keep="last")]

            group_imputed.to_csv(temp_path)  # checkpoint

            # Assign to final containers
            imputed_df.loc[idx, cols] = group_imputed.values
            method_log.loc[idx] = method_name

            previous_imputed = (
                pd.concat([previous_imputed, group_imputed])
                if previous_imputed is not None else group_imputed.copy()
            )

            # Bookkeeping for plot
            group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
            group_names.append(group_label)
            cumulative_total += len(group_data)
            cumulative_rows.append(cumulative_total)
            method_names_actual.append(method_name)

            logging.info(f"[Rank {rank}] Finished group {i+1} / {n_groups} for sequence #{plot_id}")

        finally:
            release_claim(lock_path)

    # Final sanity check
    if imputed_df.isnull().values.any():
        raise ValueError("NaNs remain after hierarchical imputation!")

    # === Plot cumulative bar ===
    output_dir = "figures/CIR-16"
    os.makedirs(output_dir, exist_ok=True)

    unique_methods = list(set(method_names_actual))
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
    plt.legend(handles=legend_handles, title="Imputation Method", loc="lower right")
    plt.tight_layout()

    filename = (
        f"seq_{plot_id:02d}_{dataset_name}_cumulative_imputation_rows.png"
        if dataset_name and plot_id is not None
        else f"{dataset_name}_cumulative_imputation_rows.png"
    )
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

    return (imputed_df, method_log) if return_method_log else imputed_df

"""----------------------------------------------------------------------------------------------------------------------"""

# ---  Look up an imputation method by name in a registry dictionary ---
# ---  and return an instance/function that can perform the imputation. ---
def get_imputer(method_name, registry):
    imputer = registry.get(method_name)
    if imputer is None:
        raise ValueError(f"Method '{method_name}' not found or not implemented.")

    if callable(imputer):
        return imputer  # SOS -> I should not copy or modify lambdas

    if hasattr(imputer, "fit") and hasattr(imputer, "transform"):
        return copy.deepcopy(imputer)

    return imputer


"""----------------------------------------------------------------------------------------------------------------------"""

""" --- Helper + onegroup runner --- """

# --- validates and converts a list of missingness ---
# --- percentage ranges into cumulative cutoffs. ---
def _cum_thresholds(thresholds):
    ct = np.cumsum(thresholds)
    if not np.isclose(ct[-1], 1.0):
        raise ValueError("Thresholds must sum to 1.0")
    return ct

# --- Build the folder path where all checkpoint files ---
# --- for a specific imputation sequence will be stored. ---
def _checkpoint_dir(plot_id, method_names):
    return os.path.join(
        "CSV/exports/CIR-16/impute/threshold",
        f"seq_{plot_id:02d}_{'_'.join(method_names)}"
    )

# --- Build the full file path for the checkpoint CSV ---
# --- that stores the imputed results for a single group ---
# --- in the hierarchical imputation process. ---
def _group_temp_path(checkpoint_dir, dataset_name, gidx):
    return os.path.join(checkpoint_dir, f"{dataset_name}_group{gidx+1:02d}_temp.csv")

# --- Determine which missingness group (if any) still ---
# --- needs to be imputed for a given dataset in a given sequence. ---
def _next_unfinished_group(n_groups, checkpoint_dir, dataset_name):
    for g in range(n_groups):
        if not os.path.exists(_group_temp_path(checkpoint_dir, dataset_name, g)):
            return g
    return None  # all done


# --- Load and combine all completed group checkpoint files
# --- for a given dataset before the group index upto_g.
# --- This is needed because your hierarchical imputation
# --- can use previously imputed groups as extra training data
# --- for the current group.
def _load_previous_imputed_upto(checkpoint_dir, dataset_name, upto_g):
    """Concatenate temp files for groups < upto_g (if any)."""
    dfs = []
    for g in range(upto_g):
        p = _group_temp_path(checkpoint_dir, dataset_name, g)
        if os.path.exists(p):
            gi = pd.read_csv(p, index_col=0)
            if not gi.index.is_unique:
                gi = gi[~gi.index.duplicated(keep="last")]
            dfs.append(gi)
    if not dfs:
        return None
    
    dfs = [df for df in dfs if not df.empty and not df.isna().all().all()]
    if not dfs:
        return None
    prev = pd.concat(dfs, axis=0)

    
    prev = prev[~prev.index.duplicated(keep="last")]
    return prev


# --- Create a short, unique ID for a list of strings
# --- (parts) so it can be used in folder names for caching. ---
def _prefix_hash(parts):
    s = "|".join(parts)
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:16]
    

# --- Build a shared cache directory path for storing/retrieving
# --- imputation results for a specific prefix of methods across 
# --- sequences.Allows multiple sequences that start with the same
# --- methods to reuse early group results instead of recomputing them.
def _shared_prefix_dir(dataset_name, gidx, method_names):
    prefix = method_names[:gidx+1]
    h = _prefix_hash([dataset_name, str(gidx)] + prefix)
    return os.path.join(
        "CSV/exports/CIR-16/impute/shared_prefix_cache",
        f"g{gidx+1:02d}",
        dataset_name,
        h
    )
# --- Create (if needed) and return the full file path for the cached
# --- imputation results inside a shared prefix directory.
def _shared_prefix_temp(shared_dir):
    os.makedirs(shared_dir, exist_ok=True)
    return os.path.join(shared_dir, "imputed.csv")

# --- Copy a file from a source path (src) to a destination path (dst)
# --- only if the destination file does not already exist.
def _copy_if_missing(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(dst):
        try:
            shutil.copy2(src, dst)
        except Exception as _e:
            logging.info(f"Copy fallback failed ({_e}); writing new file next steps will create it.")




# --- Manifest method ---

MANIFEST_NAME = "methods_manifest.json"

def _seq_dir(plot_id, method_names):
    return _checkpoint_dir(plot_id, method_names)

def _manifest_path(seq_dir):
    return os.path.join(seq_dir, MANIFEST_NAME)

def _load_manifest(seq_dir):
    p = _manifest_path(seq_dir)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            logging.warning(f"Manifest at {p} unreadable; ignoring.")
    return None

def _save_manifest(seq_dir, method_names, thresholds):
    try:
        with open(_manifest_path(seq_dir), "w", encoding="utf-8") as f:
            json.dump({"methods": method_names, "thresholds": thresholds}, f)
    except Exception as e:
        logging.warning(f"Could not write manifest: {e}")

def _first_change_index(prev_methods, new_methods):
    if not prev_methods:
        return None
    n = min(len(prev_methods), len(new_methods))
    for i in range(n):
        if prev_methods[i] != new_methods[i]:
            return i
    if len(prev_methods) != len(new_methods):
        return n
    return None  # identical

# ---
def _invalidate_from_g(seq_dir, dataset_name, start_g, n_groups):
    """Delete per-sequence checkpoints for groups >= start_g."""
    if start_g is None:
        return
    for g in range(start_g, n_groups):
        p = _group_temp_path(seq_dir, dataset_name, g)
        if os.path.exists(p):
            try:
                os.remove(p)
                logging.info(f"[Invalidate] Removed {os.path.basename(p)}")
            except Exception as e:
                logging.warning(f"[Invalidate] Could not remove {p}: {e}")






# --- ---

def seed_prefix_checkpoints(dataset_name, thresholds, method_names, plot_id):
    """
    Copy per-group temp files for the longest common prefix of methods from ANY existing sequence directory,
    BUT never seed beyond this sequence's first unfinished group. This prevents jumping ahead when an earlier
    group is still computing on another rank.
    """
    cur_dir = _checkpoint_dir(plot_id, method_names)
    os.makedirs(cur_dir, exist_ok=True)

    seeded_flag = os.path.join(cur_dir, f"{dataset_name}.seeded")
    if os.path.exists(seeded_flag):
        return

    # Determine first unfinished group in THIS sequence (per dataset)
    n_groups = len(thresholds)
    first_unfinished = _next_unfinished_group(n_groups, cur_dir, dataset_name)
    if first_unfinished is None:
        # Nothing to seed; everything is already present
        with open(seeded_flag, "w") as f:
            f.write("ok\n")
        return

    # Respect prior manifest (hard invalidation support): don't seed at/after first change
    prev = _load_manifest(cur_dir)
    change_i = _first_change_index(prev.get("methods") if prev else None, method_names)
    if change_i is not None:
        # We cannot seed at/after change_i
        first_unfinished = min(first_unfinished, change_i)

    if not os.path.isdir(BASE_OUTPUT_ROOT):
        with open(seeded_flag, "w") as f:
            f.write("ok\n")
        return

    # Find best existing sequence dir to reuse from (by longest common prefix of methods)
    best_src_dir = None
    best_prefix = 0
    for dname in os.listdir(BASE_OUTPUT_ROOT):
        src_dir = os.path.join(BASE_OUTPUT_ROOT, dname)
        if not os.path.isdir(src_dir):
            continue
        other_methods = _parse_methods_from_dirname(dname)
        if not other_methods:
            continue
        lcp = _longest_common_prefix_len(method_names, other_methods)
        if lcp > best_prefix:
            best_prefix = lcp
            best_src_dir = src_dir

    if best_src_dir and best_prefix > 0:
        # NEW: never seed beyond the first unfinished group
        max_seed_g = min(best_prefix, first_unfinished)
        for g in range(max_seed_g):
            src = _group_temp_path(best_src_dir, dataset_name, g)
            if os.path.exists(src):
                dst = _group_temp_path(cur_dir, dataset_name, g)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        logging.info(
                            f"[Rank {rank}] Seeded {dataset_name} g{g+1:02d} "
                            f"from '{os.path.basename(best_src_dir)}' into "
                            f"'{os.path.basename(cur_dir)}' (strict order)."
                        )
                    except Exception as e:
                        logging.warning(f"[Rank {rank}] Could not seed {src} -> {dst}: {e}")

    # Mark seeded to avoid repeating scanning this run
    try:
        with open(seeded_flag, "w") as f:
            f.write("ok\n")
    except Exception:
        pass




# --- Call the seeder to find the most common prefix

def worker_loop(comm):
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP or task is None:
            break

        seq_idx, dataset, thresholds, method_names = task

        df = load_dataset_by_name(dataset)
        if df is None or not isinstance(df, pd.DataFrame):
            comm.send((seq_idx, dataset, True, False), dest=0, tag=TAG_DONE)
            continue

        # seed current sequence’s checkpoint dir with the best available prefix
        seed_prefix_checkpoints(
            dataset_name=dataset,
            thresholds=thresholds,
            method_names=method_names,
            plot_id=seq_idx
        )

        done, progressed = run_one_group(
            df=df,
            dataset_name=dataset,
            thresholds=thresholds,
            method_names=method_names,
            method_registry=imputer_registry,
            random_state=0,
            plot_id=seq_idx
        )

        comm.send((seq_idx, dataset, done, progressed), dest=0, tag=TAG_DONE)



def run_one_group(df, dataset_name, thresholds, method_names, method_registry, random_state, plot_id):
    """
    Returns:
        done (bool): True if all groups finished already.
        progressed (bool): True if we actually completed one group now.
    """
    # --- Prep dataframe ---
    df_copy = df.copy()
    if not df_copy.index.is_unique:
        df_copy = df_copy.reset_index(drop=True)

    df_copy["missing_pct"] = df_copy.isnull().mean(axis=1)
    cols = df_copy.columns.drop("missing_pct")

    global_means = df_copy[cols].mean().fillna(0)
    global_min = df_copy[cols].min()
    global_max = df_copy[cols].max()

    # --- Group bookkeeping ---
    cum_th = _cum_thresholds(thresholds)
    n_groups = len(cum_th)

    # Per-sequence checkpoint dir (where dispatcher expects temps)
    seq_dir = _checkpoint_dir(plot_id, method_names)
    os.makedirs(seq_dir, exist_ok=True)

    # === NEW: detect first change vs previous manifest and invalidate >= change_i ===
    prev_manifest = _load_manifest(seq_dir)
    change_i = _first_change_index(prev_manifest.get("methods") if prev_manifest else None, method_names)
    if change_i is not None:
        _invalidate_from_g(seq_dir, dataset_name, change_i, n_groups)

    # Pick next unfinished group by looking at *sequence* directory
    g = _next_unfinished_group(n_groups, seq_dir, dataset_name)
    if g is None:
        return True, False  # all done

    lower = cum_th[g - 1] if g > 0 else 0.0
    upper = cum_th[g]
    idx = df_copy.index[
        (df_copy["missing_pct"] > lower) & (df_copy["missing_pct"] <= upper)
    ]
    group_data = df_copy.loc[idx, cols].copy()

    # Fill all-NaN columns
    for c in group_data.columns:
        if group_data[c].isnull().all():
            group_data[c] = global_means[c]

    # Paths
    seq_temp_path = _group_temp_path(seq_dir, dataset_name, g)

    # --- Shared prefix cache paths (same across sequences that share prefix up to g) ---
    shared_dir = _shared_prefix_dir(dataset_name, g, method_names)
    shared_temp_path = _shared_prefix_temp(shared_dir)  # ensures dir exists
    shared_lock_path = os.path.join(shared_dir, "compute.lock")

    # === NEW: force recompute rule — ignore reuse for groups >= first change ===
    force_recompute = (change_i is not None and g >= change_i)
    if force_recompute:
        # Ensure local checkpoint doesn't mask recompute
        if os.path.exists(seq_temp_path):
            try:
                os.remove(seq_temp_path)
            except Exception:
                pass
    else:
        # Normal reuse allowed only before first change
        if os.path.exists(shared_temp_path):
            _copy_if_missing(shared_temp_path, seq_temp_path)
            logging.info(f"[Rank {rank}] Reused shared prefix cache for {dataset_name} g{g+1:02d}")
            return False, True

        # If this sequence already has the file (rare, but possible), promote it to shared and skip
        if os.path.exists(seq_temp_path):
            _copy_if_missing(seq_temp_path, shared_temp_path)
            logging.info(f"[Rank {rank}] Promoted seq-local checkpoint to shared cache for {dataset_name} g{g+1:02d}")
            return False, True

    # If empty group, just mark as "done" in BOTH places
    if group_data.empty:
        pd.DataFrame(columns=cols, index=idx).to_csv(shared_temp_path)
        _copy_if_missing(shared_temp_path, seq_temp_path)
        return False, True

    # Resolve imputer for this group
    method_name = method_names[g]
    imputer = get_imputer(method_name, method_registry)

    # --- Shared lock so only one rank computes this prefix once ---
    lock_path = shared_lock_path
    if not try_claim(lock_path):
        # Log this *once* per lock using a notice file next to the lock
        notice_path = os.path.join(shared_dir, "compute.lock.notice")
        try:
            fd = os.open(notice_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"first_seen={time.time()}\nrank={rank}\nhost={socket.gethostname()}\n")
            logging.info(f"[Shared] Lock held; skipping {dataset_name} g{g+1:02d} for now")
        except OSError as e:
            # If the notice already exists, stay quiet; otherwise, emit a debug
            if e.errno != errno.EEXIST:
                logging.debug(f"[Rank {rank}] Notice create failed for {dataset_name} g{g+1:02d}: {e}")
        return False, False

    try:
        # Previous imputed rows come from *sequence* dir (what dispatcher expects us to build)
        previous_imputed = _load_previous_imputed_upto(seq_dir, dataset_name, g)
        combined = group_data if previous_imputed is None else pd.concat([previous_imputed, group_data])

        logging.info(
            f"[{dataset_name}][Group {g+1}] ({lower:.2f}, {upper:.2f}] -> {method_name} | "
            f"{len(group_data)} group rows | {len(combined)} total used rows"
        )

        # Run imputer
        try:
            if hasattr(imputer, "fit_transform"):
                combined_imputed = imputer.fit_transform(combined)
                combined_imputed = pd.DataFrame(combined_imputed, columns=combined.columns, index=combined.index)
            else:
                try:
                    combined_imputed = imputer(combined, random_state=random_state)
                except TypeError:
                    combined_imputed = imputer(combined)
        except Exception as e:
            logging.exception(
                f"[Rank {rank}] Error in '{method_name}' on group ({lower:.2f}, {upper:.2f}]: {e}"
            )
            return False, False

        group_imputed = combined_imputed.loc[idx].clip(lower=global_min, upper=global_max, axis=1)
        if not group_imputed.index.is_unique:
            group_imputed = group_imputed[~group_imputed.index.duplicated(keep="last")]

        # Write to shared cache first, then mirror to this sequence
        group_imputed.to_csv(shared_temp_path)
        _copy_if_missing(shared_temp_path, seq_temp_path)

        logging.info(f"[Rank {rank}] Finished group {g+1} for sequence #{plot_id}")

        return False, True

    finally:
        release_claim(lock_path)
        # Allow one-time logging again on the next contention
        try:
            os.remove(os.path.join(shared_dir, "compute.lock.notice"))
        except FileNotFoundError:
            pass



"""----------------------------------------------------------------------------------------------------------------------"""
# --- worker & dispatcher ---

def all_groups_done(dataset_name, thresholds, method_names, plot_id):
    checkpoint_dir = _checkpoint_dir(plot_id, method_names)
    n_groups = len(thresholds)
    for g in range(n_groups):
        if not os.path.exists(_group_temp_path(checkpoint_dir, dataset_name, g)):
            return False
    return True

def stitch_final_outputs(dataset_name, thresholds, method_names, plot_id):
    checkpoint_dir = _checkpoint_dir(plot_id, method_names)
    parts = []
    for g in range(len(thresholds)):
        p = _group_temp_path(checkpoint_dir, dataset_name, g)
        if os.path.exists(p):
            gi = pd.read_csv(p, index_col=0)
            parts.append(gi)
    if not parts:
        return None, None
    full = pd.concat(parts, axis=0)
    full = full[~full.index.duplicated(keep="last")].sort_index()

    method_map = {}
    for g in range(len(thresholds)):
        p = _group_temp_path(checkpoint_dir, dataset_name, g)
        if os.path.exists(p):
            gi = pd.read_csv(p, index_col=0)
            method = method_names[g]
            for idx in gi.index:
                method_map[idx] = method
    method_log = pd.Series(method_map).sort_index()

    return full, method_log


def dispatcher(comm, work_items):
    queue = [(seq_idx, dataset, thresholds, method_names)
             for (seq_idx, _folder, dataset, thresholds, method_names) in work_items]

    idle_workers = list(range(1, size))
    in_flight = {}

    # Prime the workers
    while queue and idle_workers:
        w = idle_workers.pop(0)
        task = queue.pop(0)
        in_flight[w] = task
        comm.send(task, dest=w, tag=TAG_WORK)

    while in_flight:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE, status=status)
        w = status.Get_source()
        seq_idx, dataset, done, progressed = msg
        task = in_flight.pop(w)

        # If finished, stitch + write outputs once
        if all_groups_done(dataset, task[2], task[3], seq_idx):
            imputed_df, method_log = stitch_final_outputs(dataset, task[2], task[3], seq_idx)
            base_output_root = "CSV/exports/CIR-16/impute/threshold"
            method_sequence_str = "_".join(task[3])
            subfolder_path = os.path.join(base_output_root, f"seq_{seq_idx:02d}_{method_sequence_str}")
            os.makedirs(subfolder_path, exist_ok=True)
            if imputed_df is not None:
                imputed_path = os.path.join(subfolder_path, f"{dataset}_rank0.csv")
                imputed_df.to_csv(imputed_path, index=False)
            if method_log is not None:
                method_log_path = os.path.join(subfolder_path, f"{dataset}_method_log_rank0.csv")
                method_log.to_csv(method_log_path, index=False)
            # NEW: persist manifest for this sequence dir
            seq_dir = _checkpoint_dir(seq_idx, task[3])
            _save_manifest(seq_dir, task[3], task[2])
            logging.info(f"[Rank 0] Finalized {dataset} for sequence #{seq_idx}")

        else:
            # Not complete — requeue to keep progressing next group
            queue.append(task)

        # Reassign or stop worker
        if queue:
            next_task = queue.pop(0)
            in_flight[w] = next_task
            comm.send(next_task, dest=w, tag=TAG_WORK)
        else:
            comm.send(None, dest=w, tag=TAG_STOP)

    # Ensure everyone stops
    for w in range(1, size):
        comm.send(None, dest=w, tag=TAG_STOP)


"""----------------------------------------------------------------------------------------------------------------------"""
"""
Copy per-group temp files for the longest common prefix of methods from ANY existing sequence directory, so we don't recompute groups that won't change.
"""

BASE_OUTPUT_ROOT = "CSV/exports/CIR-16/impute/threshold"

#def _seq_dir_name(plot_id, method_names):
#    return f"seq_{plot_id:02d}_{'_'.join(method_names)}"

def _longest_common_prefix_len(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n

def _parse_methods_from_dirname(dname):
    # expects: seq_XX_<method>_<method>_...
    parts = dname.split("_", 2)
    if len(parts) < 3 or not parts[0].startswith("seq"):
        return None
    methods_str = parts[2]
    return methods_str.split("_")

def seed_prefix_checkpoints(dataset_name, thresholds, method_names, plot_id):
    cur_dir = _checkpoint_dir(plot_id, method_names)
    os.makedirs(cur_dir, exist_ok=True)

    seeded_flag = os.path.join(cur_dir, f"{dataset_name}.seeded")
    if os.path.exists(seeded_flag):
        return

    # NEW: ensure the root exists (or return if it doesn't)
    if not os.path.isdir(BASE_OUTPUT_ROOT):
        # nothing to reuse yet
        with open(seeded_flag, "w") as f:
            f.write("ok\n")
        return


    # Scan all existing sequence dirs and try to reuse the best prefix we can find
    best_src_dir = None
    best_prefix = 0
    

    for dname in os.listdir(BASE_OUTPUT_ROOT):
        src_dir = os.path.join(BASE_OUTPUT_ROOT, dname)
        if not os.path.isdir(src_dir):
            continue
        other_methods = _parse_methods_from_dirname(dname)
        if not other_methods:
            continue
        lcp = _longest_common_prefix_len(method_names, other_methods)
        if lcp > best_prefix:
            best_prefix = lcp
            best_src_dir = src_dir

    if best_src_dir and best_prefix > 0:
        # Copy temp files for all groups before the change point
        for g in range(best_prefix):
            src = _group_temp_path(best_src_dir, dataset_name, g)
            if os.path.exists(src):
                dst = _group_temp_path(cur_dir, dataset_name, g)
                if not os.path.exists(dst):
                    try:
                        shutil.copy2(src, dst)
                        logging.info(f"[Rank {rank}] Seeded group {g+1:02d} for {dataset_name} "
                                     f"from '{os.path.basename(best_src_dir)}' into "
                                     f"'{os.path.basename(cur_dir)}'.")
                    except Exception as e:
                        logging.warning(f"[Rank {rank}] Could not seed {src} -> {dst}: {e}")

    # mark seeded to skip next time
    try:
        with open(seeded_flag, "w") as f:
            f.write("ok\n")
    except Exception:
        pass


"""----------------------------------------------------------------------------------------------------------------------"""


methods_all = ["mean", "median", "knn", "iterative_simple", "iterative_function", "xgboost", "gan", "lstm", "rnn"]

# Define all datasets
datasets = [
    "o1_X_external" #"o2_X_test", "o3_X_test", "o4_X_test",
    #"o1_X_validate", "o2_X_validate", "o3_X_validate", "o4_X_validate"
    
    #"o1_X_train", "o1_X_validate", "o1_X_test", "o1_X_external",
    #"o2_X_train", "o2_X_validate", "o2_X_test", "o2_X_external",
    #"o3_X_train", "o3_X_validate", "o3_X_test", "o3_X_external",
    #"o4_X_train", "o4_X_validate", "o4_X_test", "o4_X_external"
]


"""
------------------Build Sequenses----------------------------------------
"""
# Define the fixed sequence builder

def build_sequences():
    
    #------------------------0%------------------------------
    # Jerez et al., 2010; Batista & Monard, 2003; Che et al., 2018.
    
    group_A = ["knn"] #["knn", "median", "mean"] #5 1
    group_B = ["knn"] #["knn", "median", "mean"] #5 2
    group_C = ["knn"] #["knn", "median", "mean"] #5 3
    group_D = ["knn"] #["knn", "median", "mean"] #5 4
    group_E = ["knn"] #["knn", "median", "mean"] #5 5
    group_F = ["knn"] #["knn", "median", "mean"] #5 6
    
    #------------------------30%------------------------------
    # Bertsimas et al., 2018 (data-driven MICE with tree-based models); Lin et al., 2020 (XGBoost outperforms for ICU datasets).
    
    group_G = ["iterative_function"] #["iterative_function", "xgboost"] #5 7
    group_H = ["xgboost"] #["xgboost", "iterative_function"] #5 8
    group_I = ["xgboost"] #["xgboost", "iterative_function"] #5 9
    group_J = ["xgboost"] #["xgboost", "iterative_function"] #5 10
    group_K = ["xgboost"] #["xgboost", "iterative_function"] #5 11
    group_L = ["xgboost"] #["xgboost", "iterative_function"] #5 12
    
    #------------------------60%------------------------------
    # Yoon et al., 2019 (BRITS), Cao et al., 2018 (GRU-D).
    
    group_M = ["lstm", "rnn" ] #5 13
    group_N = ["lstm", "rnn"] #5 14
    group_O = ["lstm", "rnn"] #5 15
    group_P = ["lstm", "rnn"] #5 16
    
    #------------------------80%------------------------------
    # Yoon et al., 2018 (GAIN); Luo et al., 2020.
    
    group_Q = ["gan", "rnn"] #10 17
    group_R = ["gan", "rnn"] #10 18
    

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
#def split_evenly(lst, n):
#    """Split list lst into n nearly equal parts."""
#    k, m = divmod(len(lst), n)
#    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# --- Build sequences and combinations ---
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()



work_items, index_records = build_sequences()
print(f"Total sequences generated: {len(work_items)}")


work_items = []
index_records = []
timing_records = {}



# Use the new rotational sequence builder
work_items, index_records = build_sequences()
print(f"Total sequences generated: {len(work_items)}")




"""----------------------------------------------------------------------------------------------------------------------"""


# === Dynamic scheduling ===
if rank == 0:
    dispatcher(comm, work_items)

    # Save the index file at the end (like before)
    base_output_root = "CSV/exports/CIR-16/impute/threshold"
    os.makedirs(base_output_root, exist_ok=True)
    index_df = pd.DataFrame(index_records)
    index_df_path = os.path.join(base_output_root, "imputation_sequence_index.csv")
    index_df.to_csv(index_df_path, index=False)
    logging.info(f"[Rank 0] Saved index file: {index_df_path}")

    # Optional: sequence_description.txt (timings per dataset are harder now; keep if you want)
    text_log_path = os.path.join(base_output_root, "sequence_description.txt")
    with open(text_log_path, "w", encoding="utf-8") as f:
        for record in index_records:
            f.write(f"{record['sequence_id']}: {record['methods']}\n")
    logging.info(f"[Rank 0] Saved sequence description log to {text_log_path}")

else:
    worker_loop(comm)

"""----------------------------------------------------------------------------------------------------------------------"""