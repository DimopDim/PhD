"""CIR-03: Hierarchical Imputation Framework"""
import os
import sys
import pandas as pd
import numpy as np
import io
import sys
import time
import logging
import copy
from mpi4py import MPI
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from xgboost import XGBRegressor

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, Concatenate, GRU, Dropout
from tensorflow.keras.optimizers import Adam

"""----------------------------------------------------------------------------------------------------------------------"""
""" Logging Config"""
# --- MPI Setup ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rank = MPI.COMM_WORLD.Get_rank()
print(f"[Rank {rank}] Initialized successfully.")


# === Initial logger setup ===
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variable to hold the active file handler
current_file_handler = None

# Stream handler for console output
stream_handler = logging.StreamHandler(sys.stdout)  # <<<<<< Force stdout
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


"""----------------------------------------------------------------------------------------------------------------------"""
""" Load Datasets """
# Build log file
switch_log_file('logs/CIR-2.log')
logger.info("This is being logged to CIR-2.log")

# Load datasets
data_path = "CSV/imports/split_set/without_multiple_rows"
all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])

logging.info("+++++++++++++++++CIR-2+++++++++++++++++++++++++")
logging.info(f"[Start] Rank {rank} is loading dataframes.")

# Split files across ranks
files_per_rank = [all_files[i] for i in range(len(all_files)) if i % size == rank]

logging.info(f"[Rank {rank}] Assigned files: {files_per_rank}")

dataframes = {}

for file in all_files:
    var_name = file.replace(".csv", "").replace("-", "_")
    file_path = os.path.join(data_path, file)

    logging.info(f"[Rank {rank}] Loading... -> {file}")
    df = pd.read_csv(file_path).astype('float32')

    logging.info(f"[Rank {rank}] Loaded {file} with shape {df.shape}")

    dataframes[var_name] = df
    globals()[var_name] = df


logging.info(f"[Complete] Rank {rank} has loaded all data.")
logging.info("++++++++++++++++++++++++++++++++++++++++++")

# === MPI Synchronization Barrier ===
comm.Barrier()
logging.info(f"[Rank {rank}] All ranks reached synchronization barrier after loading.")

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

def lstm_imputer(df, random_state=0, epochs=100, batch_size=64):
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
        if epoch % 10 == 0 or epoch == epochs - 1:
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
def gan_imputer(df, random_state=0, epochs=100, batch_size=128):
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
def rnn_imputer(df, random_state=0, epochs=100, batch_size=64, mask_ratio=0.2, patience=10):
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
        encoded = Dropout(0.2)(encoded)  # 🧬 Dropout added
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

        if epoch % 10 == 0 or epoch == epochs - 1:
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
        estimator = ExtraTreesRegressor(n_estimators=5, random_state=0, n_jobs=-1)
    elif method == "HistGradientBoosting":
        estimator = HistGradientBoostingRegressor(random_state=0)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'ExtraTrees' or 'HistGradientBoosting'.")

    imputer = IterativeImputer(
        estimator=estimator,
        max_iter=n_iter,
        random_state=0,
        verbose=2,
        sample_posterior=False
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
        method="ExtraTrees",
        output_path="imputed_outputs/tmp.csv",
        n_iter=5
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

    df_copy = df.copy()
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

    for i, upper_bound in enumerate(cum_thresholds):
        lower_bound = cum_thresholds[i - 1] if i > 0 else 0.0
        idx = df_copy.index[
            (df_copy["missing_pct"] > lower_bound) & (df_copy["missing_pct"] <= upper_bound)
        ]
        group_data = df_copy.loc[idx, cols].copy()

        for col in group_data.columns:
            if group_data[col].isnull().all():
                group_data[col] = global_means[col]

        if group_data.empty:
            continue

        method_name = method_names[i]
        logging.info(f"[{dataset_name}][Group {i+1}] ({lower_bound:.2f}, {upper_bound:.2f}] -> {method_name} | {len(group_data)} rows")

        imputer = get_imputer(method_name, method_registry)
        if previous_imputed is None:
            combined = group_data
        else:
            combined = pd.concat([previous_imputed, group_data])

        try:
            if hasattr(imputer, "fit_transform"):
                combined_imputed = imputer.fit_transform(combined)
                combined_imputed = pd.DataFrame(combined_imputed, columns=combined.columns, index=combined.index)
            else:
                # Try to call with random_state first
                try:
                    combined_imputed = imputer(combined, random_state=random_state)
                except TypeError:
                    combined_imputed = imputer(combined)
        except Exception as e:
            logging.exception(f"[Rank {rank}] ❌ Error during method '{method_name}' in sequence #{plot_id} on group ({lower_bound:.2f}, {upper_bound:.2f}]: {e}")
            continue  # Skip to next group

        group_imputed = combined_imputed.loc[idx].clip(lower=global_min, upper=global_max, axis=1)

        imputed_df.loc[idx] = group_imputed
        method_log.loc[idx] = method_name

        previous_imputed = pd.concat([previous_imputed, group_imputed]) if previous_imputed is not None else group_imputed.copy()

        group_label = f"{int(lower_bound * 100)}%–{int(upper_bound * 100)}%"
        group_names.append(group_label)
        cumulative_total += len(group_data)
        cumulative_rows.append(cumulative_total)
        method_names_actual.append(method_name)

        logging.info(f"[Rank {rank}] Finished group {i+1} / {n_groups} for sequence #{plot_id}")

    if imputed_df.isnull().values.any():
        raise ValueError("NaNs remain after hierarchical imputation!")

    # === Plot ===
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

    filename = f"seq_{plot_id:02d}_{dataset_name}_cumulative_imputation_rows.png" if dataset_name and plot_id is not None else f"{dataset_name}_cumulative_imputation_rows.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

    return (imputed_df, method_log) if return_method_log else imputed_df





"""----------------------------------------------------------------------------------------------------------------------"""

def get_imputer(method_name, registry):
    imputer = registry.get(method_name)
    if imputer is None:
        raise ValueError(f"Method '{method_name}' not found or not implemented.")

    if callable(imputer):
        return imputer  # Don't copy or modify lambdas — just return

    if hasattr(imputer, "fit") and hasattr(imputer, "transform"):
        return copy.deepcopy(imputer)

    return imputer


"""----------------------------------------------------------------------------------------------------------------------"""

"""----------------------------------------------------------------------------------------------------------------------"""


methods_all = ["mean", "median", "knn", "iterative_function", "xgboost", "gan", "lstm", "rnn"]

# Define all datasets
datasets = [
    "o1_X_validate",
    "o2_X_validate",
    "o3_X_validate",
    "o4_X_validate"
    # "o1_X_train", "o1_X_validate", "o1_X_test", "o1_X_external",
    # "o2_X_train", "o2_X_validate", "o2_X_test", "o2_X_external",
    # "o3_X_train", "o3_X_validate", "o3_X_test", "o3_X_external",
    # "o4_X_train", "o4_X_validate", "o4_X_test", "o4_X_external"
]

# Define the fixed sequence builder
def build_sequences():
    sequences = []

    # Stage A: Run each method alone
    for method in methods_all:
        sequences.append([method] * 10)

    # Stage B: Fix KNN for 1 part, cycle rest 9
    fixed_1 = ["knn"] * 1
    for method in methods_all:
        if method != "knn":
            sequences.append(fixed_1 + [method] * 9)

    # Stage C onward: Incrementally grow the fixed prefix
    base = ["knn"] * 1 + ["iterative_function"] * 2
    sequences.append(base + ["mean"] * 7)
    sequences.append(base + ["median"] * 7)
    sequences.append(base + ["xgboost"] * 7)
    sequences.append(base + ["gan"] * 7)
    sequences.append(base + ["lstm"] * 7)
    sequences.append(base + ["rnn"] * 7)

    # Add LSTM to base
    base += ["lstm"] * 2
    sequences.append(base + ["mean"] * 5)
    sequences.append(base + ["gan"] * 5)
    sequences.append(base + ["rnn"] * 5)

    # Add RNN to base
    base += ["rnn"] * 2
    sequences.append(base + ["gan"] * 3)

    return sequences

# Custom function to split work across ranks without using numpy
def split_evenly(lst, n):
    """Split list `lst` into `n` nearly equal parts."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

# --- Build sequences and combinations ---
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sequences = build_sequences()
print(f"Total sequences generated: {len(sequences)}")
#sequences = [sequences[-1]]
work_items = []
index_records = []
timing_records = {}

# === Assign unique sequence IDs and build jobs ===
for idx, method_names in enumerate(sequences):
    thresholds = [0.10] * len(method_names)
    folder_name = f"seq_{idx:02d}"
    for dataset in datasets:
        work_items.append((idx, folder_name, dataset, thresholds, method_names))
    index_records.append({
        "sequence_id": folder_name,
        "methods": " | ".join(method_names),
        "num_methods": len(method_names)
    })

# === Distribute work across ranks ===
items_per_rank = split_evenly(work_items, size)
local_work = items_per_rank[rank]
print(f"[Rank {rank}] Received {len(local_work)} work items out of {len(work_items)} total.")

# === Ensure base output path exists ===
base_output_root = "CSV/exports/CIR-16/impute"
os.makedirs(base_output_root, exist_ok=True)


# === Process local jobs ===
for idx, _, dataset_name, thresholds, method_names in local_work:
    # === Construct folder name with method names ===
    method_sequence_str = "_".join(method_names)
    folder_name = f"seq_{idx:02d}_{method_sequence_str}"
    subfolder_path = os.path.join(base_output_root, folder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    logging.info(f"[Rank {rank}] Processing sequence #{idx:02d} on {dataset_name} using: {' | '.join(method_names)}")


    df = globals().get(dataset_name)
    if df is None or not isinstance(df, pd.DataFrame):
        logging.info(f"[Rank {rank}] Skipping {dataset_name} (not found or not a DataFrame)")
        continue

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
        timing_records[(idx, dataset_name)] = duration

        imputed_path = os.path.join(subfolder_path, f"{dataset_name}_rank{rank}.csv")
        method_log_path = os.path.join(subfolder_path, f"{dataset_name}_method_log_rank{rank}.csv")

        imputed_df.to_csv(imputed_path, index=False)
        method_log.to_csv(method_log_path, index=False)

        logging.info(f"[Rank {rank}] Saved: {imputed_path}")

    except Exception as e:
        logging.exception(f"[Rank {rank}] Exception during step #{idx} on {dataset_name}")


# === Rank 0 saves the index file and description log ===
comm.Barrier()
if rank == 0:
    index_df = pd.DataFrame(index_records)
    index_df_path = os.path.join(base_output_root, "imputation_sequence_index.csv")
    index_df.to_csv(index_df_path, index=False)
    logging.info(f"[Rank 0] Saved index file: {index_df_path}")

    # Save human-readable sequence description log
    text_log_path = os.path.join(base_output_root, "sequence_description.txt")
    with open(text_log_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(index_records):
            f.write(f"{record['sequence_id']}: {record['methods']}\n")
            # Append timing info per dataset if available
            for dataset in datasets:
                duration = timing_records.get((idx, dataset))
                if duration is not None:
                    f.write(f"    {dataset}: {duration:.2f} sec\n")
    logging.info(f"[Rank 0] Saved sequence description log to {text_log_path}")




"""----------------------------------------------------------------------------------------------------------------------"""

"""----------------------------------------------------------------------------------------------------------------------"""
