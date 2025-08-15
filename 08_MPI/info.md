Pseudocode for the MPI‑Parallel Hierarchical Imputation Framework
1) One‑time process setup

Set math/BLAS env vars to 1 thread so each MPI rank stays single‑threaded.

Import all libraries (MPI, pandas/numpy, sklearn/xgboost/tensorflow, plotting, logging, etc.).

Try to set TensorFlow intra/inter‑op threads to 1 (ignore if it fails).

2) MPI + logging bootstrap

Get comm = MPI.COMM_WORLD, rank, size.

Make every log record carry mpi_rank = rank.

Print "[Rank r] Initialized successfully." for visibility.

Create a root logger.

If rank == 0:

Add a console handler with format [Rank %(mpi_rank)d] ... and hide very chatty IO logs.

Add a per‑rank file handler: logs/rank_<rank>.log.

Define switch_log_file(path) that moves logging to a shared log file (append mode).

3) Point the run at the main log + discover datasets (lazy)

switch_log_file('logs/CIR-2.log') and log a banner.

Set data_path = "CSV/imports/split_set/without_multiple_rows".

List all *.csv in that folder.

For each file:

Map var_name = filename without .csv (hyphens -> underscores) to its absolute path in dataset_path_map.

Barrier() so all ranks finish discovery together.

Define load_dataset_by_name(name):

Look up path from dataset_path_map.

Log via the quieter IO logger.

read_csv and cast to float32.

Return the DataFrame.

4) Define individual imputers (functions or estimators)

xgboost_imputer(df): for each column with NaNs, fit XGB on other columns and predict missing values; n_jobs=1.

lstm_imputer(df): LSTM autoencoder (cache model once), train to reconstruct, then fill only original NaNs.

gan_imputer(df): GAIN‑style small MLP generator/discriminator; train, then impute; inverse‑scale.

rnn_imputer(df): GRU autoencoder with dynamic masking + early stopping; then fill only original NaNs.

impute_with_iterative(df, method, n_iter, output_path, ...): MICE with chosen estimator (ExtraTrees, etc.), logs verbose progress, writes CSV, returns DF.

Build an imputer_registry:

"mean", "median" → SimpleImputer.

"knn" → KNNImputer.

"iterative_simple" → IterativeImputer(ExtraTrees, ...).

"iterative_function" → lambda that calls impute_with_iterative(..., output_path="imputed_outputs/tmp.csv").

"xgboost", "gan", "lstm", "rnn" → the custom functions above.

Define get_imputer(name, registry):

If callable: return it.

If it looks like an sklearn transformer: return a deepcopy.

Else: error.

5) (Defined but not used by the scheduler) Full hierarchical imputer

hierarchical_impute_dynamic(df, thresholds, method_names, ...):

Validate len(thresholds) == len(method_names) and sum(thresholds) ≈ 1.0.

Compute missing_pct per row; store global_means/min/max.

For each group i by cumulative thresholds:

Pick rows in that missingness range.

If group file already exists (checkpoint_dir/dataset_groupXX_temp.csv), load it, clip to global bounds, and continue.

Else, acquire a per‑group lock; if cannot acquire, skip for now (another rank will do it).

Combine previously imputed rows + current group rows to fit the current imputer.

Run the imputer (either fit_transform or call function).

Extract just the group rows, clip to global min/max, save checkpoint.

Append to previous_imputed; update plotting bookkeeping.

After all groups: if any NaNs remain, raise error.

Plot cumulative rows per group (colored by method) and save figure.

Return imputed DF (and method log if requested).

(Note: your live scheduler below uses a lighter “one‑group‑at‑a‑time” variant instead.)

6) Checkpointing, caching, and seeding helpers

Helper functions to:

Build cumulative thresholds, sequence directories, group temp paths.

Find the next unfinished group (first missing group temp).

Load previous imputed groups up to a given group.

Build a shared prefix cache directory based on the method prefix hash (so different sequences that share early methods can reuse results).

Copy a file only if the destination is missing.

Maintain a manifest (methods_manifest.json) storing methods/thresholds for change detection.

Invalidate per‑sequence checkpoints from the first changed group onward.

⚠️ There are two seed_prefix_checkpoints(...) definitions in the file:

An earlier one (respects manifest and “first unfinished group”) and a later simpler one.

In Python, the later definition overrides the earlier. (See “Possible mistakes” below.)

7) Worker execution (ranks > 0)

worker_loop(comm):

REPEAT:

recv a task (seq_idx, dataset_name, thresholds, method_names) with tag:

If TAG_STOP or task is None: break.

Load dataset by name.

If load fails: send (seq_idx, dataset, done=True, progressed=False) with TAG_DONE.

Call seed_prefix_checkpoints(...) for this dataset/sequence to reuse earlier work if possible.

Call run_one_group(...):

Returns (done, progressed):

done=True if all groups are complete already.

progressed=True if this call completed one group (or copied from shared cache).

send (seq_idx, dataset, done, progressed) back with TAG_DONE.

8) Do exactly one group of work (idempotent)

run_one_group(df, dataset_name, thresholds, method_names, ...):

Ensure unique index; compute missing_pct, globals min/max.

Build seq_dir for this (seq_idx, methods).

Load previous manifest; if methods changed, invalidate checkpoints from the first change index onward.

Find next unfinished group g by checking for missing group checkpoint in seq_dir.

If none: return (True, False).

Extract the rows for group g; fill columns that are all‑NaN with global means.

Compute shared prefix cache folder for (dataset_name, g, method_names) and the shared temp path + shared lock path.

If we’re before the first change group:

If shared temp exists: copy to sequence path and return (False, True).

Else if sequence path exists: promote to shared and return (False, True).

If group is empty: write empty frame to shared + copy to sequence; return (False, True).

Resolve the imputer for this group.

Try to claim the shared lock for this (dataset, group, method prefix):

If cannot claim, log once via a “notice” file and return (False, False) (another rank is working).

If lock acquired:

Load all previous imputed groups from the sequence dir and concat with current group (“combined”).

Run the imputer on “combined” (either fit_transform or function call).

Extract just the target group rows, clip, ensure unique index.

Write to shared cache first, then copy to the sequence path.

Release the lock (and remove the one‑time “notice”).

Return (False, True).

9) Dispatcher (rank 0)

Build queue = list of (seq_idx, dataset, thresholds, method_names) from work_items.

Initialize idle_workers = [1..size-1] and in_flight = {}.

Prime: send as many tasks as possible to idle workers; record them in in_flight.

WHILE in_flight is not empty:

recv (seq_idx, dataset, done, progressed) with TAG_DONE.

Get the task for that worker from in_flight.

If all_groups_done(dataset, thresholds, methods, seq_idx):

stitch_final_outputs(dataset, thresholds, methods, seq_idx) by concatenating all group temps.

Save the stitched final imputed CSV and the method log into CSV/exports/CIR-16/impute/threshold/seq_<id>_<methods>/.

Save the manifest for that sequence dir.

ELSE:

Requeue the same task to continue with the next group on a later cycle.

If there’s still work in the queue:

Send the next task to this freed worker and put it in in_flight.

ELSE:

Send TAG_STOP to this worker.

After loop: send TAG_STOP to all workers to be safe.

10) Building sequences (parameter sweep)

Define method groups by missingness bands:

A–F (six 5% bands): ["knn"].

G–L (six 5% bands): mix of iterative_function and xgboost (your current setup: G uses iterative_function, H–L use xgboost).

M–P (four 5% bands): ["lstm", "rnn"] alternatives.

Q–R (two 10% bands): ["gan", "rnn"] alternatives.

Create thresholds = [0.05]*16 + [0.10]*2 (sums to 1.0 → 18 groups).

For every Cartesian product over A..R choices (as configured):

Build method_names = [a,b,...,r] (18 entries).

Build a sequence folder label seq_<id>_<a>_<b>_..._<r>.

For each dataset in datasets:

Append (seq_idx, folder_name, dataset, thresholds, method_names) to work_items.

Record an index row (for audit/logs).

Return work_items, index_records.

11) Main program flow

Re‑obtain comm, rank, size (safe).

Build sequences → (work_items, index_records).

Print “Total sequences generated”.

(The code resets work_items = [] ; index_records = [] ; timing_records = {} and rebuilds them again. It then prints the total count again.)

If rank == 0:

Run dispatcher(comm, work_items).

After finishing, save:

imputation_sequence_index.csv listing all sequences.

sequence_description.txt listing sequence → methods.

Else (worker):

Run worker_loop(comm).

Likely mistakes / fragile spots to double‑check

Duplicate function name: seed_prefix_checkpoints(...) is defined twice.

The later (simpler) definition overrides the earlier one (which had manifest and “first unfinished” safeguards).

If you intended the advanced behavior, remove/rename the second definition or merge logic.

BASE_OUTPUT_ROOT used before definition in the earlier seed_prefix_checkpoints(...).

Because it’s overridden later, you don’t hit a NameError—but it’s a code smell. Keep one definition and place BASE_OUTPUT_ROOT above it.

iterative_function writes to a fixed path "imputed_outputs/tmp.csv".

Multiple ranks/sequences can collide. Use a unique path per rank/sequence/group (e.g., include rank, seq_id, dataset, group in the filename) or write to a temp file that isn’t shared.

Two prints of “Total sequences generated” and a temporary wipe of work_items/index_records.

This is harmless but confusing. Remove the first build or the wipe.

hierarchical_impute_dynamic(...) is not called by the live scheduler.

That’s fine if intentional, but dead code invites drift. Either wire it in (for single‑process runs) or move it to a separate module.

Global Keras models (_lstm_model, _rnn_model) are cached per process.

That’s okay, but if you change input dims between datasets, you may need to rebuild the models or key them by input_dim.

Plotting in headless workers:

You save figures to disk (good). Just ensure the backend is non‑interactive on headless nodes.

Index safety:

You handle non‑unique indexes in several places. Good. Keep it consistent anywhere you read checkpoints (index_col=0) to avoid misalignment.

Method/threshold alignment:

Current thresholds length is 18; your sequence builder produces 18 methods—good. Any tweak must keep them equal or you’ll raise an error.

Locks and shared filesystems:

try_claim uses POSIX O_EXCL create—good for NFS too. Make sure all ranks see the same shared filesystem layout.
