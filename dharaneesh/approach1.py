"""Training and feature preparation for the selected Approach 1 FNN."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
import inspect
import shutil
import zipfile
import json
import os
from pathlib import Path
import time

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .submission.model import (
    SUBJECT_FIELDS, ENCODING_VERSION, bert_window, clean, encode_text, encode_token_windows,
    feature_vector, hash_vector, metadata_vector, network_logits, numeric_vector, prompt_key,
    select_windows, sigmoid, _metadata_vector,
)

HERE = Path(__file__).resolve().parent
SUBMISSION_DIR = HERE / "submission"
RUNTIME_SUBJECT_FIELDS = (*SUBJECT_FIELDS, "release_date", "access_date", "subject_features_extra")


@dataclass(frozen=True)
class Method:
    name: str = "sbert_window_fnn_brier"
    features: str = "sbert"
    estimator: str = "fnn"
    loss: str = "brier"
    hidden: tuple = (32, 16)
    dropout: float = 0.35
    l2: float = 1e-4


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_examples(data_dir):
    """Retain whole native-binary benchmarks; never invent a partial-credit threshold."""
    frames, inventory = [], []
    for folder in sorted(Path(data_dir).iterdir()):
        if not (folder / "response.parquet").is_file():
            continue
        metadata = pd.read_parquet(folder / "benchmarks.parquet")
        responses = pd.read_parquet(folder / "response.parquet")
        eligible_ids = set(metadata.loc[
            metadata["response_type"].eq("binary") & metadata["granularity"].eq("item"),
            "benchmark_id",
        ])
        inventory.append({
            "benchmark": folder.name, "rows": len(responses),
            "used": bool(eligible_ids), "response_type": metadata["response_type"].tolist(),
            "files": {name: file_sha256(folder / name) for name in (
                "items.parquet", "subjects.parquet", "response.parquet", "benchmarks.parquet",
            )},
        })
        if not eligible_ids:
            continue
        responses = responses.loc[responses["benchmark_id"].isin(eligible_ids)].copy()
        responses = responses.dropna(subset=["response"])
        if not responses["response"].isin([0, 1]).all():
            raise ValueError(f"{folder.name} is declared binary but contains other labels.")
        items = pd.read_parquet(folder / "items.parquet")
        subjects = pd.read_parquet(folder / "subjects.parquet")
        subject_columns = ["subject_id", *[field for field in RUNTIME_SUBJECT_FIELDS if field in subjects]]
        joined = responses[["response_id", "subject_id", "item_id", "benchmark_id", "response"]].merge(
            items[["item_id", "benchmark_id", "content"]], on=["item_id", "benchmark_id"],
            how="left", validate="many_to_one", indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{folder.name}: a response has no matching item.")
        joined = joined.drop(columns="_merge").merge(
            subjects[subject_columns], on="subject_id", how="left", validate="many_to_one", indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise ValueError(f"{folder.name}: a response has no matching subject.")
        joined = joined.drop(columns="_merge").rename(columns={"content": "item_content"})
        joined["source"] = folder.name
        frames.append(joined)
    if not frames:
        raise ValueError("No item-level binary benchmarks were found.")
    examples = pd.concat(frames, ignore_index=True)
    for column in ["item_content", *RUNTIME_SUBJECT_FIELDS]:
        if column not in examples:
            examples[column] = ""
        examples[column] = examples[column].fillna("").astype(str).map(clean)
    examples["prompt_group"] = examples["item_content"].map(prompt_key)
    examples["pair_key"] = examples["source"] + "::" + examples["subject_id"].astype(str)
    examples["response"] = examples["response"].astype(np.float32)
    if examples.duplicated(["source", "response_id"]).any():
        raise ValueError("Duplicate response IDs within a source benchmark.")
    return examples, inventory

def grouped_split(frame, fraction=0.2, seed=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_index, validation_index = next(splitter.split(frame, groups=frame["prompt_group"]))
    training, validation = frame.iloc[train_index].copy(), frame.iloc[validation_index].copy()
    if set(training["prompt_group"]) & set(validation["prompt_group"]):
        raise AssertionError("Duplicate prompts crossed the training and validation split.")
    return training, validation

def pair_weights(frame):
    counts = frame.groupby("pair_key")["response"].transform("size").to_numpy()
    return (len(frame) / (frame["pair_key"].nunique() * counts)).astype(np.float32)

def scores(frame, probabilities):
    probabilities = np.asarray(probabilities).reshape(-1)
    if len(probabilities) != len(frame) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction array.")
    labels = frame["response"].to_numpy()
    squared_errors = (labels - probabilities) ** 2
    per_pair = pd.Series(squared_errors).groupby(frame["pair_key"].to_numpy()).mean()
    bins = np.minimum((probabilities * 10).astype(int), 9)
    ece = sum(
        np.mean(bins == index) * abs(np.mean(probabilities[bins == index]) - np.mean(labels[bins == index]))
        for index in range(10) if np.any(bins == index)
    )
    return {
        "row_brier": float(squared_errors.mean()), "pair_brier": float(per_pair.mean()),
        "accuracy": float(np.mean((probabilities >= 0.5) == labels)), "row_ece": float(ece),
        "pairs": len(per_pair), "rows": len(frame),
    }

def find_encoder(data_dir, encoder_dir=None):
    if encoder_dir:
        directory = Path(encoder_dir).expanduser().resolve()
        if not (directory / "modules.json").is_file():
            raise FileNotFoundError(f"Not a SentenceTransformer directory: {directory}")
        return directory
    cache = Path(data_dir) / "approach1_cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    ref = cache / "refs/main"
    if ref.is_file():
        directory = cache / "snapshots" / ref.read_text().strip()
        if (directory / "modules.json").is_file():
            return directory
    snapshots = sorted((cache / "snapshots").glob("*"))
    for directory in snapshots:
        if (directory / "modules.json").is_file():
            return directory
    raise FileNotFoundError("Set AIMS_ENCODER_DIR to a local all-MiniLM-L6-v2 snapshot. No runtime downloads are used.")


def validate_features(examples, features):
    if (features.dtype != np.float32 or features.shape != (len(examples), 644)
            or not np.isfinite(features).all()):
        raise ValueError("Expected a matrix of 644 finite float32 features per response.")
    if not examples["response"].isin([0, 1]).all():
        raise ValueError("Correctness labels must be binary, without missing values.")


def embedding_cache_path(examples, encoder_dir, cache_dir):
    unique = examples[["prompt_group", "item_content"]].drop_duplicates("prompt_group").sort_values("prompt_group")
    files = {p.relative_to(encoder_dir).as_posix(): file_sha256(p) for p in sorted(encoder_dir.rglob("*"))
             if p.is_file() and p.suffix in {".json", ".txt", ".safetensors"}}
    identity = {
        "encoder_files": files, "prompts": unique["prompt_group"].tolist(),
        "recipe": ENCODING_VERSION, "implementation": encoding_signature(),
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return Path(cache_dir) / f"embeddings_{fingerprint}.npz", unique, identity


def load_features(examples, encoder_dir, cache_dir):
    """Cache by encoder, complete prompt identities, and the shared encoding code."""
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache, unique, identity = embedding_cache_path(examples, Path(encoder_dir), cache_dir)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            if not np.array_equal(archive["prompt_groups"], unique["prompt_group"].to_numpy()):
                raise ValueError("Cached prompt identities do not match the current dataset.")
            embeddings = archive["vectors"].copy()
    else:
        encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True)
        encoder.max_seq_length = 256
        vectors = []
        for number, text in enumerate(unique["item_content"], start=1):
            vectors.append(encode_text(encoder, text))
            if number % 250 == 0:
                print(f"Encoding prompts: {number:,}/{len(unique):,}", flush=True)
        embeddings = np.stack(vectors).astype(np.float32)
        np.savez_compressed(cache, vectors=embeddings, prompt_groups=unique["prompt_group"].to_numpy(dtype=str))
        save_json(cache.with_suffix(".json"), identity)
        del encoder
    if embeddings.shape != (len(unique), 384) or embeddings.dtype != np.float32 or not np.isfinite(embeddings).all():
        raise ValueError("Invalid frozen text embeddings.")
    vectors = dict(zip(unique["prompt_group"], embeddings))
    statistics = {row.prompt_group: numeric_vector(row.item_content) for row in unique.itertuples()}
    row_signature = hashlib.sha256(pd.util.hash_pandas_object(
        examples[["source", "response_id", "prompt_group", *SUBJECT_FIELDS]], index=True,
    ).values.tobytes()).hexdigest()
    signature = hashlib.sha256((cache.stem + row_signature + feature_signature()).encode()).hexdigest()
    matrix_path = cache_dir / f"features_{signature}.npy"
    if not matrix_path.is_file():
        matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(len(examples), 644))
        records = examples[[*SUBJECT_FIELDS, "prompt_group"]].to_dict("records")
        for start in range(0, len(records), 4096):
            batch = records[start:start + 4096]
            matrix[start:start + len(batch)] = np.stack([
                np.concatenate((metadata_vector(row), statistics[row["prompt_group"]], vectors[row["prompt_group"]]))
                for row in batch
            ])
        matrix.flush()
        del matrix
    features = np.load(matrix_path, mmap_mode="r")
    validate_features(examples, features)
    return features


def fit_model(method, training, validation, all_features, epochs, seed, fixed_epochs=None, checkpoint_path=None,
              training_weighting="pair", record_common_metrics=False, learning_rate_schedule=False,
              fixed_learning_rates=None):
    if learning_rate_schedule and validation is None:
        raise ValueError("Learning-rate scheduling needs an inner validation subset.")
    if fixed_learning_rates is not None:
        if validation is not None or learning_rate_schedule:
            raise ValueError("A frozen learning-rate plan is only for a full-data refit.")
        rates = np.asarray(fixed_learning_rates, dtype=np.float64)
        if rates.shape != (fixed_epochs or epochs,) or not np.isfinite(rates).all() or (rates <= 0).any():
            raise ValueError("Provide one finite positive learning rate per refit epoch.")
    train_x = all_features[training.index]
    train_y = training["response"].to_numpy(dtype=np.float32)
    if training_weighting != "pair" or method != Method():
        raise ValueError("Only the selected Brier 32/16 FNN with equal-pair weighting is supported.")
    validate_features(training, train_x)
    if validation is not None:
        validate_features(validation, all_features[validation.index])
    sample_weights = pair_weights(training)
    for frame in (training, validation):
        if frame is not None and not frame["response"].isin([0, 1]).all():
            raise ValueError("Fitting and validation labels must be binary, without missing values.")
    start = time.perf_counter()
    import keras
    import torch

    if keras.backend.backend() != "torch":
        raise RuntimeError("Restart the kernel: this experiment requires KERAS_BACKEND=torch.")
    torch.set_num_threads(4)
    keras.utils.set_random_seed(seed)
    layers = [keras.Input(shape=(train_x.shape[1],), dtype="float32")]
    for units in method.hidden:
        layers.extend([
            keras.layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(method.l2)),
            keras.layers.Dropout(method.dropout),
        ])
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4),
                  loss="mse",
                  weighted_metrics=[keras.metrics.MeanSquaredError(name="brier_score"),
                                    keras.metrics.BinaryAccuracy(name="accuracy")])
    callbacks = []
    kwargs = {}
    if validation is not None:
        val_x = all_features[validation.index]
        val_y = validation["response"].to_numpy(dtype=np.float32)

        def record_brier(epoch, logs):
            probabilities = model.predict(val_x, batch_size=2048, verbose=0).reshape(-1)
            validation_scores = scores(validation, probabilities)
            logs["val_pair_brier"] = validation_scores["pair_brier"]
            if learning_rate_schedule or record_common_metrics:
                logs["learning_rate"] = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
            if record_common_metrics:
                # Evaluate with dropout off and identical metrics for every training objective.
                with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                    training_probabilities = model.predict(train_x, batch_size=2048, verbose=0).reshape(-1)
                training_scores = scores(training, training_probabilities)
                logs["train_pair_brier"] = training_scores["pair_brier"]
                logs["train_row_accuracy"] = training_scores["accuracy"]
                logs["val_row_accuracy"] = validation_scores["accuracy"]

        callbacks = [
            keras.callbacks.LambdaCallback(on_epoch_end=record_brier),
            keras.callbacks.EarlyStopping(monitor="val_pair_brier", mode="min", patience=4,
                                           restore_best_weights=True),
        ]
        if learning_rate_schedule:
            callbacks.insert(1, keras.callbacks.ReduceLROnPlateau(
                monitor="val_pair_brier", mode="min", factor=0.5, patience=2,
                min_delta=1e-4, min_lr=1e-5, verbose=1,
            ))
        if checkpoint_path is not None:
            callbacks.insert(1, keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor="val_pair_brier", mode="min", save_best_only=True,
            ))
        kwargs["validation_data"] = (val_x, val_y[:, None], pair_weights(validation))
    if fixed_learning_rates is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(
            lambda epoch, rate: float(fixed_learning_rates[epoch]),
        ))
    if validation is None and record_common_metrics:
        def record_training(epoch, logs):
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                probabilities = model.predict(train_x, batch_size=2048, verbose=0).reshape(-1)
            metrics = scores(training, probabilities)
            logs["train_pair_brier"] = metrics["pair_brier"]
            logs["train_row_accuracy"] = metrics["accuracy"]
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
        callbacks.append(keras.callbacks.LambdaCallback(on_epoch_end=record_training))
    fitted = model.fit(train_x, train_y[:, None], sample_weight=sample_weights,
                       epochs=fixed_epochs or epochs, batch_size=512, verbose=2,
                       callbacks=callbacks, **kwargs)
    if validation is None and checkpoint_path is not None:
        model.save(checkpoint_path)
    history = pd.DataFrame(fitted.history).assign(epoch=lambda frame: np.arange(1, len(frame) + 1)).to_dict("records")
    best_epoch = int(np.argmin(fitted.history["val_pair_brier"]) + 1) if validation is not None else fixed_epochs
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    weights = {key: value for i, layer in enumerate(dense)
               for key, value in zip((f"W{i}", f"b{i}"), layer.get_weights())}
    probes = train_x[:8]
    if not np.allclose(sigmoid(network_logits(probes, weights)),
                       model.predict(probes, verbose=0).reshape(-1), atol=1e-5):
        raise AssertionError("NumPy export does not reproduce the Keras model.")
    keras.backend.clear_session()
    return weights, history, best_epoch, time.perf_counter() - start


def run_training(data_dir, output, encoder_dir=None, seed=42):
    """Reproduce the selected checkpoint; never overwrite the published bundle."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    examples, inventory = load_examples(data_dir)
    encoder_dir = find_encoder(data_dir, encoder_dir)
    features = load_features(examples, encoder_dir, Path(data_dir) / "approach1_cache/combined")
    examples = examples[["source", "response_id", "prompt_group", "pair_key", "response"]].copy()
    gc.collect()
    training, validation = grouped_split(examples, seed=42)
    weights, history, best_epoch, seconds = fit_model(
        Method(), training, validation, features, 20, seed, checkpoint_path=output / "best_fnn.keras",
        record_common_metrics=True, learning_rate_schedule=True,
    )
    np.savez_compressed(output / "weights.npz", **weights)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    probabilities = sigmoid(network_logits(features[validation.index], weights))
    summary = {
        "model": asdict(Method()), "seed": seed, "split_seed": 42, "best_epoch": best_epoch,
        "epochs_completed": len(history), "training_seconds": seconds,
        "training_rows": len(training), "validation_rows": len(validation),
        "validation_metrics": scores(validation, probabilities), "refit_on_all_public": False,
    }
    save_json(output / "summary.json", summary)
    save_json(output / "data_manifest.json", inventory)
    save_json(output / "split_manifest.json", {
        "training_prompts": sorted(training["prompt_group"].unique()),
        "validation_prompts": sorted(validation["prompt_group"].unique()),
    })
    print(json.dumps(summary, indent=2))
    return output



def encoding_signature():
    functions = (select_windows, encode_token_windows, bert_window, encode_text)
    return hashlib.sha256("".join(inspect.getsource(f) for f in functions).encode()).hexdigest()


def feature_signature():
    functions = (clean, hash_vector, _metadata_vector, metadata_vector, numeric_vector, feature_vector)
    return hashlib.sha256("".join(inspect.getsource(f) for f in functions).encode()).hexdigest()


PREDICTORS = ("FNN", "Training mean", "Constant 0.5")
BENCHMARK_NAMES = {
    "multi_swebench": "Multi-SWE-Bench", "real_webagents": "REAL WebAgents",
    "researchcodebench": "ResearchCodeBench", "swe_rebench": "SWE-rebench",
}


def split_lobo(examples, held_out, seed=42):
    test = examples.loc[examples["source"].eq(held_out)].copy()
    if test.empty:
        raise ValueError(f"No responses for held-out benchmark: {held_out}")
    test_prompts = set(test["prompt_group"])
    candidates = examples.loc[~examples["source"].eq(held_out)]
    pool = candidates.loc[~candidates["prompt_group"].isin(test_prompts)]
    if pool["prompt_group"].nunique() < 2:
        raise ValueError("Too few independent training prompts for LOBO.")
    training, validation = grouped_split(pool, seed=seed)
    for frame in (training, validation):
        if frame["source"].eq(held_out).any() or set(frame["prompt_group"]) & test_prompts:
            raise AssertionError("Held-out responses or prompts leaked into training or validation.")
    return training, validation, test, len(candidates) - len(pool)


def aggregate_pair_scores(pair_results):
    results = {}
    for name, part in pair_results.groupby("predictor", sort=False):
        results[name] = {
            "pair_brier": float(part["brier"].mean()),
            "benchmark_mean_brier": float(part.groupby("held_out")["brier"].mean().mean()),
            "row_brier": float(np.average(part["brier"], weights=part["rows"])),
            "row_accuracy": float(np.average(part["accuracy"], weights=part["rows"])),
            "pairs": len(part), "rows": int(part["rows"].sum()),
            "benchmarks": part["held_out"].nunique(),
        }
    return results


def plot_lobo_results(run_dir, fold_scores, histories, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    sources = sorted(histories)
    labels = [BENCHMARK_NAMES.get(source, source) for source in sources]
    fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = np.arange(len(sources))
    for offset, (name, color) in enumerate(zip(PREDICTORS, ("#147d92", "#bc6244", "#737373"))):
        values = fold_scores.loc[fold_scores["predictor"].eq(name)].set_index("held_out").loc[sources, "pair_brier"]
        bars = axis.bar(positions + (offset - 1) * 0.24, values, width=0.23, label=name, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(positions, labels)
    axis.set(title="LOBO: prediction error on each unseen benchmark", ylabel="Equal-pair Brier score; lower is better",
             ylim=(0, float(fold_scores["pair_brier"].max()) * 1.25))
    axis.legend(loc="upper left", ncols=3)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.savefig(run_dir / "lobo_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, source, label in zip(axes.flat, sources, labels):
        history = histories[source]
        best = int(history.loc[history["val_pair_brier"].idxmin(), "epoch"])
        axis.plot(history["epoch"], history["train_pair_brier"], label="training (dropout off)")
        axis.plot(history["epoch"], history["val_pair_brier"], label="inner validation")
        axis.axvline(best, color="gray", linestyle=":", label=f"selected epoch {best}")
        axis.set(title=f"Held out: {label}", xlabel="Epoch", ylabel="Equal-pair Brier")
        axis.set_xticks(history["epoch"].iloc[::max(1, len(history) // 6)].astype(int))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.savefig(run_dir / "lobo_learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for axis, source, label in zip(axes.flat, sources, labels):
        part = predictions.loc[predictions["source"].eq(source)]
        observed, predicted = calibration_curve(part["response"], part["fnn_probability"], n_bins=10)
        axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        axis.plot(predicted, observed, marker="o", color="#147d92", label="FNN")
        axis.set(title=label, xlabel="Predicted probability", ylabel="Observed success rate",
                 xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
    fig.suptitle("LOBO calibration on held-out responses (response-weighted)")
    fig.savefig(run_dir / "lobo_calibration.png", dpi=150)
    plt.close(fig)


def evaluate_lobo_folds(examples, features, method, run_dir, epochs=20, seed=42, split_seed=None,
                        training_weighting="pair", record_common_metrics=False, learning_rate_schedule=False):
    """Share identical folds, scoring, and checkpoint checks across FNN recipes."""
    sources = sorted(examples["source"].unique())
    if len(sources) < 2:
        raise ValueError("LOBO needs at least two benchmarks.")
    fold_rows, pair_rows, prediction_frames, fold_details, histories = [], [], [], [], {}
    for source in sources:
        training, validation, test, removed = split_lobo(examples, source, seed if split_seed is None else split_seed)
        fold_dir = run_dir / "folds" / source
        fold_dir.mkdir(parents=True)
        print(f"\nHold out {source}: fit={len(training):,}, validation={len(validation):,}, test={len(test):,}", flush=True)
        weights, history, best_epoch, seconds = fit_model(
            method, training, validation, features, epochs, seed, checkpoint_path=fold_dir / "best_fnn.keras",
            training_weighting=training_weighting, record_common_metrics=record_common_metrics,
            learning_rate_schedule=learning_rate_schedule,
        )
        probabilities = sigmoid(network_logits(features[test.index], weights))
        np.savez_compressed(fold_dir / "weights.npz", **weights)
        histories[source] = pd.DataFrame(history)
        histories[source].to_csv(fold_dir / "history.csv", index=False)
        save_json(fold_dir / "split_manifest.json", {
            "training_prompts": sorted(training["prompt_group"].unique()),
            "validation_prompts": sorted(validation["prompt_group"].unique()),
            "test_prompts": sorted(test["prompt_group"].unique()),
        })
        import keras
        checkpoint = keras.models.load_model(fold_dir / "best_fnn.keras", compile=False)
        probes = np.linspace(0, len(test) - 1, min(8, len(test))).astype(int)
        restored = checkpoint.predict(features[test.index[probes]], verbose=0).reshape(-1)
        if not np.allclose(restored, probabilities[probes], atol=1e-5):
            raise AssertionError("The saved fold checkpoint differs from the scored FNN.")
        del checkpoint
        keras.backend.clear_session()
        prior = float(training["response"].mean())
        prediction_frames.append(test[[
            "response_id", "source", "benchmark_id", "subject_id", "item_id", "prompt_group", "pair_key", "response",
        ]].assign(fnn_probability=probabilities, training_mean_probability=prior))
        for name, predicted in zip(PREDICTORS, (probabilities, np.full(len(test), prior), np.full(len(test), 0.5))):
            fold_rows.append({"held_out": source, "predictor": name, **scores(test, predicted)})
            errors = test[["pair_key", "response"]].assign(
                squared_error=(predicted - test["response"].to_numpy()) ** 2,
                correct=(predicted >= 0.5) == test["response"].to_numpy(),
            )
            pairs = errors.groupby("pair_key").agg(
                brier=("squared_error", "mean"), accuracy=("correct", "mean"), rows=("response", "size"),
            ).reset_index()
            pair_rows.extend(pairs.assign(held_out=source, predictor=name).to_dict("records"))
        fold_details.append({
            "held_out": source, "training_rows": len(training), "validation_rows": len(validation), "test_rows": len(test),
            "training_prompts": training["prompt_group"].nunique(), "validation_prompts": validation["prompt_group"].nunique(),
            "test_prompts": test["prompt_group"].nunique(), "duplicate_test_prompt_rows_removed_from_pool": removed,
            "best_epoch": best_epoch, "epochs_completed": len(history), "training_seconds": seconds,
            "training_mean_probability": prior, "checkpoint_parity_cases": len(probes),
        })
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_scores.csv", index=False)
        print(f"Held-out equal-pair Brier: {scores(test, probabilities)['pair_brier']:.6f}; best epoch: {best_epoch}", flush=True)
    fold_scores = pd.DataFrame(fold_rows)
    pair_results = pd.DataFrame(pair_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(examples) or predictions.duplicated(["source", "response_id"]).any():
        raise AssertionError("Every eligible response must be tested exactly once.")
    pair_results.to_csv(run_dir / "pair_scores.csv", index=False)
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    return fold_scores, pair_results, predictions, fold_details, histories


def run_lobo(data_dir, output, encoder_dir=None, seed=42):
    """Reuse the existing outer folds and inner checkpoint selection unchanged."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    examples, inventory = load_examples(data_dir)
    features = load_features(examples, find_encoder(data_dir, encoder_dir), Path(data_dir) / "approach1_cache/combined")
    examples = examples[["source", "response_id", "benchmark_id", "subject_id", "item_id",
                         "prompt_group", "pair_key", "response"]].copy()
    folds, pairs, predictions, details, histories = evaluate_lobo_folds(
        examples, features, Method(), output, seed=seed, split_seed=42,
        record_common_metrics=True, learning_rate_schedule=True,
    )
    summary = {"model": asdict(Method()), "seed": seed, "split_seed": 42,
               "aggregate": aggregate_pair_scores(pairs), "folds": details,
               "limitations": "Local public development LOBO, not a CodaBench score."}
    save_json(output / "summary.json", summary)
    save_json(output / "data_manifest.json", inventory)
    plot_lobo_results(output, folds, histories, predictions)
    return output



def restore_encoder():
    """The encoder is stored once in Git, inside the ready-to-upload ZIP."""
    if (SUBMISSION_DIR / "text_encoder/modules.json").is_file():
        return
    archive_path = HERE / "submission_fnn.zip"
    if not archive_path.is_file():
        raise FileNotFoundError("The local text_encoder folder and submission_fnn.zip are both missing.")
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.filename.startswith("text_encoder/") and not member.is_dir():
                target = (SUBMISSION_DIR / member.filename).resolve()
                if not target.is_relative_to(SUBMISSION_DIR.resolve()):
                    raise ValueError("Unsafe encoder path in ZIP.")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    import shutil
                    shutil.copyfileobj(source, destination)
    if not (SUBMISSION_DIR / "text_encoder/modules.json").is_file():
        raise FileNotFoundError("The ZIP does not contain the frozen encoder.")


def verify_checkpoint():
    import keras

    config = json.loads((SUBMISSION_DIR / "config.json").read_text())
    if config["method"]["hidden"] != [32, 16] or config.get("adaptation_precision") is not None:
        raise ValueError("Unexpected checkpoint or training policy.")
    checkpoint_path = (HERE / config["checkpoint_file"]).resolve()
    if not checkpoint_path.is_relative_to(HERE.resolve()):
        raise ValueError("The checkpoint must be inside this project folder.")
    if file_sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("The compiled checkpoint differs from the submission provenance.")
    model = keras.models.load_model(checkpoint_path, compile=False)
    if model.count_params() != 21185:
        raise ValueError("Expected the selected 21,185-parameter FNN.")
    with np.load(SUBMISSION_DIR / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key].copy() for key in archive.files}
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    for index, layer in enumerate(dense):
        for key, value in zip((f"W{index}", f"b{index}"), layer.get_weights()):
            np.testing.assert_array_equal(weights[key], value)
    probes = np.random.default_rng(42).normal(size=(16, 644)).astype(np.float32)
    np.testing.assert_allclose(model.predict(probes, verbose=0).reshape(-1),
                               sigmoid(network_logits(probes, weights)), atol=1e-5)
    keras.backend.clear_session()
    return {"parameters": 21185, "checkpoint_parity_cases": len(probes),
            "checkpoint_sha256": file_sha256(checkpoint_path)}


def build(output=None):
    restore_encoder()
    checks = verify_checkpoint()
    output = Path(output).resolve() if output else HERE / "submission_fnn.zip"
    if output.is_relative_to(SUBMISSION_DIR.resolve()):
        raise ValueError("The ZIP must be outside its source submission folder.")
    required = ("model.py", "config.json", "weights.npz",
                "requirements.txt", "prediction_config.json", "text_encoder/modules.json",
                "text_encoder/model.safetensors")
    if any(not (SUBMISSION_DIR / name).is_file() for name in required):
        raise FileNotFoundError("The submission folder is incomplete.")
    files = [path for path in sorted(SUBMISSION_DIR.rglob("*"))
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(SUBMISSION_DIR).as_posix())
    temporary.replace(output)
    checks.update({"zip_sha256": file_sha256(output), "zip_bytes": output.stat().st_size,
                   "bundle_files": {path.relative_to(SUBMISSION_DIR).as_posix(): file_sha256(path) for path in files},
                   "note": "Local packaging checks only; no CodaBench submission was made."})
    report = json.loads((HERE / "results.json").read_text())
    report["packaging"] = checks
    save_json(HERE / "results.json", report)
    print(f"Ready to upload: {output}")
    print(f"SHA-256: {checks['zip_sha256']}")
    return output


def refit_plan(report):
    """Freeze a full-data policy using inner-validation histories only."""
    folds = report["lobo"]["folds"]
    if len(folds) != 4 or len({fold["held_out"] for fold in folds}) != 4:
        raise ValueError("Expected the four completed LOBO inner-validation runs.")
    stopping_epochs, histories = [], []
    for fold in folds:
        history = pd.DataFrame(fold["history"])
        if history.empty or history["epoch"].tolist() != list(range(1, len(history) + 1)):
            raise ValueError("Inner-validation epochs are incomplete or out of order.")
        if not np.isfinite(history[["val_pair_brier", "learning_rate"]]).all().all():
            raise ValueError("Inner-validation histories must be finite.")
        best_epoch = int(history.loc[history["val_pair_brier"].idxmin(), "epoch"])
        if best_epoch != fold["best_epoch"]:
            raise ValueError("The saved best epoch differs from its inner-validation history.")
        stopping_epochs.append(best_epoch)
        histories.append(history)
    epochs = int(np.floor(np.median(stopping_epochs) + 0.5))
    if not 1 <= epochs <= 20:
        raise ValueError("The derived full-data epoch count is outside the evaluated budget.")
    rates = [round(float(np.median([
        history["learning_rate"].iloc[min(index, len(history) - 1)] for history in histories
    ])), 10) for index in range(epochs)]
    if not np.isfinite(rates).all() or min(rates) <= 0:
        raise ValueError("Invalid frozen learning-rate plan.")
    return {
        "epochs": epochs, "learning_rates": rates, "inner_best_epochs": stopping_epochs,
        "epoch_policy": "Median inner-validation best epoch, rounded half up.",
        "learning_rate_policy": "Per-epoch median of the four inner-validation learning-rate histories; "
                                "carry the last rate forward after a fold ends.",
        "outer_test_scores_used": False, "early_stopping": False, "validation_rows": 0,
        "note": "The schedule is frozen before fitting; no validation labels remain for adaptive stopping.",
    }


def refit_all_data(data_dir, output, encoder_dir=None, seed=42):
    """Fit fresh weights on every eligible public response, then publish the bundle."""
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.is_relative_to(HERE):
        raise ValueError("Keep detailed run artifacts outside the compact project folder.")
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((HERE / "results.json").read_text())
    plan = refit_plan(report)
    save_json(output / "refit_plan.json", plan)
    examples, inventory = load_examples(data_dir)
    previous = report["data_manifest"]
    if [row for row in inventory if row["used"]] != [row for row in previous if row["used"]]:
        raise ValueError("Training data changed since evaluation; repeat validation first.")
    if set(examples["source"]) != {fold["held_out"] for fold in report["lobo"]["folds"]}:
        raise ValueError("The full-data fit must use the same four binary benchmarks.")
    features = load_features(examples, find_encoder(data_dir, encoder_dir), data_dir / "approach1_cache/combined")
    probe_indices = []
    for _, part in examples.groupby("source", sort=True):
        lengths = part["item_content"].str.len()
        probe_indices.extend((part.index[0], part.index[-1], lengths.idxmin(), lengths.idxmax()))
    probe_indices = sorted(set(probe_indices))
    probes = examples.loc[probe_indices].to_dict("records")
    examples = examples[["source", "response_id", "prompt_group", "pair_key", "response"]].copy()
    gc.collect()
    sample_weights = pair_weights(examples)
    totals = pd.Series(sample_weights).groupby(examples["pair_key"].to_numpy()).sum()
    np.testing.assert_allclose(totals, totals.iloc[0], rtol=1e-5)
    np.testing.assert_allclose(sample_weights.mean(), 1.0, rtol=1e-5)
    print(f"Fresh full-data FNN: {len(examples):,} responses; {plan['epochs']} fixed epochs; no validation set.", flush=True)
    print("Frozen learning rates:", plan["learning_rates"], flush=True)
    checkpoint = output / "model.keras"
    weights, history, _, seconds = fit_model(
        Method(), examples, None, features, plan["epochs"], seed, fixed_epochs=plan["epochs"],
        fixed_learning_rates=plan["learning_rates"], checkpoint_path=checkpoint, record_common_metrics=True,
    )
    np.savez_compressed(output / "weights.npz", **weights)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    probabilities = sigmoid(network_logits(features, weights))
    summary = {
        "model": asdict(Method()), "seed": seed, "refit_on_all_public": True,
        "training_rows": len(examples), "validation_rows": 0,
        "training_prompts": int(examples["prompt_group"].nunique()),
        "training_pairs": int(examples["pair_key"].nunique()),
        "benchmarks": sorted(examples["source"].unique()),
        "epochs_completed": len(history), "training_seconds": seconds,
        "plan": plan, "history": history, "training_fit_metrics": scores(examples, probabilities),
        "evaluation_note": "Full-data metrics are in-sample, not validation, LOBO, or a CodaBench score.",
    }
    save_json(output / "summary.json", summary)
    save_json(output / "data_manifest.json", inventory)
    if len(history) != plan["epochs"] or any(key.startswith("val_") for key in history[0]):
        raise AssertionError("Full-data fitting did not follow its fixed, validation-free plan.")
    config = {
        "recipe_version": 2, "feature_mode": "sbert", "encoding_recipe": ENCODING_VERSION,
        "method": asdict(Method()), "input_dimensions": 644, "encoder_max_seq_length": 256,
        "maximum_windows": 4, "content_tokens_per_window": 254,
        "training_loss": "brier", "training_weighting": "equal model-benchmark pairs",
        "adaptation_precision": None, "shrinkage": 1.0, "refit_on_all_public": True,
        "seed": seed, "refit_epochs": plan["epochs"], "training_rows": len(examples), "validation_rows": 0,
        "checkpoint_file": "model.keras", "checkpoint_sha256": file_sha256(checkpoint),
        "refit_policy": plan, "source_run": output.name,
        "evaluation_note": "LOBO evaluates the recipe through separate fold fits, not this full-data checkpoint.",
    }
    shutil.copy2(checkpoint, HERE / "model.keras")
    shutil.copy2(output / "weights.npz", SUBMISSION_DIR / "weights.npz")
    save_json(SUBMISSION_DIR / "config.json", config)
    report["final"] = summary
    report["active_model"] = "Full-data refit"
    save_json(HERE / "results.json", report)
    build()
    from .submission import model as runtime
    runtime.load_bundle.cache_clear()
    runtime.item_embedding.cache_clear()
    errors = []
    for index, row in zip(probe_indices, probes):
        predicted = runtime.predict([row, row])
        np.testing.assert_allclose(predicted, probabilities[index], atol=1e-5)
        if predicted != runtime.predict([row, row], labeled=[[[row, row], 1]]):
            raise AssertionError("The fixed FNN unexpectedly used revealed labels.")
        errors.append(abs(predicted - float(probabilities[index])))
    report = json.loads((HERE / "results.json").read_text())
    report["packaging"]["real_task_parity"] = {
        "cases": len(errors), "benchmarks": 4, "maximum_probability_absolute_error": max(errors),
        "revealed_labels_ignored": True,
    }
    save_json(HERE / "results.json", report)
    save_json(output / "submission_checks.json", report["packaging"])
    print(json.dumps(summary, indent=2), flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "lobo", "refit", "package"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoder-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "package":
        build(args.output)
    else:
        if args.data_dir is None or args.output is None:
            parser.error("--data-dir and a new --output folder are required for training or evaluation.")
        function = {"train": run_training, "lobo": run_lobo, "refit": refit_all_data}[args.command]
        function(args.data_dir, args.output, args.encoder_dir, args.seed)


if __name__ == "__main__":
    main()
