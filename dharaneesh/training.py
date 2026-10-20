"""Training and feature preparation for the selected Approach 1 FNN."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import hashlib
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

from .submission.features import SUBJECT_FIELDS, clean, metadata_vector, network_logits, numeric_vector, prompt_key, sigmoid
from .submission.text_encoding import ENCODING_VERSION, encode_text

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
        "recipe": ENCODING_VERSION, "implementation": file_sha256(SUBMISSION_DIR / "text_encoding.py"),
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
    signature = hashlib.sha256((cache.stem + row_signature + file_sha256(SUBMISSION_DIR / "features.py")).encode()).hexdigest()
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
              training_weighting="pair", record_common_metrics=False, learning_rate_schedule=False):
    if learning_rate_schedule and validation is None:
        raise ValueError("Learning-rate scheduling needs an inner validation subset.")
    train_x = all_features[training.index]
    train_y = training["response"].to_numpy(dtype=np.float32)
    if training_weighting != "pair" or method != Method():
        raise ValueError("Only the selected Brier 32/16 FNN with equal-pair weighting is supported.")
    validate_features(training, all_features[training.index])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_training(args.data_dir, args.output, args.encoder_dir, args.seed)


if __name__ == "__main__":
    main()
