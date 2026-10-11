"""Compare simple models under an item split and held-out benchmark evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from .submission.features import (
    RECIPE_VERSION, SBERT_ID, SUBJECT_FIELDS, adapt_probability, clean,
    feature_vector, network_logits, prompt_key, sigmoid,
)

HERE = Path(__file__).resolve().parent
SUBMISSION_DIR = HERE / "submission"
BUDGETS = (0, 1, 3, 7, 15, 31)
ALC_WEIGHTS = (0.1, 0.2, 0.2, 0.2, 0.2, 0.1)
RUNTIME_SUBJECT_FIELDS = (*SUBJECT_FIELDS, "release_date", "access_date", "subject_features_extra")


@dataclass(frozen=True)
class Method:
    name: str
    features: str
    estimator: str
    loss: str = "bce"
    hidden: tuple = (64, 16)
    dropout: float = 0.35
    l2: float = 1e-4


METHODS = {
    method.name: method for method in [
        Method("metadata_logistic", "metadata", "logistic"),
        Method("hash_text_logistic", "hash_text", "logistic"),
        Method("sbert_logistic", "sbert", "logistic"),
        Method("sbert_fnn_bce", "sbert", "fnn", "bce"),
        Method("sbert_fnn_brier", "sbert", "fnn", "brier"),
    ]
}


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


def cached_embeddings(examples, data_dir, encoder_dir, cache_dir):
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)
    encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True)
    encoder.max_seq_length = 256
    unique = examples[["prompt_group", "item_content"]].drop_duplicates("prompt_group").sort_values("prompt_group")
    encoder_files = {path.relative_to(encoder_dir).as_posix(): file_sha256(path)
                     for path in sorted(encoder_dir.rglob("*"))
                     if path.is_file() and path.suffix in {".json", ".txt", ".safetensors"}}
    fingerprint = hashlib.sha256(json.dumps({
        "encoder_files": encoder_files, "prompts": unique["prompt_group"].tolist(),
        "max_seq_length": 256, "normalize_embeddings": True,
    }, sort_keys=True).encode()).hexdigest()
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = Path(cache_dir) / f"sbert_{fingerprint}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive["vectors"].copy()
        print(f"Loaded checked SBERT cache: {vectors.shape}", flush=True)
    else:
        legacy_dir = Path(data_dir) / "approach1_cache"
        legacy_path = legacy_dir / "item_embeddings_all-MiniLM-L6-v2.npy"
        legacy_index_path = legacy_dir / "item_embedding_index.parquet"
        vectors = None
        if legacy_path.is_file() and legacy_index_path.is_file():
            legacy_index = pd.read_parquet(legacy_index_path).set_index("item_key")["embedding_index"]
            item_keys = examples["source"] + "::" + examples["item_id"].astype(str)
            prompt_indices = dict(zip(examples["prompt_group"], item_keys.map(legacy_index)))
            ordered = np.asarray([prompt_indices[group] for group in unique["prompt_group"]])
            if np.isfinite(ordered).all():
                legacy_vectors = np.load(legacy_path)
                proposed = legacy_vectors[ordered.astype(int)]
                probes = np.linspace(0, len(unique) - 1, 5).astype(int)
                expected = encoder.encode(unique["item_content"].iloc[probes].tolist(),
                                          normalize_embeddings=True, show_progress_bar=False)
                if np.allclose(proposed[probes], expected, atol=1e-5, rtol=1e-4):
                    vectors = proposed.astype(np.float32)
                    print("Reused the original item cache after five encoder parity checks.", flush=True)
        if vectors is None:
            vectors = encoder.encode(unique["item_content"].tolist(), batch_size=64,
                                     normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
        np.savez_compressed(path, vectors=vectors)
    if vectors.shape != (len(unique), 384) or not np.isfinite(vectors).all():
        raise ValueError("Invalid SBERT cache dimensions or values.")
    return dict(zip(unique["prompt_group"], vectors)), {
        "repo": SBERT_ID, "snapshot": encoder_dir.name, "files": encoder_files,
        "cache_fingerprint": fingerprint, "max_seq_length": 256,
        "normalization": "l2", "truncation": "first 256 tokens, no chunking",
    }


def build_features(examples, mode, vectors=None):
    records = examples[[*RUNTIME_SUBJECT_FIELDS, "item_content", "prompt_group"]].to_dict("records")
    return np.stack([
        feature_vector(record, record, mode, vectors[record["prompt_group"]] if mode == "sbert" else None)
        for record in records
    ])


def fit_model(method, training, validation, all_features, epochs, seed, fixed_epochs=None):
    train_x = all_features[training.index]
    train_y = training["response"].to_numpy()
    sample_weights = pair_weights(training)
    start = time.perf_counter()
    if method.estimator == "logistic":
        estimator = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs", random_state=seed)
        estimator.fit(train_x, train_y, sample_weight=sample_weights)
        weights = {"W0": estimator.coef_.T.astype(np.float32),
                   "b0": estimator.intercept_.astype(np.float32)}
        history = []
        best_epoch = 1
    else:
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
                      loss="binary_crossentropy" if method.loss == "bce" else "mse")
        callbacks = []
        kwargs = {}
        if validation is not None:
            val_x = all_features[validation.index]
            val_y = validation["response"].to_numpy()

            def record_brier(epoch, logs):
                probabilities = model.predict(val_x, batch_size=2048, verbose=0).reshape(-1)
                logs["val_pair_brier"] = scores(validation, probabilities)["pair_brier"]

            callbacks = [
                keras.callbacks.LambdaCallback(on_epoch_end=record_brier),
                keras.callbacks.EarlyStopping(monitor="val_pair_brier", mode="min", patience=4,
                                               restore_best_weights=True),
            ]
            kwargs["validation_data"] = (val_x, val_y[:, None])
        fitted = model.fit(train_x, train_y[:, None], sample_weight=sample_weights,
                           epochs=fixed_epochs or epochs, batch_size=512, verbose=2,
                           callbacks=callbacks, **kwargs)
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


def budget_scores(frame, probabilities, shrinkage, precision, seed):
    """Replay a fixed random acquisition trajectory on held-out benchmark pairs."""
    frame = frame.reset_index(drop=True)
    probabilities = 0.5 + shrinkage * (np.asarray(probabilities) - 0.5)
    records = []
    for pair, part in frame.groupby("pair_key", sort=True):
        groups = sorted(part["prompt_group"].unique())
        if len(groups) < 80:
            continue
        pair_seed = int(hashlib.sha256(f"{seed}:{pair}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(pair_seed)
        rng.shuffle(groups)
        acquisition_groups = groups[:len(groups) // 2]
        evaluation = part.loc[part["prompt_group"].isin(groups[len(groups) // 2:])]
        acquired_rows = [int(rng.choice(part.loc[part["prompt_group"].eq(group)].index))
                         for group in acquisition_groups[:31]]
        target = probabilities[evaluation.index]
        labels = evaluation["response"].to_numpy()
        values = []
        for budget in BUDGETS:
            evidence = acquired_rows[:budget]
            if precision == "empirical":
                predictions = np.full(len(target), (part.loc[evidence, "response"].sum() + 1) / (budget + 2))
            elif precision is None or budget == 0:
                predictions = target
            else:
                adapted_reference = adapt_probability(0.5, probabilities[evidence],
                                                       frame.loc[evidence, "response"].to_numpy(), precision)
                offset = np.log(adapted_reference / (1 - adapted_reference))
                clipped = np.clip(target, 1e-6, 1 - 1e-6)
                predictions = sigmoid(np.log(clipped / (1 - clipped)) + offset)
            values.append(float(np.mean((predictions - labels) ** 2)))
        records.append({"pair": pair, "source": part["source"].iloc[0],
                        "brier_alc": float(np.dot(values, ALC_WEIGHTS)),
                        **{f"B{budget}": value for budget, value in zip(BUDGETS, values)}})
    return records


def export_bundle(run_dir, method, weights, config, encoder_dir=None):
    bundle = Path(run_dir) / "submission"
    bundle.mkdir()
    shutil.copy2(SUBMISSION_DIR / "model.py", bundle / "model.py")
    shutil.copy2(SUBMISSION_DIR / "features.py", bundle / "features.py")
    np.savez_compressed(bundle / "weights.npz", **weights)
    save_json(bundle / "config.json", config)
    if method.features == "sbert":
        shutil.copytree(encoder_dir, bundle / "text_encoder", ignore=shutil.ignore_patterns(
            ".cache", "*.bin", "*.onnx", "openvino", "onnx", "README.md", "train_script.py", "*.msgpack", "*.h5",
        ))
        (bundle / "requirements.txt").write_text("sentence-transformers>=3.3,<6\n", encoding="utf-8")
    save_json(bundle / "prediction_config.json", {"concurrency": 1})
    archive_path = Path(run_dir) / f"{Path(run_dir).name}.zip"
    archive_path.parent.mkdir(exist_ok=True)
    if archive_path.exists():
        raise FileExistsError(f"Choose another run name: {archive_path}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(bundle).as_posix())
    return bundle, archive_path


def package_saved_submission(output=None):
    """Rebuild the selected predictor ZIP; no retraining or network access occurs."""
    required = {"model.py", "features.py", "config.json", "weights.npz"}
    missing = [name for name in sorted(required) if not (SUBMISSION_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete saved submission: {missing}")
    archive_path = Path(output) if output else HERE / "submission.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(SUBMISSION_DIR.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(SUBMISSION_DIR).as_posix())
    print(f"Submission ZIP: {archive_path.resolve()}")
    print(f"SHA-256: {file_sha256(archive_path)}")
    return archive_path.resolve()


def plot_results(run_dir, comparison, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    best = comparison.sort_values("mean_pair_alc").drop_duplicates("method").head(8).copy()
    names = best["method"].str.replace("_", " ")
    axes[0].barh(names, best["mean_pair_alc"], color="teal")
    axes[0].axvline(0.25, linestyle="--", color="gray", label="constant 0.5")
    axes[0].invert_yaxis()
    axes[0].set(title="Best development setting for each method", xlabel="Equal-pair Brier ALC")
    axes[0].legend()
    if history:
        frame = pd.DataFrame(history)
        axes[1].plot(frame["epoch"], frame["val_pair_brier"], marker="o")
        axes[1].set(title="Final fit checkpoint selection", xlabel="Epoch", ylabel="Validation Brier (no L2 penalty)")
    else:
        for method_name in (best.iloc[0]["method"], "beta_mean", "constant_0.5"):
            row = best.loc[best["method"].eq(method_name)]
            if not row.empty:
                values = row.iloc[0][[f"B{budget}" for budget in BUDGETS]].astype(float)
                axes[1].plot(range(len(BUDGETS)), values, marker="o", label=method_name.replace("_", " "))
        axes[1].set(title="Held-out score across revealed-label budgets", xlabel="Revealed labels", ylabel="Equal-pair Brier")
        axes[1].set_xticks(range(len(BUDGETS)), BUDGETS)
        axes[1].legend(fontsize=8)
    fig.savefig(Path(run_dir) / "comparison.png", dpi=160)
    plt.close(fig)


def run_experiments(data_dir, run_name, methods=None, epochs=20, seed=42, encoder_dir=None):
    run_dir = Path(data_dir) / "approach1_outputs" / "experiment_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    examples, inventory = load_examples(data_dir)
    save_json(run_dir / "data_manifest.json", inventory)
    chosen_methods = [METHODS[name] for name in (methods or METHODS)]
    print(f"Native binary responses: {len(examples):,}; prompts: {examples['prompt_group'].nunique():,}", flush=True)
    sources = sorted(examples["source"].unique())
    save_json(run_dir / "settings.json", {"seed": seed, "epochs": epochs, "methods": [asdict(m) for m in chosen_methods],
                                        "sources": sources, "label_policy": "native binary only",
                                        "library_versions": {p: importlib.metadata.version(p) for p in (
                                            "numpy", "pandas", "scikit-learn", "keras", "torch", "sentence-transformers")}})
    vectors, encoder_source = None, None
    if any(method.features == "sbert" for method in chosen_methods):
        encoder_dir = find_encoder(data_dir, encoder_dir)
        vectors, encoder_source = cached_embeddings(
            examples, data_dir, encoder_dir, Path(data_dir) / "approach1_cache" / "experiment_cache",
        )
        save_json(run_dir / "encoder_source.json", encoder_source)
    validation_rows, adaptation_rows, split_records = [], [], []
    for feature_mode in sorted({method.features for method in chosen_methods}):
        features = build_features(examples, feature_mode, vectors)
        for method in [method for method in chosen_methods if method.features == feature_mode]:
            for source in sources:
                test = examples.loc[examples["source"].eq(source)]
                test_groups = set(test["prompt_group"])
                pool = examples.loc[~examples["source"].eq(source) & ~examples["prompt_group"].isin(test_groups)]
                training, stopping = grouped_split(pool, seed=seed)
                print(f"\n{method.name}: hold out {source}; fit={len(training):,}, stop={len(stopping):,}, test={len(test):,}", flush=True)
                weights, history, best_epoch, seconds = fit_model(method, training, stopping, features, epochs, seed)
                probabilities = sigmoid(network_logits(features[test.index], weights))
                validation_rows.append({"method": method.name, "held_out": source,
                                        "best_epoch": best_epoch, "train_seconds": seconds, **scores(test, probabilities)})
                save_json(run_dir / f"{method.name}_{source}_history.json", history)
                split_records.append({"method": method.name, "held_out": source,
                                      "fit_prompts": sorted(training["prompt_group"].unique()),
                                      "stop_prompts": sorted(stopping["prompt_group"].unique()),
                                      "test_prompts": sorted(test_groups)})
                for shrinkage in (1.0, 0.75, 0.5):
                    for precision in (None, 1.0, 4.0, 16.0):
                        for row in budget_scores(test, probabilities, shrinkage, precision, seed):
                            adaptation_rows.append({"method": method.name, "shrinkage": shrinkage,
                                                    "precision": "none" if precision is None else str(precision), **row})
            pd.DataFrame(validation_rows).to_csv(run_dir / "held_out_scores.csv", index=False)
            pd.DataFrame(adaptation_rows).to_csv(run_dir / "budget_pair_scores.csv", index=False)
        del features
    for source in sources:
        test = examples.loc[examples["source"].eq(source)]
        for name, precision in (("constant_0.5", None), ("beta_mean", "empirical")):
            for row in budget_scores(test, np.full(len(test), 0.5), 1.0, precision, seed):
                adaptation_rows.append({"method": name, "shrinkage": 1.0,
                                        "precision": "none" if precision is None else precision, **row})
    budget_results = pd.DataFrame(adaptation_rows)
    if budget_results.empty:
        raise ValueError("No subject-benchmark pairs have 80 distinct items for a budget simulation.")
    budget_results.to_csv(run_dir / "budget_pair_scores.csv", index=False)
    comparison = budget_results.groupby(["method", "shrinkage", "precision"]).agg(
        mean_pair_alc=("brier_alc", "mean"), pair_count=("pair", "size"),
        **{f"B{budget}": (f"B{budget}", "mean") for budget in BUDGETS},
    ).reset_index().sort_values("mean_pair_alc")
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    save_json(run_dir / "split_manifest.json", split_records)
    selected = comparison.loc[comparison["method"].isin(METHODS)].iloc[0]
    method = METHODS[selected["method"]]
    features = build_features(examples, method.features, vectors)
    fit, stopping = grouped_split(examples, seed=seed)
    weights, history, best_epoch, seconds = fit_model(method, fit, stopping, features, epochs, seed)
    final_validation = scores(stopping, sigmoid(network_logits(features[stopping.index], weights)))
    # The final artifact is refit on all public rows for a fixed, previously selected epoch count.
    weights, _, _, refit_seconds = fit_model(method, examples, None, features, epochs, seed, fixed_epochs=best_epoch)
    config = {
        "recipe_version": RECIPE_VERSION, "feature_mode": method.features,
        "method": asdict(method), "shrinkage": float(selected["shrinkage"]),
        "adaptation_precision": None if selected["precision"] == "none" else float(selected["precision"]),
        "encoder_max_seq_length": 256, "training_loss": method.loss,
        "selection_metric": "held-out benchmark equal-pair simulated Brier ALC",
        "selection_is_development": True, "refit_epochs": best_epoch,
    }
    bundle, archive_path = export_bundle(run_dir, method, weights, config, encoder_dir)
    # Preserve the zero-label Approach 1 comparison as a separate archive.
    no_adaptation = dict(config, adaptation_precision=None)
    baseline_dir = run_dir / f"{run_name}_approach1_only"
    baseline_dir.mkdir()
    _, baseline_archive = export_bundle(baseline_dir, method, weights, no_adaptation, encoder_dir)
    # Export parity is checked against independently reconstructed runtime inputs.
    import sys
    sys.path.insert(0, str(bundle))
    specification = importlib.util.spec_from_file_location("dharaneesh_export_check", bundle / "model.py")
    exported = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(exported)
    probes = examples.iloc[np.linspace(0, len(examples) - 1, 8).astype(int)]
    for index, row in probes.iterrows():
        subject = {field: row[field] for field in RUNTIME_SUBJECT_FIELDS}
        item = {"item_content": row["item_content"], "benchmark_id": "benchmark_unknown",
                "item_features": "", "interactors": ""}
        expected = 0.5 + config["shrinkage"] * (float(sigmoid(network_logits(features[[index]], weights))[0]) - 0.5)
        if not np.isclose(exported.predict([subject, item], []), expected, atol=1e-5):
            raise AssertionError("Exported inference differs from the training pipeline.")
    summary = {
        "run": run_name, "selected": config, "selected_development_alc": float(selected["mean_pair_alc"]),
        "eligible_pair_count": int(selected["pair_count"]), "final_checkpoint_validation": final_validation,
        "checkpoint_training_seconds": seconds, "all_public_refit_seconds": refit_seconds,
        "zip": str(archive_path), "zip_sha256": file_sha256(archive_path),
        "approach1_zip": str(baseline_archive), "export_parity_cases": len(probes),
        "warning": "Development folds were used for method selection; this is not an unbiased final test score.",
    }
    save_json(run_dir / "summary.json", summary)
    plot_results(run_dir, comparison, history)
    print(comparison.head(12).to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--methods", nargs="+", choices=list(METHODS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-dir", type=Path, default=os.environ.get("AIMS_ENCODER_DIR"))
    args = parser.parse_args()
    run_experiments(args.data_dir, args.run_name, args.methods, args.epochs, args.seed, args.encoder_dir)


if __name__ == "__main__":
    main()
