"""Exported Keras Approach 1 prediction interface.

The notebook defines the classifier architecture and saves it in classifier.keras.
It imports preprocessing from feature_utils.py. Export
copies that same file to feature_utils.py beside this adapter so tokenization,
chunk pooling, and metadata ordering stay identical at prediction time.
"""

from functools import lru_cache
from pathlib import Path
import ast
import importlib.util
import json
import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import keras
import numpy as np
import torch

if keras.backend.backend() != "torch":
    raise RuntimeError("Restart the kernel with KERAS_BACKEND=torch before importing this adapter.")

BASE = Path(__file__).resolve().parent


def write_feature_helpers(source, destination):
    """Bundle the original preprocessing without its unused PyTorch predictor."""
    tree = ast.parse(Path(source).read_text())
    excluded = {"Approach1", "_load_bundle", "_question_vector", "predict"}
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)) or node.name not in excluded
    ]
    Path(destination).write_text(ast.unparse(tree) + "\n")


@lru_cache(maxsize=1)
def _load_bundle():
    torch.set_num_threads(4)
    specification = importlib.util.spec_from_file_location(
        "keras_export_feature_utils", BASE / "feature_utils.py"
    )
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    config = json.loads((BASE / "config.json").read_text())
    if config["text_pooling"] != "token_weighted_mean_l2":
        raise ValueError("Unsupported saved chunk pooling method.")
    metadata = helpers.MetadataEncoder(config["categories"])
    encoder = helpers.make_text_encoder(
        BASE / "text_encoder", config["text_max_length"],
        device="cpu", chunk_size=config["text_chunk_size"]
    )
    expected_dim = metadata.dimensions + encoder.get_sentence_embedding_dimension()
    if config["input_dim"] != expected_dim:
        raise ValueError("Saved feature dimensions do not match the encoders.")
    with keras.device("cpu"):
        classifier = keras.models.load_model(BASE / "classifier.keras", compile=False)
    if classifier.input_shape != (None, expected_dim) or classifier.output_shape != (None, 1):
        raise ValueError("Saved classifier shapes do not match the feature recipe.")
    return classifier, metadata, encoder, helpers


@lru_cache(maxsize=128)
def _question_vector(prompt):
    _, _, encoder, helpers = _load_bundle()
    return helpers.embed_text(encoder, [prompt])[0]


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    classifier, metadata, _, _ = _load_bundle()
    features = np.concatenate([
        metadata.transform([subject])[0],
        _question_vector(item.get("item_content", "")),
    ]).astype(np.float32)
    with keras.device("cpu"), torch.no_grad():
        logits = classifier(features[None, :], training=False)
        probability = float(keras.ops.convert_to_numpy(keras.ops.sigmoid(logits)).reshape(-1)[0])
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("The prediction is not a finite probability.")
    return probability


# Approach 1 uses no revealed labels and retains predict(input, labeled) -> float.
