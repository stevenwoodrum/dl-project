"""CodaBench entry point. No TensorFlow or Keras is needed for inference."""

from functools import lru_cache
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

from features import (
    RECIPE_VERSION, adapt_probability, feature_vector, network_logits, sigmoid, subject_key,
)

BASE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_bundle():
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    if config["recipe_version"] != RECIPE_VERSION:
        raise ValueError("Feature recipe and saved configuration do not match.")
    with np.load(BASE / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key].copy() for key in archive.files}
    encoder = None
    if config["feature_mode"] == "sbert":
        from sentence_transformers import SentenceTransformer
        import torch

        torch.set_num_threads(4)
        encoder = SentenceTransformer(str(BASE / "text_encoder"), device="cpu", local_files_only=True)
        encoder.max_seq_length = config["encoder_max_seq_length"]
    return config, weights, encoder


@lru_cache(maxsize=512)
def item_embedding(text):
    _, _, encoder = load_bundle()
    return encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                          show_progress_bar=False)[0].astype(np.float32)


def base_probability(subject, item):
    config, weights, _ = load_bundle()
    embedding = item_embedding(item.get("item_content", "")) if config["feature_mode"] == "sbert" else None
    vector = feature_vector(subject, item, config["feature_mode"], embedding)
    if vector.shape[0] != weights["W0"].shape[0]:
        raise ValueError("Feature dimensions do not match the trained model.")
    probability = float(sigmoid(network_logits(vector[None, :], weights))[0])
    return 0.5 + config.get("shrinkage", 1.0) * (probability - 0.5)


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    config, _, _ = load_bundle()
    probability = base_probability(subject, item)
    precision = config.get("adaptation_precision")
    if precision is not None and labeled:
        evidence_probabilities, labels = [], []
        benchmark = item.get("benchmark_id", "")
        for pair, label in labeled:
            evidence_subject, evidence_item = pair
            if (subject_key(evidence_subject) == subject_key(subject)
                    and evidence_item.get("benchmark_id", "") == benchmark):
                evidence_probabilities.append(base_probability(evidence_subject, evidence_item))
                labels.append(float(label))
        probability = adapt_probability(probability, evidence_probabilities, labels, precision)
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Prediction must be a finite probability between 0 and 1.")
    return float(probability)
