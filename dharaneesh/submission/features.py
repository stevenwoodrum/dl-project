"""Feature preparation shared by training and the exported predictor."""

from collections import Counter
from functools import lru_cache
import hashlib
import re

import numpy as np

SUBJECT_FIELDS = (
    "normalized_name", "provider", "harness", "harness_version", "reasoning_effort",
)
META_DIM = 256
TEXT_DIM = 1024
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"
RECIPE_VERSION = 1


def clean(value):
    return "" if value is None else str(value).strip()


def prompt_key(text):
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()


def subject_key(subject):
    # Identity uses every runtime subject field when matching revealed evidence.
    fields = (*SUBJECT_FIELDS, "release_date", "access_date", "subject_features_extra")
    return tuple(clean(subject.get(field, "")) for field in fields)


def hash_vector(tokens, dimensions):
    vector = np.zeros(dimensions, dtype=np.float32)
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % dimensions
        vector[index] += np.log1p(count)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


@lru_cache(maxsize=2048)
def _metadata_vector(values):
    tokens = []
    for field, value in zip(SUBJECT_FIELDS, values):
        value = value.lower()
        tokens.append(f"{field}={value or '<missing>'}")
        tokens.extend(f"{field}:{word}" for word in re.findall(r"[a-z0-9]+", value))
    return hash_vector(tokens, META_DIM)


def metadata_vector(subject):
    return _metadata_vector(tuple(clean(subject.get(field, "")) for field in SUBJECT_FIELDS))


@lru_cache(maxsize=2048)
def text_hash_vector(text):
    # Bound work for very long inputs; the same bound applies at training and inference.
    words = re.findall(r"[a-z0-9_]+", clean(text).lower()[:50000])[:8192]
    tokens = words + [a + " " + b for a, b in zip(words, words[1:])]
    return hash_vector(tokens, TEXT_DIM)


def numeric_vector(text):
    text = clean(text)
    length = max(1, len(text))
    return np.asarray([
        min(np.log1p(length) / 12.0, 2.0),
        min(np.log1p(text.count("\n")) / 8.0, 2.0),
        sum(character.isdigit() for character in text) / length,
        sum(character in "{}[]();=" for character in text) / length,
    ], dtype=np.float32)


def feature_vector(subject, item, mode, embedding=None):
    text = clean(item.get("item_content", ""))
    parts = [metadata_vector(subject), numeric_vector(text)]
    if mode == "hash_text":
        parts.append(text_hash_vector(text))
    elif mode == "sbert":
        if embedding is None:
            raise ValueError("SBERT features require an item embedding.")
        parts.append(np.asarray(embedding, dtype=np.float32))
    elif mode != "metadata":
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.concatenate(parts).astype(np.float32)


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64)))


def network_logits(features, weights):
    values = np.asarray(features, dtype=np.float32)
    layers = len(weights) // 2
    for index in range(layers):
        values = values @ weights[f"W{index}"] + weights[f"b{index}"]
        if index < layers - 1:
            values = np.maximum(values, 0.0)
    return values.reshape(-1)


def adapt_probability(probability, evidence_probabilities, labels, precision):
    """Fit a regularized logit offset using only the supplied pair's evidence."""
    if precision is None or not len(labels):
        return float(probability)
    if precision <= 0:
        raise ValueError("Adaptation prior precision must be positive.")
    evidence = np.clip(np.asarray(evidence_probabilities), 1e-6, 1 - 1e-6)
    logits = np.log(evidence / (1 - evidence))
    labels = np.asarray(labels, dtype=np.float64)
    offset = 0.0
    for _ in range(12):
        probabilities = sigmoid(logits + offset)
        gradient = np.sum(probabilities - labels) + precision * offset
        curvature = np.sum(probabilities * (1 - probabilities)) + precision
        step = gradient / curvature
        offset = float(np.clip(offset - step, -8.0, 8.0))
        if abs(step) < 1e-7:
            break
    probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
    return float(sigmoid(np.log(probability / (1 - probability)) + offset))
