"""The same stateless feature transforms are used in fitting and prediction."""

from collections import Counter
from functools import lru_cache
import hashlib
import re

import numpy as np

RECIPE_VERSION = 2
SUBJECT_FIELDS = ("normalized_name", "provider", "harness", "harness_version", "reasoning_effort")
META_DIM = 256
SBERT_ID = "sentence-transformers/all-MiniLM-L6-v2"


def clean(value):
    return "" if value is None else str(value).strip()


def prompt_key(text):
    return hashlib.sha256(clean(text).encode("utf-8")).hexdigest()


def hash_vector(tokens, dimensions):
    vector = np.zeros(dimensions, dtype=np.float32)
    for token, count in Counter(tokens).items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        vector[int.from_bytes(digest, "little") % dimensions] += np.log1p(count)
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


def numeric_vector(text):
    text = clean(text)
    length = max(1, len(text))
    return np.asarray([
        min(np.log1p(length) / 12.0, 2.0),
        min(np.log1p(text.count("\n")) / 8.0, 2.0),
        sum(character.isdigit() for character in text) / length,
        sum(character in "{}[]();=" for character in text) / length,
    ], dtype=np.float32)


def feature_vector(subject, item, embedding):
    vector = np.concatenate((metadata_vector(subject), numeric_vector(item.get("item_content", "")),
                             np.asarray(embedding, dtype=np.float32))).astype(np.float32)
    if vector.shape != (644,) or not np.isfinite(vector).all():
        raise ValueError("Expected 644 finite float32 features.")
    return vector


def network_logits(features, weights):
    values = np.asarray(features, dtype=np.float32)
    for index in range(3):
        values = values @ weights[f"W{index}"] + weights[f"b{index}"]
        if index < 2:
            values = np.maximum(values, 0.0)
    return values.reshape(-1)


def sigmoid(logits):
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64)))
