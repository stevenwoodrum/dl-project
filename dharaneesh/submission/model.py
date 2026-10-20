"""Portable prediction entry point for the selected 32/16 FNN."""

from functools import lru_cache
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np

if __package__:
    from .features import RECIPE_VERSION, clean, feature_vector, network_logits, sigmoid
    from .text_encoding import ENCODING_VERSION, encode_text
else:
    from features import RECIPE_VERSION, clean, feature_vector, network_logits, sigmoid
    from text_encoding import ENCODING_VERSION, encode_text

BASE = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def load_bundle():
    config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
    if config["recipe_version"] != RECIPE_VERSION or config["encoding_recipe"] != ENCODING_VERSION:
        raise ValueError("The feature recipe does not match the saved model.")
    if config["method"]["hidden"] != [32, 16] or config.get("adaptation_precision") is not None:
        raise ValueError("This bundle supports only the selected nonadaptive 32/16 FNN.")
    with np.load(BASE / "weights.npz", allow_pickle=False) as archive:
        weights = {key: archive[key].copy() for key in archive.files}
    shapes = {"W0": (644, 32), "b0": (32,), "W1": (32, 16), "b1": (16,),
              "W2": (16, 1), "b2": (1,)}
    if set(weights) != set(shapes) or any(weights[k].shape != shape or
            weights[k].dtype != np.float32 or not np.isfinite(weights[k]).all() for k, shape in shapes.items()):
        raise ValueError("The trained FNN weights have invalid shapes or values.")
    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(4)
    encoder = SentenceTransformer(str(BASE / "text_encoder"), device="cpu", local_files_only=True)
    encoder.max_seq_length = config["encoder_max_seq_length"]
    return config, weights, encoder


@lru_cache(maxsize=512)
def item_embedding(text):
    return encode_text(load_bundle()[2], text)


def predict(input: list, labeled: list | None = None) -> float:
    # Revealed labels are deliberately unused in this fixed Approach 1 predictor.
    subject, item = input
    _, weights, _ = load_bundle()
    vector = feature_vector(subject, item, item_embedding(clean(item.get("item_content", ""))))
    probability = float(sigmoid(network_logits(vector[None, :], weights))[0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Prediction must be a finite probability between zero and one.")
    return probability
