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



from bisect import bisect_left
import re
import numpy as np

MAX_TOKENS = 256
WINDOW_TOKENS = 254
ENCODING_VERSION = "context_code_windows_v1"


def select_windows(text, offsets, width=WINDOW_TOKENS):
    """Select content-only windows, without consulting benchmark IDs or labels."""
    count = len(offsets)
    if count <= width:
        return [(0, count)]
    starts = [offset[0] for offset in offsets]

    def token_at(character):
        return min(count, bisect_left(starts, character))

    abstract = re.search(r"\\begin\{abstract\}", text)
    document = re.search(r"\\begin\{document\}", text)
    context = token_at((abstract or document).end()) if abstract or document else 0
    end_document = re.search(r"\\end\{document\}", text)
    context_end = token_at(end_document.start()) if end_document else count
    code = re.search(r"(?im)^\s*Here is the code[^\n]*:\s*", text)
    guidelines = re.search(r"(?im)^\s*Please implement the missing code in the TODO blocks", text)
    useful_end = token_at(guidelines.start()) if code and guidelines else count
    code_start = token_at(code.end()) if code else 0
    targets = list(re.finditer(r"(?im)^\s*#\s*TODO:\s*Implement block[^\n]*", text))
    targets = [token_at(match.start()) for match in targets
               if code and code.end() <= match.start() < (guidelines.start() if guidelines else len(text))]
    if code and targets:
        proposals = [context, max(context, (context + context_end - width) // 2),
                     targets[0] - 96, targets[-1] - 96, code_start]
    else:
        proposals = np.linspace(context, max(context, useful_end - width), 4).astype(int).tolist()
    proposals.extend(np.linspace(context, max(context, useful_end - width), 4).astype(int).tolist())
    windows = []
    for proposed in proposals:
        start = min(max(0, int(proposed)), max(0, useful_end - width))
        window = (start, min(start + width, useful_end))
        if window not in windows:
            windows.append(window)
        if len(windows) == 4:
            break
    return windows

def encode_token_windows(encoder, sequences, batch_size=64):
    """Run the frozen SBERT stack on exact token windows, with no retokenization."""
    import torch

    tokenizer = encoder.tokenizer
    encoder.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start + batch_size]
            inputs = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt")
            inputs = {key: value.to(encoder.device) for key, value in inputs.items()}
            vectors = encoder(inputs)["sentence_embedding"]
            result.append(torch.nn.functional.normalize(vectors, p=2, dim=1).cpu().numpy())
            if len(sequences) > 1000 and start % 1024 == 0:
                print(f"Frozen SBERT windows: {min(start + batch_size, len(sequences)):,}/{len(sequences):,}", flush=True)
    return np.concatenate(result).astype(np.float32)

def bert_window(tokenizer, ids):
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("The frozen MiniLM tokenizer must define BERT CLS and SEP tokens.")
    return [tokenizer.cls_token_id, *ids, tokenizer.sep_token_id]

def encode_text(encoder, text):
    """Match the evaluated short-prompt and normalized-window representations."""
    tokenized = encoder.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                  truncation=False, verbose=False)
    ids = tokenized["input_ids"]
    if len(ids) <= WINDOW_TOKENS:
        return encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True,
                              show_progress_bar=False)[0].astype(np.float32)
    windows = select_windows(text, tokenized["offset_mapping"])
    sequences = [bert_window(encoder.tokenizer, ids[a:b]) for a, b in windows]
    if any(len(sequence) > MAX_TOKENS for sequence in sequences):
        raise ValueError("A selected window exceeds the encoder limit.")
    pooled = encode_token_windows(encoder, sequences).mean(axis=0)
    return (pooled / max(np.linalg.norm(pooled), 1e-12)).astype(np.float32)



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
