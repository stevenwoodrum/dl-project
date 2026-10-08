"""Shared metadata and frozen text features for the Keras notebook."""

from functools import lru_cache

from pathlib import Path

import json

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

import torch

from torch import nn

from torch.nn import functional as F

from sentence_transformers import SentenceTransformer

from tqdm import tqdm

BASE = Path(__file__).resolve().parent

SUBJECT_COLUMNS = [
    "normalized_name", "provider", "reasoning_effort"
]

class MetadataEncoder:
    def __init__(self, categories):
        self.categories = categories
        self.lookup = {}
        offset = 0
        for column, values in categories.items():
            self.lookup[column] = {value: offset + index for index, value in enumerate(values)}
            offset += len(values)
        self.dimensions = offset

    # Unknown categories leave their field's one-hot entries at zero.

    @classmethod
    def fit(cls, records):
        return cls({
            column: sorted({record.get(column, "") for record in records})
            for column in SUBJECT_COLUMNS
        })

    # Fit category dictionaries on training subjects only.

    def transform(self, records):
        result = np.zeros((len(records), self.dimensions), dtype=np.float32)
        for row, record in enumerate(records):
            for column, lookup in self.lookup.items():
                index = lookup.get(record.get(column, ""))
                if index is not None:
                    result[row, index] = 1.0
        return result

def make_text_encoder(directory, max_length=2048, device="cpu", chunk_size=512):
    if not Path(directory).is_dir():
        raise FileNotFoundError(f"Local encoder directory is missing: {directory}")
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("The total prompt token limit must be a positive integer.")
    encoder = SentenceTransformer(str(directory), device=str(device))
    capacity = encoder[0].auto_model.config.max_position_embeddings
    special_tokens = encoder.tokenizer.num_special_tokens_to_add(pair=False)
    if not isinstance(chunk_size, int) or not special_tokens < chunk_size <= capacity:
        raise ValueError(f"Chunk size must exceed {special_tokens} and be at most {capacity}.")
    encoder.max_seq_length = chunk_size
    encoder.prompt_token_limit = max_length
    encoder.chunk_token_limit = chunk_size
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder

def prepare_chunk_features(tokenizer, token_ids):
    legacy_prepare = getattr(tokenizer, "prepare_for_model", None)
    if callable(legacy_prepare):
        return legacy_prepare(
            token_ids, add_special_tokens=True, truncation=False,
            return_attention_mask=True, return_token_type_ids=True, verbose=False
        )
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None or tokenizer.encode("", add_special_tokens=True) != [cls_id, sep_id]:
        raise ValueError("Expected the saved MiniLM/BERT [CLS] sequence [SEP] template")
    input_ids = [cls_id, *token_ids, sep_id]
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "token_type_ids": [0] * len(input_ids),
    }

def prompt_chunks(encoder, text):
    tokenizer = encoder.tokenizer
    token_ids = tokenizer.encode(
        text, add_special_tokens=False, truncation=True,
        max_length=encoder.prompt_token_limit, verbose=False
    )
    payload_size = encoder.chunk_token_limit - tokenizer.num_special_tokens_to_add(pair=False)
    pieces = [token_ids[start:start + payload_size] for start in range(0, len(token_ids), payload_size)]
    if not pieces:
        pieces = [[]]
    for piece in pieces:
        features = prepare_chunk_features(tokenizer, piece)
        if len(features["input_ids"]) > encoder.chunk_token_limit:
            raise ValueError("A prepared chunk exceeds the encoder's positional limit.")
        yield features, max(1, len(piece))

def embed_text(encoder, texts, batch_size=32, show_progress=False):
    if batch_size < 1:
        raise ValueError("Chunk batch size must be positive.")
    texts = list(texts)
    dimensions = encoder.get_sentence_embedding_dimension()
    totals = np.zeros((len(texts), dimensions), dtype=np.float32)
    weights = np.zeros(len(texts), dtype=np.float32)
    pending = []
    encoder.eval()

    def flush_batch():
        features = encoder.tokenizer.pad(
            [entry[2] for entry in pending], padding=True,
            return_tensors="pt", verbose=False
        )
        features = {key: value.to(encoder.device) for key, value in features.items()}
        with torch.inference_mode():
            embeddings = encoder(features)["sentence_embedding"]
            vectors = F.normalize(embeddings, p=2, dim=1).cpu().numpy().astype(np.float32)
        for (owner, weight, _), vector in zip(pending, vectors):
            totals[owner] += weight * vector
            weights[owner] += weight
        pending.clear()

    # Batch chunks across prompts, then accumulate them on CPU to bound GPU memory.

    for owner, text in tqdm(
        enumerate(texts), total=len(texts), desc="Embedding prompts", disable=not show_progress
    ):
        for features, weight in prompt_chunks(encoder, text):
            pending.append((owner, weight, features))
            if len(pending) == batch_size:
                flush_batch()
    if pending:
        flush_batch()
    means = totals / np.maximum(weights[:, None], 1.0)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    result = (means / np.maximum(norms, 1e-12)).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Chunk pooling produced a nonfinite embedding.")
    return result
