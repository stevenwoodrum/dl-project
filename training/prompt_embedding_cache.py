"""Local disk cache for frozen prompt vectors; independent of classifier settings."""

import hashlib
import inspect
import json
import os
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np


def _prompt_hashes(prompts):
    return np.asarray([
        hashlib.sha256(prompt.encode("utf-8")).hexdigest().encode("ascii")
        for prompt in prompts
    ], dtype="S64")


def _cache_key(encoder_dir, prompt_hashes, settings, encoding_functions):
    digest = hashlib.sha256()
    recipe = {
        "cache_version": 1,
        "settings": settings,
        "packages": {name: version(name) for name in (
            "sentence-transformers", "transformers", "torch", "tokenizers", "numpy"
        )},
        "encoding_code": [inspect.getsource(function) for function in encoding_functions],
    }
    digest.update(json.dumps(recipe, sort_keys=True).encode("utf-8"))
    digest.update(prompt_hashes.tobytes())
    encoder_dir = Path(encoder_dir)
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f"Local encoder directory is missing: {encoder_dir}")
    for path in sorted(encoder_dir.rglob("*")):
        relative = path.relative_to(encoder_dir)
        if not path.is_file() or ".cache" in relative.parts or path.name in {
            "README.md", "source.json", "data_config.json"
        }:
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _valid_vectors(vectors, count):
    return (
        vectors.dtype == np.float32 and vectors.ndim == 2
        and vectors.shape[0] == count and vectors.shape[1] > 0
        and np.isfinite(vectors).all()
        and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    )


def cached_prompt_embeddings(
    prompts, *, encoder_dir, cache_dir, settings, encoding_functions, compute
):
    """Load matching vectors, or compute and atomically save a new cache entry."""
    prompts = list(prompts)
    hashes = _prompt_hashes(prompts)
    key = _cache_key(encoder_dir, hashes, settings, encoding_functions)
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"{key}.npz"
    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as saved:
                vectors = saved["vectors"]
                if (
                    saved["key"].item() == key
                    and np.array_equal(saved["prompt_hashes"], hashes)
                    and _valid_vectors(vectors, len(prompts))
                ):
                    print(f"Loaded cached embeddings: {len(prompts):,} prompts from {cache_file}")
                    return vectors
            print("Embedding cache failed validation; recomputing.")
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            print("Embedding cache could not be read; recomputing.")
    else:
        print("No matching embedding cache; encoding prompts once.")

    vectors = np.asarray(compute(), dtype=np.float32)
    if not _valid_vectors(vectors, len(prompts)):
        raise ValueError("Prompt embeddings must be a finite, normalized matrix with one row per prompt.")
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as output:
            temporary = Path(output.name)
            np.savez(output, key=np.asarray(key), prompt_hashes=hashes, vectors=vectors)
        os.replace(temporary, cache_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Saved cached embeddings: {len(prompts):,} prompts to {cache_file}")
    return vectors
