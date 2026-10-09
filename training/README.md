# Keras training notebook

Open `all_benchmarks_flattened_training_keras.ipynb` from this folder or the repository root. This is the Keras training notebook with its required preprocessing, export adapter, and prompt embedding cache. Saved outputs were cleared because the working notebook had execution counts from different runs. Its current architecture, training settings, BCE objective, split, plots, and early-stopping patience 5 are preserved.

The classifier uses Keras with the torch backend. Frozen MiniLM text encoding also uses torch. `feature_utils.py` contains preprocessing only; no separate PyTorch classifier is included in this folder.

## Environment and data

Use Python 3.12 and install `pip install -r training/requirements.txt` from the repository root. Select that Python environment as the notebook kernel. These package pins record the environment used to verify the included cache.

The benchmark parquet files and MiniLM model weights are external inputs. If already downloaded, set these environment variables before launching Jupyter/VS Code, or edit the paths in the first and Setup cells:

- `AIMS_DATA_DIR`: directory containing `matharena/`, `mmdocrag/`, `multi_swebench/`, `real_webagents/`, `researchcodebench/`, and `swe_rebench/`.
- `AIMS_ENCODER_DIR`: local MiniLM directory containing `model.safetensors`, tokenizer files, and `1_Pooling/`.
- `AIMS_STARTER_KIT_DIR`: official competition starting kit, needed only for the final ZIP/smoke-test cell.

Defaults are `training/data`, `training/resources/text_encoder`, and `training/official-starting-kit`. The data and encoder revisions are recorded in `data_manifest.json` and `encoder_source.json`. The export cell records those same provenance files, so use these exact inputs.

To obtain the pinned public inputs, run this from the repository root with network access:

```python
import hashlib
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

root = Path("training")
manifest = json.loads((root / "data_manifest.json").read_text())
for entry in manifest["files"]:
    downloaded = Path(hf_hub_download(
        manifest["repo"], entry["path"], repo_type="dataset",
        revision=manifest["revision"], local_dir=root / "data",
    ))
    assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == entry["sha256"]
source = json.loads((root / "encoder_source.json").read_text())
snapshot_download(
    source["repo"], revision=source["revision"],
    local_dir=root / "resources/text_encoder",
    allow_patterns=[
        "config.json", "config_sentence_transformers.json", "model.safetensors",
        "modules.json", "sentence_bert_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt", "1_Pooling/*",
    ],
)
```

Obtain the official starting kit from the competition resources before running the final optional submission check.

## Included prompt cache

`prompt-embedding-cache/` contains a finite, normalized float32 matrix for **8,794 unique prompts × 384 dimensions**, ordered by the prompt hashes stored in the NPZ. The accompanying manifest records checksums and validation evidence.

The notebook automatically loads a matching cache. Its key includes prompt content/order, encoder files, token/chunk settings, preprocessing functions, and relevant library versions. Changing those causes a rebuild; changing classifier dropout, regularization, epochs, or hidden units does not. Cross-platform package differences can intentionally cause a rebuild. The encoder files are still needed to validate the cache fingerprint and export a predictor.

The packaged cache was checked against the existing local cache and then loaded again with encoding disabled to prove a cache hit. The tokenizer compatibility helper is included in the fingerprint and exported predictor. The active notebook in the original workspace is separate from this repository copy.

## Prompt coverage

For the completed local run and the matching prompt set:

- Encoding failures: **0 / 8,794 = 0%**.
- Encoded in full within the 2,048-token budget: **8,311 / 8,794 = 94.51%**.
- Truncated after 2,048 content tokens: **483 / 8,794 = 5.49%**. These still have valid embeddings; the remaining text was omitted.
- Needed multiple chunks: **2,577 / 8,794 = 29.30%**.
- Total chunks: **13,611**, using a 512-token window including the two BERT special tokens.

These percentages are over unique local prompts, not response rows or private Codabench inputs. Hosted diagnostic failures do not establish a percentage of private prompts that failed.

## Export

Choose a new `RUN_NAME` for each saved run. The export writes `runs/` and `dist/`, which are ignored by Git. Run the export checks before uploading a new artifact. The notebook is training code; inclusion here does not establish that its exported predictor has passed the hosted competition runtime.


## Verified submission compatibility

The included `prepare_chunk_features` helper repairs the hosted tokenizer API mismatch. The notebook exports that helper automatically. The exact previously trained model with this fix passed Codabench as submission **970407**; download [final18_tokfix_req.zip](../final18_tokfix_req.zip) with Git LFS. This verifies the compatibility path, while newly trained exports still need their own checks. See [SUBMISSION_FIX.md](SUBMISSION_FIX.md) for the controlled evidence, artifact checksum and model settings.
