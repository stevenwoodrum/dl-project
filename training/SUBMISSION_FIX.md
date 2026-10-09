# Tokenizer compatibility fix and working submission

## Cause and repair

The hosted tokenizer lacks callable `prepare_for_model`. The original chunk preparation called that method directly and failed. `feature_utils.py::prepare_chunk_features` now calls it when available; otherwise it creates the equivalent MiniLM/BERT `[CLS] + token IDs + [SEP]` sequence, attention mask and token-type IDs, after verifying the expected token template.

`prompt_chunks` calls this helper. Chunk boundaries, the 2,048-token prompt budget, 512-token windows, token-weighted pooling, normalization and model weights are preserved. The notebook's export cell includes this shared helper in its exported predictor. This source fix was already included in commit `4cc5dcb`; this update adds the exact successful artifact and its verified provenance.

## Controlled evidence

- Submission **970373**: original cached chunk pipeline failed.
- Submission **970402**: Finished; an unconditional assertion confirms `prepare_for_model` is absent on that hosted tokenizer.
- Submission **970404**: Finished after changing only the preparation helper from 970373.
- Submission **970407**: the complete trained predictor Finished with the same helper repair.
- Requirements-only Keras import, saved-model loading and forward-pass controls **970380–970382** all Finished.

The combination gives high confidence in the API mismatch as the failure mechanism. Exact private package versions and the original raw traceback remain unavailable. The local smoke test uses the caller's installed Python environment, so passing locally does not establish compatibility with the hosted package stack.

## Exact working artifact

- File: [final18_tokfix_req.zip](../final18_tokfix_req.zip) (Git LFS).
- Codabench submission: **970407**, verified Finished on October 8, 2026.
- SHA-256: `9679966960eeaf762bd73eaecfae78213641f6b214aa564da09d4a353759121c`.
- Size: **83,721,397 bytes**.
- Brier: **0.2575673301**; calibration ECE: **0.2660412553**. The Brier result is worse than the constant-0.5 baseline's 0.25. This is a runtime fix, not a model-quality improvement.
- Only `feature_utils.py` changed from original failed full submission 970077. Classifier, configuration, local encoder files, requirements and adapter stayed byte-identical.
- Root requirements: `keras==3.15.1` and `sentence-transformers>=3.3,<6`; no vendored libraries.
- Saved classifier: 460 inputs, 128-unit ReLU layer, dropout 0.2, one output logit; L2 0.0001 on both Dense kernels, L1 zero; checkpoint epoch 9 selected by validation equal-pair Brier. This archive preserves that earlier trained model; the notebook's current architecture settings may differ.

## Validation and reuse

The exact ZIP passed official ZIP and isolated-worker checks offline under Linux with Transformers 4.46.3 and 5.0.0. Sixteen prompt cases preserve chunk IDs, masks, weights and embeddings; eight checked full-model probabilities match the original v4 predictor. The branch's metadata/encoder/chunk functions were compared by AST against the successful archive.

After cloning, run `git lfs pull`. Verify the download with:

```sh
shasum -a 256 final18_tokfix_req.zip
```

For a new model, run the notebook Setup cell, choose a new `RUN_NAME`, train or retain the intended model, export, and run the export checks. Keep declared settings matched to the trained model. Validate new artifacts independently; do not replace the proven helper with the removed tokenizer API.
