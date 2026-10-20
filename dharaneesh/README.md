# Approach 1: Text-Window FNN

Open **Approach1_Experiments.ipynb** for the saved training outputs, graphs, and results. This folder now contains only the selected combined FNN. Earlier experiments were archived locally outside this repository.

## Method

The frozen all-MiniLM-L6-v2 encoder represents a task using up to four selected 254-token text windows. Their normalized embeddings are averaged and normalized again into 384 features. Short prompts keep the original single-window representation. The input also includes 256 hashed model-metadata features and four text statistics, giving 644 features in total.

The FNN has hidden layers of 32 and 16 ReLU neurons, 35% dropout after each hidden layer, L2 regularization of 0.0001, and a sigmoid probability output. Training uses equal model-benchmark-pair weights, Brier loss, Adam at an initial learning rate of 0.0005, a batch size of 512, a maximum of 20 epochs, early stopping with patience four, and ReduceLROnPlateau with patience two. There is no encoder fine-tuning, ensemble, logistic replacement, or revealed-label adaptation.

## Results

Four-fold local LOBO Brier was **0.261456**, compared with **0.283996** for the original FNN, **0.250000** for constant-0.5 predictions, and **0.231825** for the fitting-mean baseline. Lower is better. The new FNN improved on the original but did not beat these baselines. LOBO used initialization seed 42 and is public development evaluation, not a CodaBench score.

Prompt-grouped 80/20 validation over initialization seeds 42, 123, and 2026 gave a mean equal-pair Brier of **0.161223**, with a sample standard deviation of **0.000700**. This familiar-benchmark validation also selects checkpoints and is not an independent test.

## Files

- `Approach1_Experiments.ipynb`: main notebook, with saved outputs and optional training cells.
- `training.py`: validated data joins, shared feature preparation, and FNN training.
- `lobo.py`: the existing held-out-benchmark evaluation procedure.
- `results/validation/`: compiled selected model, training history, and validation results.
- `results/lobo/`: selected-method fold results and graphs.
- `submission/`: prediction code, trained weights, configuration, and local frozen encoder.
- `submission_fnn.zip`: ready-to-upload CodaBench file, with `model.py` at the ZIP root.
- `build_submission.py`: checks the saved checkpoint and rebuilds the ZIP.
- `test_pipeline.py`: split, feature, model, and portable-inference checks.

The packaged model is the combined method's **seed-42 prompt-validation checkpoint**, selected at epoch 12 after a 16-epoch run. It was fitted on **109,305 responses**, with **26,977 validation responses**. It is **not** a fit on all 136,282 responses and is **not** an outer-LOBO test-selected checkpoint. The scheduler ran during training; the best checkpoint preceded its first learning-rate reduction. The 0.261456 score evaluates the recipe using four separately fitted LOBO models, not the packaged checkpoint on an independent test set.

## Run Locally

Run commands from the repository root using the same Python environment as the notebook:

```powershell
python -m pip install -r dharaneesh/requirements.txt
python -m unittest dharaneesh.test_pipeline -v
python -m dharaneesh.build_submission
```

The large encoder is stored once in Git, inside `submission_fnn.zip`. On a fresh checkout, `build_submission` restores it into `submission/text_encoder/` before checking and packaging. The complete local submission folder can then be inspected or zipped again without downloading anything.

To train a fresh checkpoint without replacing the published submission:

```powershell
python -m dharaneesh.training --data-dir "PATH_TO_MEASUREMENT_DB" --output "PATH_TO_A_NEW_RUN_FOLDER"
python -m dharaneesh.lobo --data-dir "PATH_TO_MEASUREMENT_DB" --output "PATH_TO_A_NEW_LOBO_FOLDER"
```

Set `AIMS_ENCODER_DIR` in the notebook, or pass `--encoder-dir`, if the encoder is not in the dataset's local Hugging Face cache. New runs do not automatically replace the saved model or ZIP. Submission to CodaBench is manual.
