# Approach 1: Text-Window FNN

Open **Approach1_Experiments.ipynb**. It contains the architecture, saved training logs, LOBO results, graphs, and short report summary.

The model combines frozen MiniLM SBERT text-window embeddings, hashed model metadata, and four text statistics. The FNN has 32/16 ReLU hidden layers, dropout, L2 regularization, and a sigmoid output. Validation uses Brier loss, equal-pair weights, Adam, early stopping, and learning-rate scheduling. No logistic model, encoder fine-tuning, or revealed-label adaptation is used.

## Final Model

The final FNN is trained from fresh initialization on **all 136,282 eligible native-binary responses** from four benchmarks, including the earlier validation rows. The other two dataset sections contain nonbinary outcomes and are not converted into artificial binary labels.

The refit uses **7 fixed epochs**, chosen as the median of the four inner-validation best epochs (9, 4, 5, 13). Its learning-rate plan is frozen from those inner-validation histories before training. All seven rates are 0.0005. There is no validation split, adaptive early stopping, or live plateau scheduler in the all-data fit.

The earlier local LOBO Brier is **0.261456**, compared with 0.283996 for the original FNN, 0.250000 for constant-0.5 predictions, and 0.231825 for the fitting-mean baseline. LOBO evaluates the recipe through separate fold models. Full-data training metrics are in-sample only, and no CodaBench score is claimed for the final model.

## Essential Files

- `Approach1_Experiments.ipynb`: main notebook, with embedded graphs and outputs.
- `approach1.py`: shared training, LOBO evaluation, and packaging code.
- `model.keras`: compiled final FNN trained on all eligible data.
- `results.json`: completed training logs, evaluation results, and packaging checks.
- `submission/`: complete local predictor folder; its encoder is already bundled in the ZIP.
- `submission_fnn.zip`: the file to upload manually to CodaBench.

Earlier models, detailed reports, and tests are archived locally outside this repository. The frozen encoder is ignored as a loose Git folder to avoid storing it twice; `package` restores it from the ZIP on a fresh checkout.

## Run

From the repository root, install `python -m pip install -r dharaneesh/requirements.txt` and select that environment in Jupyter. The notebook shows saved results without retraining by default.

Rebuild the ZIP with `python -m dharaneesh.approach1 package`.

To repeat the full-data refit, use `python -m dharaneesh.approach1 refit --data-dir "PATH_TO_MEASUREMENT_DB" --output "PATH_TO_A_NEW_RUN_FOLDER" --encoder-dir "PATH_TO_LOCAL_MINILM"`. This replaces the active model and ZIP after fitting and checking the export. Use `train` instead of `refit` for a fresh 80/20 development run, or `lobo` for held-out-benchmark evaluation; these save separate run folders.
