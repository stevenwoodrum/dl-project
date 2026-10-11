# Approach 1 experiments

**Main Jupyter notebook: Approach1_Experiments.ipynb.**

The notebook compares five models that predict whether an AI subject will succeed on a task. It contains the completed local results. It does not connect to CodaBench or submit anything.

## Necessary files

| File | Purpose |
| --- | --- |
| Approach1_Experiments.ipynb | Main notebook: settings, training, comparison, and ZIP creation |
| experiments.py | Shared training, evaluation, and export functions |
| requirements.txt | Packages needed for the notebook and training |
| test_pipeline.py | Small local tests for splits, scores, and inference |
| submission/ | Selected predictor, shared features, configuration, and trained weights |
| submission.zip | Ready-to-upload archive with model.py at its root |
| results/ | Recorded comparison table, fold scores, summary, and chart |

The extra development scripts and old run directories were moved outside this repository to a local archive. Teammates' files were not changed. New training runs and embedding caches are stored in the external dataset directory, not in this branch.

## Run the notebook

1. Open Approach1_Experiments.ipynb and select the existing **Python 3 (approach1)** kernel.
2. With RUN_NEW_EXPERIMENT = False, Run All displays the saved results and rebuilds submission.zip. It does not retrain or need the dataset.
3. To train again, set RUN_NEW_EXPERIMENT = True and choose a fresh RUN_NAME. Set AIMS_DATA_DIR if the benchmark folders are not next to dl-project.
4. SBERT experiments need a local all-MiniLM-L6-v2 encoder. The original Approach 1 cache is detected automatically; otherwise set AIMS_ENCODER_DIR. No encoder download occurs during inference.

Install requirements.txt in the notebook's Python environment when necessary. Restart the kernel if Keras was previously imported with TensorFlow: this comparison uses Keras's torch backend. Training features are numeric float32 arrays, avoiding the previous object-dtype error.

## What was tested

Both FNNs use the same 64/16 hidden layers, dropout 0.35, L2 0.0001, and grouped early stopping. One trains with binary cross-entropy; the other with MSE, which is the binary Brier loss. BCE is valid even though the competition evaluates Brier ALC. A different training loss does not automatically improve that evaluation score.

The inputs are normalized model name, provider, harness, harness version, reasoning effort, and simple task-text statistics. Text variants add hashed words and word pairs or a frozen 384-dimensional MiniLM embedding. Local subject and benchmark IDs, responses, and execution traces are not prediction features. Revealed responses are used only by the optional adaptation step.

Only four native-binary, item-level benchmarks were used: Multi-SWE-Bench, REAL, ResearchCodeBench, and SWE-rebench. They contain 136,282 responses and 8,794 distinct stripped task texts. MathArena and MMDocRAG have fractional outcomes and were not silently converted into binary labels.

Evaluation holds out a whole benchmark, removes duplicate held-out prompts from training, and uses an inner 80/20 prompt-grouped split for checkpoint selection. Scores weight subject-benchmark pairs equally. The local budget simulation uses separate acquisition/evaluation items, nested revealed labels at budgets 0, 1, 3, 7, 15, and 31, and Brier ALC weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1. Adaptation fits a regularized logit offset using only the current pair's revealed labels.

## Measured results

Lower is better. Each learned model below uses its best tested development setting: shrinkage 0.5 toward probability 0.5 and adaptation precision 1.0.

| Predictor with adaptation | Local simulated Brier ALC |
| --- | ---: |
| Metadata logistic regression | **0.198097** |
| Simple Beta(1,1) revealed-label mean | 0.198661 |
| SBERT FNN trained with Brier | 0.200122 |
| SBERT FNN trained with BCE | 0.200361 |
| Hashed-text logistic regression | 0.205496 |
| SBERT logistic regression | 0.220053 |
| Constant probability 0.5 | 0.250000 |

The selected submission is logistic regression, not an FNN. Both FNN implementations remain available in the notebook comparison. Without revealed-label adaptation, the selected model's local ALC is 0.238338. Its final weights were refit on all public binary rows; there is no independent test score for that refit.

Nishan's reported hosted score was 0.257567. Our local 0.198097 is numerically lower, but the data, sampling, and evaluation environments differ. It does not prove a better CodaBench result. The near tie with the adaptive-mean baseline is not statistically significant evidence. These folds were also used for method selection, so the results are development estimates. The earlier within-benchmark 0.1703 score is not comparable either.

## Team review and validation

Reviewed branch snapshots: approach-1-steven f0b64fb, nicks-branch f98534b, nishans-branch 27edec7, and main f459c12. Nishan already uses BCE and duplicate-prompt grouping. Nick tests benchmark transfer, but trains on fractional grades while evaluating thresholded labels. Steven's training source is an item-text-only, 500-row MathArena prototype. Our comparison uses consistent native-binary targets and subject-item inputs. Source snapshots can differ from the ZIPs your teammates actually submitted.

Run local tests from the repository root with `python -m unittest dharaneesh.test_pipeline -v`. The selected inference code uses NumPy only, resolves artifacts relative to model.py, and does not import TensorFlow or Keras. Local execution checks do not guarantee hosted compatibility. Upload submission.zip yourself, not the notebook or the containing folder.

Dataset: [Measurement Data Bank](https://huggingface.co/datasets/aims-foundations/measurement-db). Interface and scoring: [Official competition](https://aimslab.stanford.edu/competition). Pretrained encoder: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Code developed with assistance from Codex; disclose that assistance in the project report.
