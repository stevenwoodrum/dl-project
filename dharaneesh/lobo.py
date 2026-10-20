"""Evaluate the current FNN on one unseen public benchmark at a time."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from .experiments import (
    HERE, SUBMISSION_DIR, Method, build_features, cached_embeddings, file_sha256,
    find_encoder, fit_model, grouped_split, load_examples, save_json, scores,
)
from .submission.features import network_logits, sigmoid

PREDICTORS = ("FNN", "Training mean", "Constant 0.5")
BENCHMARK_NAMES = {
    "multi_swebench": "Multi-SWE-Bench", "real_webagents": "REAL WebAgents",
    "researchcodebench": "ResearchCodeBench", "swe_rebench": "SWE-rebench",
}


def split_lobo(examples, held_out, seed=42):
    test = examples.loc[examples["source"].eq(held_out)].copy()
    if test.empty:
        raise ValueError(f"No responses for held-out benchmark: {held_out}")
    test_prompts = set(test["prompt_group"])
    candidates = examples.loc[~examples["source"].eq(held_out)]
    pool = candidates.loc[~candidates["prompt_group"].isin(test_prompts)]
    if pool["prompt_group"].nunique() < 2:
        raise ValueError("Too few independent training prompts for LOBO.")
    training, validation = grouped_split(pool, seed=seed)
    for frame in (training, validation):
        if frame["source"].eq(held_out).any() or set(frame["prompt_group"]) & test_prompts:
            raise AssertionError("Held-out responses or prompts leaked into training or validation.")
    return training, validation, test, len(candidates) - len(pool)


def aggregate_pair_scores(pair_results):
    results = {}
    for name, part in pair_results.groupby("predictor", sort=False):
        results[name] = {
            "pair_brier": float(part["brier"].mean()),
            "benchmark_mean_brier": float(part.groupby("held_out")["brier"].mean().mean()),
            "row_brier": float(np.average(part["brier"], weights=part["rows"])),
            "row_accuracy": float(np.average(part["accuracy"], weights=part["rows"])),
            "pairs": len(part), "rows": int(part["rows"].sum()),
            "benchmarks": part["held_out"].nunique(),
        }
    return results


def make_team_summary(summary, fold_scores):
    fnn = summary["aggregate"]["FNN"]
    baseline = summary["aggregate"]["Training mean"]
    constant = summary["aggregate"]["Constant 0.5"]
    model = summary["model"]
    hidden = " and ".join(str(units) for units in model["hidden"])
    loss = "Brier loss" if model["loss"] == "brier" else "binary cross-entropy"
    comparison = fold_scores.pivot(index="held_out", columns="predictor", values="pair_brier")
    wins = int((comparison["FNN"] < comparison["Training mean"]).sum())
    if fnn["pair_brier"] > max(baseline["pair_brier"], constant["pair_brier"]):
        finding = "The current model seems to learn the training benchmarks better than it transfers to new ones."
    else:
        finding = "This is a first comparison using one seed, so more runs would help check how consistent the results are."
    return (
        "I ran LOBO on my SBERT FNN across the four binary benchmarks, holding out one benchmark "
        "at a time and training from scratch on the others. I kept the task embeddings, model metadata, "
        f"and text statistics, with hidden layers of {hidden} neurons, {loss}, {model['dropout']:.0%} dropout, "
        f"and L2 regularization ({model['l2']:g}). Early stopping used a separate validation split.\n\n"
        f"The overall equal-pair Brier was {fnn['pair_brier']:.4f}, compared with "
        f"{baseline['pair_brier']:.4f} for a training-average baseline and 0.2500 for constant 0.5 "
        f"predictions. The FNN beat the training-average baseline on {wins} of {len(comparison)} "
        f"benchmarks. The mean across benchmarks was {fnn['benchmark_mean_brier']:.4f}. {finding} "
        "These are local results, not CodaBench scores.\n"
    )


def plot_lobo_results(run_dir, fold_scores, histories, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    sources = sorted(histories)
    labels = [BENCHMARK_NAMES.get(source, source) for source in sources]
    fig, axis = plt.subplots(figsize=(10, 4.5), layout="constrained")
    positions = np.arange(len(sources))
    for offset, (name, color) in enumerate(zip(PREDICTORS, ("#147d92", "#bc6244", "#737373"))):
        values = fold_scores.loc[fold_scores["predictor"].eq(name)].set_index("held_out").loc[sources, "pair_brier"]
        bars = axis.bar(positions + (offset - 1) * 0.24, values, width=0.23, label=name, color=color)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axis.set_xticks(positions, labels)
    axis.set(title="LOBO: prediction error on each unseen benchmark", ylabel="Equal-pair Brier score; lower is better",
             ylim=(0, float(fold_scores["pair_brier"].max()) * 1.25))
    axis.legend(loc="upper left", ncols=3)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    fig.savefig(run_dir / "lobo_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for axis, source, label in zip(axes.flat, sources, labels):
        history = histories[source]
        best = int(history.loc[history["val_pair_brier"].idxmin(), "epoch"])
        axis.plot(history["epoch"], history["brier_score"], label="training")
        axis.plot(history["epoch"], history["val_pair_brier"], label="inner validation")
        axis.axvline(best, color="gray", linestyle=":", label=f"selected epoch {best}")
        axis.set(title=f"Held out: {label}", xlabel="Epoch", ylabel="Equal-pair Brier")
        axis.set_xticks(history["epoch"].iloc[::max(1, len(history) // 6)].astype(int))
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    fig.savefig(run_dir / "lobo_learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    for axis, source, label in zip(axes.flat, sources, labels):
        part = predictions.loc[predictions["source"].eq(source)]
        observed, predicted = calibration_curve(part["response"], part["fnn_probability"], n_bins=10)
        axis.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        axis.plot(predicted, observed, marker="o", color="#147d92", label="FNN")
        axis.set(title=label, xlabel="Predicted probability", ylabel="Observed success rate",
                 xlim=(0, 1), ylim=(0, 1))
        axis.legend(fontsize=8)
    fig.suptitle("LOBO calibration on held-out responses (response-weighted)")
    fig.savefig(run_dir / "lobo_calibration.png", dpi=150)
    plt.close(fig)


def run_lobo(data_dir, run_name, epochs=20, seed=42, encoder_dir=None):
    start = time.perf_counter()
    config = json.loads((SUBMISSION_DIR / "config.json").read_text())
    if config["method"]["estimator"] != "fnn" or config["feature_mode"] != "sbert":
        raise ValueError("LOBO expects the current SBERT FNN configuration.")
    if config["adaptation_precision"] is not None or config["encoder_max_seq_length"] != 256:
        raise ValueError("This evaluation uses the current nonadaptive, 256-token FNN recipe.")
    method = Method(**dict(config["method"], hidden=tuple(config["method"]["hidden"])))
    protected_paths = [*SUBMISSION_DIR.glob("*"), HERE / "submission_fnn.zip", *HERE.glob("results/fnn_*")]
    protected = {str(path.relative_to(HERE)): file_sha256(path) for path in protected_paths if path.is_file()}
    run_dir = Path(data_dir) / "approach1_outputs" / "experiment_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    examples, inventory = load_examples(data_dir)
    save_json(run_dir / "data_manifest.json", inventory)
    sources = sorted(examples["source"].unique())
    if len(sources) < 2:
        raise ValueError("LOBO needs at least two benchmarks.")
    encoder_dir = find_encoder(data_dir, encoder_dir)
    vectors, encoder_source = cached_embeddings(
        examples, data_dir, encoder_dir, Path(data_dir) / "approach1_cache" / "experiment_cache",
    )
    features = build_features(examples, "sbert", vectors)
    if features.shape[1] != config["input_dimensions"]:
        raise ValueError("LOBO features do not match the current FNN input dimensions.")
    save_json(run_dir / "encoder_source.json", encoder_source)
    source_dir = run_dir / "source"
    source_dir.mkdir()
    for path in (Path(__file__), HERE / "experiments.py", SUBMISSION_DIR / "features.py", SUBMISSION_DIR / "config.json"):
        shutil.copy2(path, source_dir / path.name)
    settings = {
        "run": run_name, "method": asdict(method), "seed": seed, "max_epochs": epochs,
        "sources": sources, "input_dimensions": features.shape[1],
        "learning_rate": 0.0005, "batch_size": 512, "inner_validation_fraction": 0.2,
        "early_stopping_patience": 4, "selection_metric": "inner-validation equal-pair Brier",
        "label_policy": "native item-level binary responses only", "adaptation": False,
        "initialization": "fresh FNN weights in every fold; no reuse of the submitted classifier",
        "feature_policy": "fixed pretrained SBERT and stateless metadata/text transforms; no label-fitted preprocessing",
        "baseline_policy": "constant 0.5 and response-average success rate from the fitting subset only",
        "refit_policy": "keep the best inner-validation checkpoint; no refit on validation responses",
        "library_versions": {name: importlib.metadata.version(name) for name in (
            "numpy", "pandas", "scikit-learn", "keras", "torch", "sentence-transformers",
        )},
        "protected_artifacts": protected,
    }
    save_json(run_dir / "settings.json", settings)
    fold_rows, pair_rows, prediction_frames, fold_details, histories = [], [], [], [], {}
    for source in sources:
        training, validation, test, removed = split_lobo(examples, source, seed)
        fold_dir = run_dir / "folds" / source
        fold_dir.mkdir(parents=True)
        print(f"\nHold out {source}: fit={len(training):,}, validation={len(validation):,}, test={len(test):,}", flush=True)
        weights, history, best_epoch, seconds = fit_model(
            method, training, validation, features, epochs, seed, checkpoint_path=fold_dir / "best_fnn.keras",
        )
        probabilities = sigmoid(network_logits(features[test.index], weights))
        np.savez_compressed(fold_dir / "weights.npz", **weights)
        histories[source] = pd.DataFrame(history)
        histories[source].to_csv(fold_dir / "history.csv", index=False)
        save_json(fold_dir / "split_manifest.json", {
            "training_prompts": sorted(training["prompt_group"].unique()),
            "validation_prompts": sorted(validation["prompt_group"].unique()),
            "test_prompts": sorted(test["prompt_group"].unique()),
        })
        import keras
        checkpoint = keras.models.load_model(fold_dir / "best_fnn.keras", compile=False)
        probes = np.linspace(0, len(test) - 1, min(8, len(test))).astype(int)
        restored = checkpoint.predict(features[test.index[probes]], verbose=0).reshape(-1)
        if not np.allclose(restored, probabilities[probes], atol=1e-5):
            raise AssertionError("The saved fold checkpoint differs from the scored FNN.")
        del checkpoint
        keras.backend.clear_session()
        prior = float(training["response"].mean())
        prediction_frames.append(test[[
            "response_id", "source", "benchmark_id", "subject_id", "item_id", "prompt_group", "pair_key", "response",
        ]].assign(fnn_probability=probabilities, training_mean_probability=prior))
        for name, predicted in zip(PREDICTORS, (probabilities, np.full(len(test), prior), np.full(len(test), 0.5))):
            fold_rows.append({"held_out": source, "predictor": name, **scores(test, predicted)})
            errors = test[["pair_key", "response"]].assign(
                squared_error=(predicted - test["response"].to_numpy()) ** 2,
                correct=(predicted >= 0.5) == test["response"].to_numpy(),
            )
            pairs = errors.groupby("pair_key").agg(
                brier=("squared_error", "mean"), accuracy=("correct", "mean"), rows=("response", "size"),
            ).reset_index()
            pair_rows.extend(pairs.assign(held_out=source, predictor=name).to_dict("records"))
        fold_details.append({
            "held_out": source, "training_rows": len(training), "validation_rows": len(validation), "test_rows": len(test),
            "training_prompts": training["prompt_group"].nunique(), "validation_prompts": validation["prompt_group"].nunique(),
            "test_prompts": test["prompt_group"].nunique(), "duplicate_test_prompt_rows_removed_from_pool": removed,
            "best_epoch": best_epoch, "epochs_completed": len(history), "training_seconds": seconds,
            "training_mean_probability": prior, "checkpoint_parity_cases": len(probes),
        })
        pd.DataFrame(fold_rows).to_csv(run_dir / "fold_scores.csv", index=False)
        print(f"Held-out equal-pair Brier: {scores(test, probabilities)['pair_brier']:.6f}; best epoch: {best_epoch}", flush=True)
    fold_scores = pd.DataFrame(fold_rows)
    pair_results = pd.DataFrame(pair_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if len(predictions) != len(examples) or predictions.duplicated(["source", "response_id"]).any():
        raise AssertionError("Every eligible response must be tested exactly once.")
    pair_results.to_csv(run_dir / "pair_scores.csv", index=False)
    predictions.to_parquet(run_dir / "test_predictions.parquet", index=False)
    plot_lobo_results(run_dir, fold_scores, histories, predictions)
    for relative, digest in protected.items():
        if file_sha256(HERE / relative) != digest:
            raise AssertionError(f"The original model or result was modified: {relative}")
    summary = {
        "run": run_name, "seed": seed, "model": asdict(method), "folds": fold_details,
        "aggregate": aggregate_pair_scores(pair_results), "total_seconds": time.perf_counter() - start,
        "training_seconds": sum(fold["training_seconds"] for fold in fold_details),
        "original_submission_unchanged": True, "run_directory": str(run_dir.resolve()),
        "aggregation": "pair_brier gives every model-benchmark pair equal weight; benchmark_mean_brier averages the four fold scores equally",
        "limitations": "One seed, four public binary benchmarks, no revealed-label adaptation. Public LOBO test results, not a CodaBench score.",
    }
    save_json(run_dir / "summary.json", summary)
    (run_dir / "team_summary.md").write_text(make_team_summary(summary, fold_scores), encoding="utf-8")
    report_dir = HERE / "results" / "lobo"
    report_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fold_scores.csv", "pair_scores.csv", "summary.json", "team_summary.md", "lobo_comparison.png",
                 "lobo_learning_curves.png", "lobo_calibration.png"):
        shutil.copy2(run_dir / name, report_dir / name)
    for source in sources:
        histories[source].to_csv(report_dir / f"history_{source}.csv", index=False)
    print(fold_scores.pivot(index="held_out", columns="predictor", values="pair_brier").to_string(), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-dir", type=Path)
    args = parser.parse_args()
    run_lobo(args.data_dir, args.run_name, args.epochs, args.seed, args.encoder_dir)


if __name__ == "__main__":
    main()
