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

from .training import (
    HERE, Method, find_encoder, fit_model, grouped_split, load_examples, load_features, save_json, scores,
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
        axis.plot(history["epoch"], history["train_pair_brier"], label="training (dropout off)")
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


def evaluate_lobo_folds(examples, features, method, run_dir, epochs=20, seed=42, split_seed=None,
                        training_weighting="pair", record_common_metrics=False, learning_rate_schedule=False):
    """Share identical folds, scoring, and checkpoint checks across FNN recipes."""
    sources = sorted(examples["source"].unique())
    if len(sources) < 2:
        raise ValueError("LOBO needs at least two benchmarks.")
    fold_rows, pair_rows, prediction_frames, fold_details, histories = [], [], [], [], {}
    for source in sources:
        training, validation, test, removed = split_lobo(examples, source, seed if split_seed is None else split_seed)
        fold_dir = run_dir / "folds" / source
        fold_dir.mkdir(parents=True)
        print(f"\nHold out {source}: fit={len(training):,}, validation={len(validation):,}, test={len(test):,}", flush=True)
        weights, history, best_epoch, seconds = fit_model(
            method, training, validation, features, epochs, seed, checkpoint_path=fold_dir / "best_fnn.keras",
            training_weighting=training_weighting, record_common_metrics=record_common_metrics,
            learning_rate_schedule=learning_rate_schedule,
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
    return fold_scores, pair_results, predictions, fold_details, histories


def run_lobo(data_dir, output, encoder_dir=None, seed=42):
    """Reuse the existing outer folds and inner checkpoint selection unchanged."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    examples, inventory = load_examples(data_dir)
    features = load_features(examples, find_encoder(data_dir, encoder_dir), Path(data_dir) / "approach1_cache/combined")
    examples = examples[["source", "response_id", "benchmark_id", "subject_id", "item_id",
                         "prompt_group", "pair_key", "response"]].copy()
    folds, pairs, predictions, details, histories = evaluate_lobo_folds(
        examples, features, Method(), output, seed=seed, split_seed=42,
        record_common_metrics=True, learning_rate_schedule=True,
    )
    summary = {"model": asdict(Method()), "seed": seed, "split_seed": 42,
               "aggregate": aggregate_pair_scores(pairs), "folds": details,
               "limitations": "Local public development LOBO, not a CodaBench score."}
    save_json(output / "summary.json", summary)
    save_json(output / "data_manifest.json", inventory)
    plot_lobo_results(output, folds, histories, predictions)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_lobo(args.data_dir, args.output, args.encoder_dir, args.seed)


if __name__ == "__main__":
    main()
