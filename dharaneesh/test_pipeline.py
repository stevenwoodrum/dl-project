"""Checks for leakage prevention, scoring, adaptation, and portable inference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from .experiments import BUDGETS, budget_scores, grouped_split, pair_weights, scores
from .submission.features import adapt_probability, feature_vector, hash_vector


class PipelineTests(unittest.TestCase):
    def test_duplicate_prompts_stay_in_one_split(self):
        frame = pd.DataFrame({"prompt_group": [str(i // 3) for i in range(60)]})
        train, validation = grouped_split(frame)
        self.assertFalse(set(train["prompt_group"]) & set(validation["prompt_group"]))
        self.assertEqual(len(train) + len(validation), 60)

    def test_pairs_receive_equal_total_training_weight(self):
        frame = pd.DataFrame({"pair_key": ["a"] * 9 + ["b"], "response": [0] * 10})
        weights = pair_weights(frame)
        self.assertAlmostEqual(float(weights[:9].sum()), float(weights[9]))

    def test_pair_metric_is_not_row_metric(self):
        frame = pd.DataFrame({"pair_key": ["a"] * 9 + ["b"], "response": [0] * 10})
        result = scores(frame, np.array([0] * 9 + [1]))
        self.assertAlmostEqual(result["row_brier"], 0.1)
        self.assertAlmostEqual(result["pair_brier"], 0.5)

    def test_lobo_keeps_the_test_benchmark_and_its_prompts_out(self):
        from .lobo import split_lobo

        frame = pd.DataFrame({
            "source": ["held"] * 3 + ["other"] * 20,
            "prompt_group": ["shared", "held-a", "held-b", "shared", *[f"train-{i}" for i in range(19)]],
        })
        training, validation, test, removed = split_lobo(frame, "held")
        self.assertEqual(removed, 1)
        self.assertEqual(len(test), 3)
        self.assertEqual(len(training) + len(validation), 19)
        self.assertFalse(set(training["prompt_group"]) & set(validation["prompt_group"]))
        for part in (training, validation):
            self.assertFalse(part["source"].eq("held").any())
            self.assertFalse(set(part["prompt_group"]) & set(test["prompt_group"]))

    def test_lobo_rejects_an_unknown_benchmark(self):
        from .lobo import split_lobo

        frame = pd.DataFrame({"source": ["a", "b"], "prompt_group": ["one", "two"]})
        with self.assertRaisesRegex(ValueError, "No responses"):
            split_lobo(frame, "missing")

    def test_lobo_distinguishes_pair_benchmark_and_row_averages(self):
        from .lobo import aggregate_pair_scores

        pairs = pd.DataFrame({
            "predictor": ["FNN"] * 3, "held_out": ["a", "a", "b"],
            "brier": [0.1, 0.1, 0.9], "accuracy": [1.0, 1.0, 0.0], "rows": [10, 10, 1],
        })
        result = aggregate_pair_scores(pairs)["FNN"]
        self.assertAlmostEqual(result["pair_brier"], 1.1 / 3)
        self.assertAlmostEqual(result["benchmark_mean_brier"], 0.5)
        self.assertAlmostEqual(result["row_brier"], 2.9 / 21)
        self.assertAlmostEqual(result["row_accuracy"], 20 / 21)

    def test_lobo_summary_uses_the_actual_model_and_scores(self):
        from .lobo import make_team_summary

        summary = {
            "model": {"hidden": [8, 4], "loss": "bce", "dropout": 0.2, "l2": 0.001},
            "aggregate": {
                "FNN": {"pair_brier": 0.3, "benchmark_mean_brier": 0.31},
                "Training mean": {"pair_brier": 0.2}, "Constant 0.5": {"pair_brier": 0.25},
            },
        }
        folds = pd.DataFrame({
            "held_out": ["a", "a"], "predictor": ["FNN", "Training mean"], "pair_brier": [0.3, 0.2],
        })
        text = make_team_summary(summary, folds)
        for expected in ("8 and 4", "binary cross-entropy", "20% dropout", "0.3000", "0.2000", "0 of 1"):
            self.assertIn(expected, text)

    def test_anonymous_ids_cannot_change_features(self):
        subject = {"normalized_name": "model-a", "provider": "organization"}
        a = feature_vector(subject, {"item_content": "Fix a bug", "benchmark_id": "a"}, "hash_text")
        b = feature_vector(dict(subject, subject_id="unseen"),
                           {"item_content": "Fix a bug", "benchmark_id": "b"}, "hash_text")
        np.testing.assert_array_equal(a, b)

    def test_hash_is_stable_across_python_processes(self):
        code = "from dharaneesh.submission.features import hash_vector; print(hash_vector(['model=a'], 8).tolist())"
        result = subprocess.check_output([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent.parent)
        np.testing.assert_array_equal(hash_vector(["model=a"], 8), json.loads(result))

    def test_adaptation_uses_successes_in_the_expected_direction(self):
        self.assertEqual(adapt_probability(0.3, [], [], 4), 0.3)
        self.assertGreater(adapt_probability(0.3, [0.3] * 7, [1] * 7, 4), 0.3)
        self.assertLess(adapt_probability(0.3, [0.3] * 7, [0] * 7, 4), 0.3)

    def test_constant_half_has_alc_quarter(self):
        frame = pd.DataFrame({"pair_key": ["a"] * 100, "prompt_group": [str(i) for i in range(100)],
                              "response": [i % 2 for i in range(100)], "source": ["demo"] * 100})
        result = budget_scores(frame, np.full(100, 0.5), 1.0, None, 42)[0]
        self.assertAlmostEqual(result["brier_alc"], 0.25)
        for budget in BUDGETS:
            self.assertAlmostEqual(result[f"B{budget}"], 0.25)

    def test_export_ignores_evidence_from_other_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            # Export weights into a temporary bundle; ZIP building is exercised by the full run.
            root = Path(directory)
            import shutil
            here = Path(__file__).resolve().parent
            shutil.copy2(here / "submission/model.py", root / "model.py")
            shutil.copy2(here / "submission/features.py", root / "features.py")
            dimension = len(feature_vector({}, {"item_content": "sample"}, "metadata"))
            np.savez(root / "weights.npz", W0=np.zeros((dimension, 1), dtype=np.float32),
                     b0=np.zeros(1, dtype=np.float32))
            (root / "config.json").write_text(json.dumps({"recipe_version": 1,
                "feature_mode": "metadata", "adaptation_precision": 4.0, "shrinkage": 0.75}))
            code = (
                "import model; target=[{'normalized_name':'a'},{'item_content':'sample','benchmark_id':'x'}]; "
                "other=[{'normalized_name':'b'},{'item_content':'sample','benchmark_id':'x'}]; "
                "assert model.predict(target, [[other,1]])==0.5; "
                "assert model.predict(target, [[target,1]])>0.5; "
                "assert isinstance(model.predict(target, []), float)"
            )
            subprocess.run([sys.executable, "-c", code], cwd=root, check=True)

    def test_saved_submission_zip_has_root_entry_point(self):
        from .experiments import package_saved_submission

        with tempfile.TemporaryDirectory() as directory:
            path = package_saved_submission(Path(directory) / "submission.zip")
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertTrue({"model.py", "features.py", "weights.npz", "config.json"}.issubset(names))
                self.assertFalse(any("__pycache__" in name for name in names))

    def test_new_training_run_exports_shared_predictor(self):
        from .experiments import Method, export_bundle, fit_model
        from .submission.features import network_logits, sigmoid

        frame = pd.DataFrame({"pair_key": ["a"] * 20, "response": [0, 1] * 10})
        features = np.stack([feature_vector(
            {"normalized_name": f"model-{index % 2}"}, {"item_content": "sample"}, "metadata",
        ) for index in range(len(frame))])
        method = Method("tiny_fnn", "metadata", "fnn", "brier", hidden=(8, 4), dropout=0.0)
        weights, _, _, _ = fit_model(method, frame, None, features, epochs=1, seed=42)
        config = {"recipe_version": 1, "feature_mode": "metadata", "shrinkage": 1.0,
                  "adaptation_precision": None}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "new_run"
            root.mkdir()
            bundle, archive = export_bundle(root, method, weights, config)
            expected = float(sigmoid(network_logits(features[[0]], weights))[0])
            code = (
                "import model; target=[{'normalized_name':'model-0'},{'item_content':'sample'}]; "
                f"assert abs(model.predict(target, [])-{expected!r})<1e-6"
            )
            subprocess.run([sys.executable, "-c", code], cwd=bundle, check=True)
            self.assertTrue(archive.is_file())

    def test_main_artifact_is_strict_fnn(self):
        directory = Path(__file__).resolve().parent / "submission"
        config = json.loads((directory / "config.json").read_text())
        self.assertEqual(config["method"]["estimator"], "fnn")
        self.assertIsNone(config["adaptation_precision"])
        with np.load(directory / "weights.npz", allow_pickle=False) as weights:
            self.assertEqual(set(weights.files), {"W0", "b0", "W1", "b1", "W2", "b2"})
            self.assertEqual(weights["W0"].shape[1], 64)
            self.assertEqual(weights["W1"].shape, (64, 16))
            self.assertEqual(weights["W2"].shape, (16, 1))

    def test_final_epochs_use_inner_validation_not_outer_test_scores(self):
        from .experiments import refit_epoch_count

        summary = {"folds": [{"best_epoch": epoch} for epoch in [7, 4, 3, 11]],
                   "aggregate": {"FNN": {"pair_brier": 0.1}}}
        self.assertEqual(refit_epoch_count(summary), 6)
        summary["aggregate"]["FNN"]["pair_brier"] = 0.9
        self.assertEqual(refit_epoch_count(summary), 6)

    def test_final_epochs_require_valid_lobo_stopping_epochs(self):
        from .experiments import refit_epoch_count

        for summary in ({"folds": []}, {"folds": [{"best_epoch": 0}]}, {"folds": [{"best_epoch": None}]}):
            with self.assertRaisesRegex(ValueError, "stopping epochs"):
                refit_epoch_count(summary)

    def test_full_data_fit_saves_a_matching_keras_checkpoint(self):
        import keras
        from .experiments import Method, fit_model
        from .submission.features import network_logits, sigmoid

        training = pd.DataFrame({"pair_key": ["a"] * 12, "response": np.array([0, 1] * 6, dtype=np.float32)})
        features = np.arange(48, dtype=np.float32).reshape(12, 4) / 48
        method = Method("tiny_final_fnn", "metadata", "fnn", "brier", hidden=(8, 4), dropout=0.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "final.keras"
            weights, history, epoch, _ = fit_model(
                method, training, None, features, epochs=2, seed=42, fixed_epochs=2, checkpoint_path=checkpoint,
            )
            restored = keras.models.load_model(checkpoint, compile=False)
            np.testing.assert_allclose(restored.predict(features, verbose=0).reshape(-1),
                                       sigmoid(network_logits(features, weights)), atol=1e-5)
            self.assertEqual(len(history), 2)
            self.assertEqual(epoch, 2)
            self.assertFalse(any(key.startswith("val_") for key in history[0]))
            keras.backend.clear_session()

    def test_published_final_checkpoint_matches_submission_weights(self):
        import keras

        directory = Path(__file__).resolve().parent
        summary = json.loads((directory / "results/final/summary.json").read_text())
        config = json.loads((directory / "submission/config.json").read_text())
        self.assertEqual(config, summary["config"])
        self.assertTrue(config["refit_on_all_public"])
        self.assertEqual(summary["validation_rows"], 0)
        checkpoint = keras.models.load_model(directory / "results/final/final_fnn.keras", compile=False)
        dense = [layer for layer in checkpoint.layers if isinstance(layer, keras.layers.Dense)]
        with np.load(directory / "submission/weights.npz", allow_pickle=False) as weights:
            for index, layer in enumerate(dense):
                kernel, bias = layer.get_weights()
                np.testing.assert_array_equal(kernel, weights[f"W{index}"])
                np.testing.assert_array_equal(bias, weights[f"b{index}"])
        keras.backend.clear_session()

    def test_baseline_cannot_be_packaged_as_main_fnn(self):
        from . import experiments

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("model.py", "features.py", "weights.npz"):
                (root / name).write_bytes(b"")
            (root / "config.json").write_text(json.dumps({"method": {"estimator": "logistic"}}))
            with patch.object(experiments, "SUBMISSION_DIR", root):
                with self.assertRaisesRegex(ValueError, "must be an FNN"):
                    experiments.package_saved_submission(root / "should_not_exist.zip")
            self.assertFalse((root / "should_not_exist.zip").exists())


if __name__ == "__main__":
    unittest.main()
