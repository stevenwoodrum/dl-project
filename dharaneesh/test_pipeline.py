"""Focused checks for the one retained Approach 1 method."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from .training import HERE, Method, grouped_split, pair_weights, scores, validate_features
from .lobo import split_lobo
from .submission.features import feature_vector, metadata_vector, network_logits, numeric_vector, sigmoid
from .submission.text_encoding import bert_window, select_windows


class FeatureTests(unittest.TestCase):
    def test_feature_dimensions_and_dtype(self):
        vector = feature_vector({}, {"item_content": "Task 12\nreturn x;"}, np.zeros(384))
        self.assertEqual(vector.shape, (644,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertTrue(np.isfinite(vector).all())

    def test_metadata_is_normalized_and_deterministic(self):
        subject = {"normalized_name": "model-a", "provider": "provider-b"}
        vector = metadata_vector(subject)
        np.testing.assert_array_equal(vector, metadata_vector(subject))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_empty_text_statistics_are_finite(self):
        self.assertTrue(np.isfinite(numeric_vector("")).all())
        self.assertEqual(numeric_vector("").shape, (4,))

    def test_invalid_embeddings_are_rejected(self):
        for vector in (np.zeros(383), np.full(384, np.nan)):
            with self.assertRaises(ValueError):
                feature_vector({}, {}, vector)

    def test_object_float64_nonfinite_and_wrong_shape_are_rejected(self):
        rows = pd.DataFrame({"response": [0, 1]})
        validate_features(rows, np.zeros((2, 644), dtype=np.float32))
        for values in (np.zeros((2, 644), dtype=object), np.zeros((2, 644)),
                       np.full((2, 644), np.inf, dtype=np.float32), np.zeros((2, 643), dtype=np.float32)):
            with self.assertRaises(ValueError):
                validate_features(rows, values)

    def test_nonbinary_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_features(pd.DataFrame({"response": [0.5]}), np.zeros((1, 644), dtype=np.float32))


class WindowTests(unittest.TestCase):
    def test_short_prompt_has_one_unchanged_window(self):
        self.assertEqual(select_windows("hello", [(i, i + 1) for i in range(5)]), [(0, 5)])

    def test_long_prompt_has_at_most_four_bounded_windows(self):
        text = "a" * 2000
        windows = select_windows(text, [(i, i + 1) for i in range(len(text))])
        self.assertEqual(len(windows), 4)
        self.assertEqual(len(set(windows)), 4)
        self.assertTrue(all(0 <= start < end <= len(text) and end - start <= 254 for start, end in windows))

    def test_latex_preamble_is_skipped(self):
        text = "x" * 500 + r"\begin{abstract}" + "content " * 1000
        windows = select_windows(text, [(i, i + 1) for i in range(len(text))])
        self.assertTrue(all(start >= 500 for start, _ in windows))

    def test_special_token_wrapper(self):
        tokenizer = type("Tokenizer", (), {"cls_token_id": 101, "sep_token_id": 102})()
        self.assertEqual(bert_window(tokenizer, [3, 4]), [101, 3, 4, 102])
        tokenizer.sep_token_id = None
        with self.assertRaises(ValueError):
            bert_window(tokenizer, [3])


class EvaluationTests(unittest.TestCase):
    def test_duplicate_prompts_stay_together(self):
        rows = pd.DataFrame({"prompt_group": [str(i // 3) for i in range(60)]})
        train, validation = grouped_split(rows)
        self.assertFalse(set(train["prompt_group"]) & set(validation["prompt_group"]))
        self.assertEqual(len(train) + len(validation), 60)

    def test_pair_weights_have_equal_totals_and_unit_mean(self):
        rows = pd.DataFrame({"pair_key": ["a"] * 9 + ["b"], "response": [0] * 10})
        weights = pair_weights(rows)
        self.assertEqual(weights.dtype, np.float32)
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(float(weights[:9].sum()), float(weights[9]))

    def test_pair_brier_is_not_response_weighted(self):
        rows = pd.DataFrame({"pair_key": ["a", "a", "a", "b"], "response": [0, 0, 0, 1]})
        result = scores(rows, np.zeros(4))
        self.assertAlmostEqual(result["pair_brier"], 0.5)
        self.assertAlmostEqual(result["row_brier"], 0.25)

    def test_lobo_excludes_outer_benchmark_and_duplicate_prompts(self):
        rows = pd.DataFrame({"source": ["outer"] * 2 + ["inner"] * 30,
                             "prompt_group": ["duplicate", "outer-only", "duplicate"] + [str(i) for i in range(29)]})
        train, validation, test, removed = split_lobo(rows, "outer")
        self.assertEqual(removed, 1)
        for subset in (train, validation):
            self.assertNotIn("outer", subset["source"].unique())
            self.assertFalse(set(subset["prompt_group"]) & set(test["prompt_group"]))

    def test_only_the_selected_recipe_is_retained(self):
        self.assertEqual(Method().hidden, (32, 16))
        self.assertEqual(Method().loss, "brier")
        self.assertEqual(Method().estimator, "fnn")
        for filename in ("experiments.py", "optimization.py", "controlled.py"):
            self.assertFalse((HERE / filename).exists())

    def test_fresh_training_models_and_scheduler_are_supported(self):
        from .training import fit_model

        rows = pd.DataFrame({"response": np.tile([0, 1], 32).astype(np.float32),
                             "pair_key": ["a", "b"] * 32})
        features = np.random.default_rng(7).normal(size=(64, 644)).astype(np.float32)
        fitted = []
        for seed in (42, 123):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "best.keras"
                weights, history, epoch, _ = fit_model(
                    Method(), rows.iloc[:48], rows.iloc[48:], features, 1, seed,
                    checkpoint_path=path, record_common_metrics=True, learning_rate_schedule=True,
                )
                self.assertTrue(path.is_file())
                self.assertEqual(epoch, 1)
                self.assertIn("train_pair_brier", history[0])
                self.assertIn("learning_rate", history[0])
                self.assertEqual(weights["W0"].shape, (644, 32))
                fitted.append(weights["W0"])
        self.assertFalse(np.array_equal(*fitted))

    def test_scheduler_cannot_use_missing_validation_data(self):
        from .training import fit_model

        with self.assertRaisesRegex(ValueError, "validation"):
            fit_model(Method(), None, None, None, 1, 42, learning_rate_schedule=True)


class ArtifactTests(unittest.TestCase):
    def test_compiled_checkpoint_matches_exported_weights(self):
        from .build_submission import verify_checkpoint
        self.assertEqual(verify_checkpoint()["parameters"], 21185)

    def test_no_adaptation_or_all_data_claim(self):
        config = json.loads((HERE / "submission/config.json").read_text())
        self.assertIsNone(config["adaptation_precision"])
        self.assertFalse(config["refit_on_all_public"])
        self.assertEqual(config["best_epoch"], 12)
        self.assertEqual(config["training_rows"], 109305)

    def test_saved_training_and_validation_prompts_do_not_overlap(self):
        manifest = json.loads((HERE / "results/validation/split_manifest.json").read_text())
        self.assertFalse(set(manifest["training_prompts"]) & set(manifest["validation_prompts"]))

    def test_predict_is_finite_and_ignores_revealed_labels(self):
        from .submission import model
        weights = {"W0": np.zeros((644, 32), dtype=np.float32), "b0": np.zeros(32, dtype=np.float32),
                   "W1": np.zeros((32, 16), dtype=np.float32), "b1": np.zeros(16, dtype=np.float32),
                   "W2": np.zeros((16, 1), dtype=np.float32), "b2": np.zeros(1, dtype=np.float32)}
        with patch.object(model, "load_bundle", return_value=({}, weights, None)), \
                patch.object(model, "item_embedding", return_value=np.zeros(384, dtype=np.float32)):
            self.assertEqual(model.predict([{}, {}]), 0.5)
            self.assertEqual(model.predict([{}, {}], labeled=["unused"]), 0.5)

    def test_zip_has_root_entrypoint_and_no_training_dependencies(self):
        with zipfile.ZipFile(HERE / "submission_fnn.zip") as archive:
            names = set(archive.namelist())
            self.assertTrue({"model.py", "features.py", "text_encoding.py", "weights.npz", "config.json",
                             "text_encoder/modules.json", "text_encoder/model.safetensors"} <= names)
            self.assertFalse(any("__pycache__" in name for name in names))
            source = archive.read("model.py").decode()
            imports = [node for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))]
            self.assertFalse(any("tensorflow" in ast.dump(node) or "keras" in ast.dump(node) for node in imports))

    def test_lobo_metrics_match_saved_pair_scores(self):
        summary = json.loads((HERE / "results/lobo/summary.json").read_text())
        pairs = pd.read_csv(HERE / "results/lobo/pair_scores.csv")
        self.assertAlmostEqual(pairs.loc[pairs["predictor"].eq("FNN"), "brier"].mean(),
                               summary["aggregate"]["FNN"]["pair_brier"])
        self.assertAlmostEqual(summary["aggregate"]["FNN"]["pair_brier"], 0.26145620706787814)

    def test_zip_hash_matches_manifest(self):
        from .training import file_sha256
        checks = json.loads((HERE / "results/submission_checks.json").read_text())
        self.assertEqual(file_sha256(HERE / "submission_fnn.zip"), checks["zip_sha256"])

    def test_extracted_zip_predicts_from_an_unrelated_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            with zipfile.ZipFile(HERE / "submission_fnn.zip") as archive:
                archive.extractall(bundle)
            script = (
                "import sys,json;sys.path.insert(0,sys.argv[1]);import model;"
                "p=model.predict([{'normalized_name':'example-model'},{'item_content':'Return the sum of two numbers.'}]);"
                "assert 0<=p<=1;assert 'tensorflow' not in sys.modules;assert 'keras' not in sys.modules;"
                "print(json.dumps({'probability':p}))"
            )
            result = subprocess.run([sys.executable, "-I", "-c", script, str(bundle)], cwd=root,
                                    capture_output=True, text=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"probability"', result.stdout)


if __name__ == "__main__":
    unittest.main()
