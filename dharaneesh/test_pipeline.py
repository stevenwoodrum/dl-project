"""Checks for leakage prevention, scoring, adaptation, and portable inference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

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
        from .experiments import METHODS, export_bundle, fit_model
        from .submission.features import network_logits, sigmoid

        frame = pd.DataFrame({"pair_key": ["a"] * 20, "response": [0, 1] * 10})
        features = np.stack([feature_vector(
            {"normalized_name": f"model-{index % 2}"}, {"item_content": "sample"}, "metadata",
        ) for index in range(len(frame))])
        method = METHODS["metadata_logistic"]
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


if __name__ == "__main__":
    unittest.main()
