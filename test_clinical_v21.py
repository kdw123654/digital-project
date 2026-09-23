"""Public contract for the five prespecified v21 fold-1 research examples.

Copy beside the published ``models`` package after export verification. A
missing package is a test failure. All examples below are synthetic.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "models/clinical_v21"
EXAMPLES = {
    "enhanced_AP_EPF": ("EPF", "enhanced", "AP", "AP", 42),
    "enhanced_AP_MLP": ("MLP", "enhanced", "AP", "AP", 42),
    "plain_AP_EPF": ("EPF", "plain", "AP", "AP", 42),
    "LR_AP": ("LR", "linear_reference", "AP", "AP", None),
    "LR_BCE": ("LR", "linear_reference", "BCE", "BCE", None),
}
RISK_KINDS = ("raw_any_precal", "raw_any", "adjusted_any")
POLICIES = ("0.9", "0.95")
SYNTHETIC_STATE = {"age": 30, "sex": 1, "HE_BMI": 25.0}


class ClinicalV21PublicContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        sys.path.insert(0, str(HERE))
        cls.portable = importlib.import_module("models.clinical_v21.predict")
        cls.manifest = json.loads((PACKAGE / "manifest.json").read_text(encoding="utf-8"))

    def _raw(self, state=None):
        source = SYNTHETIC_STATE if state is None else state
        return np.asarray([np.nan if source.get(key) is None else source[key]
                           for key in self.portable.FEATURES], dtype=np.float64)

    def test_five_frozen_identities_and_selection_policies(self):
        self.assertEqual(self.manifest["version"], "clinical_condition_v21")
        self.assertTrue(self.manifest["not_full_cohort_refit"])
        self.assertIn("raw pre-calibration four-component macro AP",
                      self.manifest["primary_comparison"])
        self.assertEqual(set(self.manifest["models"]), set(EXAMPLES))
        for name, identity in EXAMPLES.items():
            spec = self.manifest["models"][name]
            self.assertEqual((spec["family"], spec["stage"],
                              spec["config_policy"], spec["checkpoint_policy"],
                              spec["seed"]), identity)
            self.assertEqual(spec["fold"], 1)
            self.assertTrue(spec["state_file"].endswith(".npz"))

    def test_four_components_three_risks_and_frechet_bounds(self):
        for name in EXAMPLES:
            predictor = self.portable.ClinicalV21Predictor(name)
            self.assertEqual(predictor.transform(self._raw()).shape, (1, 72))
            result = predictor.predict(SYNTHETIC_STATE)
            self.assertTrue(result["research_only"])
            self.assertTrue(result["not_full_cohort_refit"])
            self.assertEqual(len(result["raw_component_probabilities"]), 4)
            self.assertEqual(len(result["calibrated_component_probabilities"]), 4)
            self.assertEqual(set(result["screening"]), set(RISK_KINDS))
            for kind in RISK_KINDS:
                self.assertEqual(set(result["screening"][kind]), set(POLICIES))
            batch = predictor.infer(self._raw())
            raw = batch["raw_components"][0]
            calibrated = batch["calibrated_components"][0]
            for components, risk_keys in ((raw, ("raw_any_precal",)),
                                          (calibrated, ("raw_any", "adjusted_any"))):
                lower, upper = components.max(), min(components.sum(), 1.)
                for key in risk_keys:
                    self.assertGreaterEqual(float(batch[key][0]), float(lower) - 1e-12)
                    self.assertLessEqual(float(batch[key][0]), float(upper) + 1e-12)
            self.assertAlmostEqual(float(batch["raw_any_precal"][0]),
                                   float(1 - np.prod(1 - raw)), places=12)
            self.assertAlmostEqual(float(batch["raw_any"][0]),
                                   float(1 - np.prod(1 - calibrated)), places=12)

    def test_single_forward_and_row_isolation(self):
        first = self._raw()
        second = self._raw(SYNTHETIC_STATE | {"age": 31, "HE_BMI": 27.})
        for name in ("enhanced_AP_EPF", "enhanced_AP_MLP", "plain_AP_EPF"):
            predictor = self.portable.ClinicalV21Predictor(name)
            with patch.object(predictor.model, "forward", wraps=predictor.model.forward) as called:
                predictor.predict(SYNTHETIC_STATE)
                self.assertEqual(called.call_count, 1)
            pair = predictor.infer(np.stack((first, second)))
            single = predictor.infer(first)
            for key in ("raw_logits", "raw_components", "calibrated_components",
                        "raw_any_precal", "raw_any", "adjusted_any"):
                np.testing.assert_allclose(pair[key][0], single[key][0], rtol=1e-5, atol=1e-6)

    def test_forbidden_measurements_and_invalid_demographics(self):
        predictor = self.portable.ClinicalV21Predictor("LR_AP")
        for measurement in ("HE_glu", "HE_sbp", "HE_dbp", "HE_TG", "HE_HDL_st2"):
            with self.subTest(measurement=measurement):
                with self.assertRaises(ValueError):
                    predictor.predict(SYNTHETIC_STATE | {measurement: 110})
        for bad in (SYNTHETIC_STATE | {"age": 60},
                    SYNTHETIC_STATE | {"age": 18},
                    SYNTHETIC_STATE | {"sex": 9},
                    {"age": 30, "sex": 1}):
            with self.assertRaises(ValueError):
                predictor.predict(bad)

    def test_module_cli_with_synthetic_json(self):
        with tempfile.TemporaryDirectory(prefix="v21-public-contract-") as temporary:
            input_file = Path(temporary) / "synthetic.json"
            input_file.write_text(json.dumps(SYNTHETIC_STATE), encoding="utf-8")
            env = dict(os.environ, PYTHONPATH=str(HERE))
            command = [sys.executable, "-m", "models.clinical_v21.predict",
                       "--model", "LR_AP", "--input", str(input_file), "--device", "cpu"]
            output = subprocess.run(command, cwd=HERE, env=env, capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(output.returncode, 0, output.stderr)
            payload = json.loads(output.stdout)
            self.assertEqual(payload["model"], "LR_AP")
            self.assertEqual(set(payload["screening"]), set(RISK_KINDS))
            self.assertTrue(payload["research_only"])
            input_file.write_text(json.dumps(SYNTHETIC_STATE | {"HE_glu": 110}),
                                  encoding="utf-8")
            rejected = subprocess.run(command, cwd=HERE, env=env, capture_output=True,
                                      text=True, timeout=60)
            self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
    unittest.main()
