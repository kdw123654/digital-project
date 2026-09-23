"""Public contract checks for the prespecified v19 examples; no private data."""
import unittest
from unittest.mock import patch

import numpy as np

from models.predict_clinical_v19 import ClinicalV19Predictor, FEATURES


STATE = {"age": 30, "sex": 1, "HE_BMI": 25.0}


class ClinicalV19PublicContract(unittest.TestCase):
    def test_all_examples_and_raw_encoder_dimensions(self):
        expected = {"EPF_basic_clean_native", "EPF_field_only_basic_clean_native",
                    "MLP_basic_clean_native", "LR_basic_clean", "LR_engineered_clean",
                    "EPF_regression", "EPF_joint"}
        root = ClinicalV19Predictor()
        self.assertEqual(set(root.metadata["models"]), expected)
        for name in expected:
            predictor = ClinicalV19Predictor(name)
            output = predictor.predict(STATE)
            self.assertEqual(predictor.transform([np.nan if STATE.get(k) is None else STATE[k] for k in FEATURES]).shape[1],
                             predictor.input_dim)
            self.assertEqual(len(output["component_probabilities"]), 4)
            self.assertTrue(0 <= output["risk_raw"] <= 1)
            self.assertTrue(0 <= output["risk_adjusted"] <= 1)
            self.assertEqual(set(output["screening"]), {"raw_any", "adjusted_any"})

    def test_one_forward_and_row_isolation(self):
        predictor = ClinicalV19Predictor("EPF_joint")
        with patch.object(predictor.model, "forward", wraps=predictor.model.forward) as called:
            predictor.predict(STATE)
            self.assertEqual(called.call_count, 1)
        raw = np.array([np.nan if STATE.get(k) is None else STATE[k] for k in FEATURES])
        pair = predictor.infer(np.stack((raw, raw)))
        single = predictor.infer(raw)
        np.testing.assert_allclose(pair["components"][0], single["components"][0], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(pair["continuous"][0], single["continuous"][0], rtol=1e-5, atol=1e-5)

    def test_invalid_inputs_and_field_only(self):
        predictor = ClinicalV19Predictor("EPF_field_only_basic_clean_native")
        self.assertIsNone(predictor.model.linear_skip)
        for bad in (STATE | {"HE_glu": 110}, STATE | {"HE_sbp": 130},
                    STATE | {"age": 60}, STATE | {"sex": 9},
                    {"age": 30, "sex": 1}):
            with self.assertRaises(ValueError):
                predictor.predict(bad)


if __name__ == "__main__":
    unittest.main()
