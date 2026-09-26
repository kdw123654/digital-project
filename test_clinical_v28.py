"""Portable v28 contract checks using synthetic arrays, without study artifacts."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.clinical_v28 import experiment as fx
from models.clinical_v28.clinical_v23_distribution import JointTargetTransform
from models.clinical_v28.synthetic_demo import synthetic_inputs


class PortableV28Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.inputs = synthetic_inputs()
        cls.device = torch.device("cpu")

    def test_all_ten_cells_probability_contract_and_state_reload(self):
        row = self.inputs["roles"]["validation"]
        for cell in fx.CELLS:
            with self.subTest(cell=cell):
                design = cell.split("_", 1)[0]
                model = fx.build(cell, 42, self.inputs, self.device).eval()
                target = row["y"] if design == "binary" else row["z"]
                args = (row["x"], row["sex"], target,
                        self.inputs["item"].target_transform, self.device)
                output = fx.predict(design, model, *args)
                self.assertEqual(output["p4"].shape, (8, 4))
                self.assertEqual(output["p16"].shape, (8, 16))
                self.assertEqual(output["score"].shape, (8,))
                self.assertTrue(all(np.isfinite(value).all() for value in output.values()))
                for name in ("p4", "p16"):
                    self.assertTrue(np.all((output[name] >= 0.) & (output[name] <= 1.)))
                np.testing.assert_allclose(output["p16"].sum(1), 1., atol=1e-10, rtol=0)
                restored = fx.build(cell, 43, self.inputs, self.device).eval()
                restored.load_state_dict(copy.deepcopy(model.state_dict()))
                replay = fx.predict(design, restored, *args)
                for name in output:
                    np.testing.assert_array_equal(output[name], replay[name])

    def test_binary_independent_joint_law(self):
        class ZeroLogits(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(len(x), 4)
        output = fx.predict("binary", ZeroLogits(), np.zeros((3, 114)), None,
                            np.zeros((3, 4)), None, self.device)
        np.testing.assert_allclose(output["p16"], 1. / 16.)
        np.testing.assert_allclose(output["p4"], .5)
        np.testing.assert_allclose(output["score"], np.log(2.))

    def test_invalid_joint_anchor_is_rejected(self):
        inputs = copy.deepcopy(self.inputs)
        inputs["baseline"]["residual_cov"][0, 0] = -1.
        with self.assertRaises(ValueError):
            fx.build("joint_MLP", 42, inputs, self.device)

    def test_invalid_target_measurements_are_rejected(self):
        # SBP must exceed DBP to define positive pulse pressure.
        with self.assertRaises(ValueError):
            JointTargetTransform().fit(np.array([[100., 70., 80., 150., 45.]]), np.ones(1))
        record = self.inputs["item"].target_transform.to_dict()
        record["scale"][0] = 0.
        with self.assertRaises(ValueError):
            JointTargetTransform.from_dict(record)

    def test_two_epoch_training_updates_both_designs(self):
        with patch.multiple(fx, MAX_EPOCHS=2, PATIENCE=2):
            for cell in ("binary_EPF", "joint_EPF"):
                with self.subTest(cell=cell):
                    initial = fx.build(cell, 42, self.inputs, self.device)
                    parameters = dict(initial.named_parameters())
                    predictions, record = fx.train_unit(self.inputs, cell, 1e-3, 1e-3, 42, self.device)
                    self.assertEqual(record["meta"]["epochs_run"], 2)
                    self.assertIn(record["meta"]["best_epoch"], (1, 2))
                    self.assertTrue(any(not torch.equal(value.detach(), record["state_dict"][name])
                                        for name, value in parameters.items() if value.requires_grad))
                    self.assertEqual(set(predictions), {"validation", "calibration"})
                    for output in predictions.values():
                        self.assertTrue(all(np.isfinite(value).all() for value in output.values()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
