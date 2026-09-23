"""Synthetic, dependency-light checks of the published v20 view generator."""

import unittest

import numpy as np

from models.clinical_v20_missingness import corrected_mcar, gaussian_fit_sd_noise


class CorrectedMissingnessContract(unittest.TestCase):
    def test_waist_group_uses_one_nominal_draw(self):
        raw = np.full((100_000, 19), 2.0)
        _, mask = corrected_mcar(raw, seed=7, rate=.2, return_mask=True)
        np.testing.assert_array_equal(mask[:, 2], mask[:, 3])
        self.assertFalse(mask[:, [0, 4]].any())  # age and sex
        self.assertLess(abs(mask[:, 2].mean() - .2), .005)
        self.assertLess(abs(mask[:, 1].mean() - .2), .005)
        # Two independent p=.2 draws combined by OR would approach .36.
        self.assertGreater(abs(mask[:, 2].mean() - (1 - (1 - .2) ** 2)), .1)

    def test_natural_missingness_is_preserved(self):
        raw = np.full((40_000, 19), 2.0)
        raw[:10_000, 2:4] = np.nan
        corrected, mask = corrected_mcar(raw, seed=8, rate=.2, return_mask=True)
        self.assertTrue(np.isnan(corrected[:10_000, 2:4]).all())
        np.testing.assert_array_equal(mask[:, 2], mask[:, 3])
        observed = np.isfinite(raw[:, 2])
        newly_missing = observed & np.isnan(corrected[:, 2])
        self.assertLess(abs(newly_missing.sum() / observed.sum() - .2), .007)
        self.assertLess(abs(np.isnan(corrected[:, 2]).mean() - (.25 + .75 * .2)), .007)

    def test_waist_noise_recomputes_derived_whtr(self):
        raw = np.full((1000, 19), 2.0)
        raw[:, 1] = np.linspace(20, 30, len(raw))
        raw[:, 2] = np.linspace(70, 90, len(raw))
        height = np.full(len(raw), 170.0)
        raw[:, 3] = raw[:, 2] / height
        raw[0, 2:4] = np.nan
        waist, metadata = gaussian_fit_sd_noise(
            raw, np.arange(500), feature="HE_wc", percent_of_fit_sd=10,
            seed=6, height=height,
        )
        self.assertTrue(metadata["derived_WHtR_recomputed"])
        self.assertEqual(metadata["nominal_people_selected_fraction"], 1.0)
        self.assertAlmostEqual(metadata["noise_sigma_original_units"],
                               np.nanstd(raw[:500, 2]) * .1)
        self.assertTrue(np.isnan(waist[0, 2:4]).all())
        np.testing.assert_allclose(waist[1:, 3], waist[1:, 2] / height[1:])
        np.testing.assert_array_equal(waist[:, 1], raw[:, 1])
        self.assertGreater(np.count_nonzero(waist[1:, 2] != raw[1:, 2]), 990)


if __name__ == "__main__":
    unittest.main()
