"""Versioned input-corruption views for future experiments only.

v19 artifacts are frozen. In v19, separate WC and WHtR Bernoulli draws were ORed,
so a nominal p became 1-(1-p)^2 for that measurement pair. Here one draw masks
WC and its derived WHtR together at p. Every other random draw keeps the v19
seed/order, isolating the correction to that pair. No targets enter these views.
"""

from __future__ import annotations

import numpy as np


RAW_COLUMNS = (
    "age", "HE_BMI", "HE_wc", "WHtR", "sex", "incm", "edu", "sm_presnt",
    "dr_month", "pa_aerobic", "sedentary_hours", "walking_days", "stress_level",
    "employed", "alcohol_frequency", "alcohol_amount", "strength_days",
    "family_history", "living_alone",
)
PROTECTED = (0, 4)
WAIST_GROUP = (2, 3)
BODY_BLOCK = (1, 2, 3)
BEHAVIOR_BLOCK = (7, 8, 9, 10, 11, 12, 14, 15, 16)


def _raw(raw) -> np.ndarray:
    value = np.asarray(raw, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != len(RAW_COLUMNS) or len(value) == 0 or np.isinf(value).any():
        raise ValueError("Expected nonempty raw [n,19] finite-or-missing input")
    return value


def corrected_mcar(raw, seed: int, rate: float = .2, *, return_mask: bool = False):
    """Mask each independent measurement group at the stated probability.

    Draw one Bernoulli per row for WC and WHtR, and one for each other eligible
    input. Age/sex are protected. ``return_mask`` is for aggregate auditing; the
    per-person mask must not be included in public artifacts.
    """
    source = _raw(raw)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("rate must be a finite probability")
    draws = np.random.default_rng(seed).random(source.shape) < rate
    draws[:, PROTECTED] = False
    draws[:, WAIST_GROUP[1]] = draws[:, WAIST_GROUP[0]]
    result = np.array(source, copy=True)
    result[draws] = np.nan
    return (result, draws) if return_mask else result


def gaussian_fit_sd_noise(raw, fit_indices, *, feature: str, percent_of_fit_sd: int,
                          seed: int, height=None) -> tuple[np.ndarray, dict]:
    """Apply Gaussian *amplitude* to every observed selected measurement.

    ``percent_of_fit_sd=25`` means sigma = 0.25 x fit-role SD, not that 25% of
    people are selected. WC and WHtR stay algebraically linked when WC changes.
    Height is auxiliary view-generation metadata, never an added model input.
    """
    source = _raw(raw)
    if feature not in ("HE_BMI", "HE_wc"):
        raise ValueError("Only BMI or waist measurement noise is supported")
    if percent_of_fit_sd not in (10, 25):
        raise ValueError("Only the prespecified 10/25 percent SD amplitudes are supported")
    fit = np.asarray(fit_indices)
    if fit.ndim != 1 or not np.issubdtype(fit.dtype, np.integer) or len(fit) == 0 or (fit < 0).any() or (fit >= len(source)).any():
        raise ValueError("Invalid fit indices")
    column = RAW_COLUMNS.index(feature)
    fit_sd = float(np.nanstd(source[fit, column]))
    if not np.isfinite(fit_sd) or fit_sd <= 0:
        raise ValueError("Fit measurement SD must be positive")
    if feature == "HE_wc":
        height = np.asarray(height, dtype=np.float64)
        if height.shape != (len(source),) or np.isinf(height).any():
            raise ValueError("One finite-or-missing height is required per row for waist noise")
    scale = fit_sd * percent_of_fit_sd / 100
    # Preserve v19's two-stage multiplication order exactly. Reassociating
    # to ``normal * (fit_sd * amount / 100)`` changes a few low bits and would
    # confound the intended mask-only correction.
    draws = np.random.default_rng(seed).normal(size=len(source)) * fit_sd * percent_of_fit_sd / 100
    result = np.array(source, copy=True)
    result[:, column] = np.maximum(.01, result[:, column] + draws)
    if feature == "HE_wc":
        result[:, 3] = np.where(np.isfinite(source[:, 3]) & (height > 0),
                                result[:, 2] / height, np.nan)
    name = f"{feature}_gaussian_fitSD{percent_of_fit_sd}pct_all_observed"
    metadata = {"name": name, "feature": feature, "seed": int(seed),
                "noise_distribution": "Normal(0, fit_role_SD * percent_of_fit_sd / 100)",
                "percent_of_fit_sd": percent_of_fit_sd, "fit_role_sd": fit_sd,
                "noise_sigma_original_units": scale,
                "target_population": "every originally observed selected measurement",
                "nominal_people_selected_fraction": 1.0,
                "minimum_measurement_value": .01,
                "derived_WHtR_recomputed": feature == "HE_wc",
                "height_or_weight_added_to_model_input": False}
    return result, metadata


def views_for_v20(raw, height, fit_indices) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Build new views without overwriting v19 cached predictions/view labels.

    A clean-trained v19 checkpoint can be evaluated on a corrected stress view;
    paired20-trained checkpoints require fresh training for a corrected fit arm.
    """
    source = _raw(raw)
    views = {"clean": source}
    metadata = {"clean": {"description": "Unmodified raw19 inputs"}}
    for i in range(3):
        name = f"missing20_{i}"
        views[name] = corrected_mcar(source, 20261100 + i, .2)
        metadata[name] = {"nominal_group_mask_probability": .2, "waist_and_WHtR_share_one_draw": True}
    for amount in (10, 30, 40):
        name = f"missing{amount}"
        views[name] = corrected_mcar(source, 20261200 + amount, amount / 100)
        metadata[name] = {"nominal_group_mask_probability": amount / 100,
                          "waist_and_WHtR_share_one_draw": True}
    for name, columns in (("body_missing", BODY_BLOCK), ("behavior_missing", BEHAVIOR_BLOCK)):
        view = np.array(source, copy=True)
        view[:, columns] = np.nan
        views[name] = view
        metadata[name] = {"blocked_columns": [RAW_COLUMNS[index] for index in columns]}
    for feature, column in (("HE_BMI", 1), ("HE_wc", 2)):
        for amount in (10, 25):
            seed = 20261300 + column * 100 + amount
            view, spec = gaussian_fit_sd_noise(source, fit_indices, feature=feature,
                                               percent_of_fit_sd=amount, seed=seed,
                                               height=height if feature == "HE_wc" else None)
            views[spec["name"]] = view
            metadata[spec["name"]] = spec
    return views, metadata
