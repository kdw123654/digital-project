"""Standalone raw38 inference for prespecified fold-1 clinical v22 examples.

The published package contains this file, four local model-source files, fit-only
numeric statistics, seven safe NPZ states, and aggregate calibration metadata.
It never imports the training workspace or reads a survey file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .event_field_v22 import V22MultiTask


HERE = Path(__file__).resolve().parent
OLD_NAMES = (
    "age", "HE_BMI", "HE_wc", "WHtR", "sex", "incm", "edu", "sm_presnt",
    "dr_month", "pa_aerobic", "sedentary_hours", "walking_days", "stress_level",
    "employed", "alcohol_frequency", "alcohol_amount", "strength_days",
    "family_history", "living_alone",
)
NEW_NUMERIC = (
    "sleep_weekday_hours", "sleep_weekend_hours", "vigorous_work_met_min_week",
    "moderate_work_met_min_week", "transport_met_min_week",
    "vigorous_leisure_met_min_week", "moderate_leisure_met_min_week",
    "cigarettes_day", "alcohol_10plus_glasses_tail",
)
NEW_CATEGORICAL = tuple(
    f"family_{disease}_{relative}"
    for disease in ("HP", "HL", "DM")
    for relative in ("father", "mother", "sibling")
) + ("pregnancy_history",)
RAW38_NAMES = OLD_NAMES + NEW_NUMERIC + NEW_CATEGORICAL
COMPONENTS = ("elevated_glucose", "elevated_bp", "elevated_tg", "low_hdl")
BASE_CATEGORIES = ((1, 2), (1, 2, 3, 4), (1, 2, 3, 4), (0, 1), (0, 1), (0, 1))
EXTRA_CATEGORIES = ((1, 2, 3, 4), (0, 1), (0, 1, 2, 3, 4, 5),
                    (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5), (0, 1), (0, 1))
NEW_LEVELS = tuple((0, 1, 8) if name.endswith("sibling") else (0, 1)
                   for name in NEW_CATEGORICAL[:-1]) + ((2, 1, 8),)
PREPROCESS_KEYS = frozenset({
    "base_median", "base_mean", "base_scale", "base_knots",
    "extra_median", "extra_mean", "extra_scale",
    "new_lower", "new_upper", "new_median", "new_mean", "new_scale",
    "new_all_missing",
})
RISK_POLICIES = ("0.9", "0.95")
EXAMPLE_IDS = ("binary4_BCE_EPF", "binary4_BCE_MLP", "full10_BCE_EPF",
               "full10_BCE_MLP", "state16_BCE_EPF", "state16_BCE_MLP", "LR_BCE")
EXAMPLE_SPECS = {
    **{f"{mode}_BCE_{family}": (mode, family, 42)
       for mode in ("binary4", "full10", "state16") for family in ("EPF", "MLP")},
    "LR_BCE": ("binary4", "LR", None),
}
CORE_FILES = ("event_field_v22.py", "event_field_v21.py", "event_field_v19.py",
              "event_plastic_field.py")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _numeric_npz(path: Path, expected: set[str] | frozenset[str] | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if expected is not None and set(archive.files) != set(expected):
            raise ValueError("Public NPZ schema differs from manifest")
        values = {name: archive[name].copy() for name in archive.files}
    if not values or any(array.dtype.kind not in "biuf" or
                         (array.dtype.kind in "uf" and not np.isfinite(array).all())
                         for array in values.values()):
        raise ValueError("Public NPZ must contain only finite numeric or boolean arrays")
    return values


def _sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = 1 / (1 + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    result[~positive] = z / (1 + z)
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    centered = x - np.max(x, axis=1, keepdims=True)
    energy = np.exp(centered)
    return energy / energy.sum(axis=1, keepdims=True)


def _bounded_union(probabilities: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Four valid component probabilities required")
    if not np.isfinite(gamma) or not -1 <= gamma <= 1:
        raise ValueError("Bounded-union gamma is invalid")
    lower, upper = np.max(p, axis=1), np.minimum(np.sum(p, axis=1), 1.)
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = -np.expm1(np.log1p(-p).sum(axis=1))
    q0 = np.clip(q0, lower, upper)
    adjusted = ((1 + gamma) * q0 - gamma * lower if gamma < 0 else
                (1 - gamma) * q0 + gamma * upper)
    return q0, np.clip(adjusted, lower, upper)


def _raw_matrix(raw: Mapping | np.ndarray) -> np.ndarray:
    single_json = isinstance(raw, Mapping)
    if single_json:
        if set(raw) - set(RAW38_NAMES):
            raise ValueError("Only the 38 prespecified non-invasive inputs are accepted")
        row = []
        for name in RAW38_NAMES:
            value = raw.get(name)
            if value is None:
                row.append(np.nan)
            elif isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Input {name} must be numeric or null")
            else:
                number = float(value)
                if not np.isfinite(number):
                    raise ValueError(f"Input {name} must use null for missingness")
                row.append(number)
        result = np.asarray(row, dtype=np.float64)[None, :]
    else:
        result = np.asarray(raw, dtype=np.float64)
        if result.ndim == 1:
            result = result[None, :]
    if result.ndim != 2 or result.shape[1] != 38 or len(result) == 0 or np.isinf(result).any():
        raise ValueError("Expected nonempty [n,38] finite-or-missing raw inputs")
    if not np.isfinite(result[:, 0]).all() or ((result[:, 0] < 19) | (result[:, 0] > 39)).any():
        raise ValueError("Age must be 19–39")
    if not np.isin(result[:, 4], (1, 2)).all():
        raise ValueError("Sex must be 1 or 2")
    if (np.isfinite(result[:, 1:4]) & (result[:, 1:4] <= 0)).any():
        raise ValueError("Available body measurements must be positive")
    if single_json and np.isnan(result[:, 1:4]).all(axis=1).any():
        raise ValueError("At least one positive body measurement is required")
    for index, categories in enumerate(BASE_CATEGORIES, start=4):
        column = result[:, index]
        if not (np.isnan(column) | np.isin(column, categories)).all():
            raise ValueError(f"Unknown base category: {OLD_NAMES[index]}")
    for index, categories in enumerate(EXTRA_CATEGORIES, start=12):
        column = result[:, index]
        if not (np.isnan(column) | np.isin(column, categories)).all():
            raise ValueError(f"Unknown original category: {OLD_NAMES[index]}")
    if (np.isfinite(result[:, 10]) & ((result[:, 10] < 0) | (result[:, 10] > 24))).any():
        raise ValueError("Sedentary hours must be 0–24")
    if (np.isfinite(result[:, 11]) & ~np.isin(result[:, 11], np.arange(8))).any():
        raise ValueError("Walking days must be 0–7")
    for index in (19, 20):
        column = result[:, index]
        if (np.isfinite(column) & ((column <= 0) | (column > 24))).any():
            raise ValueError("Sleep duration must be in (0,24]")
    if (np.isfinite(result[:, 21:26]) & (result[:, 21:26] < 0)).any():
        raise ValueError("Activity MET-minutes cannot be negative")
    smoking = result[:, 26]
    if (np.isfinite(smoking) & ((smoking < 0) | (smoking >= 888))).any():
        raise ValueError("Cigarettes per day are invalid")
    tail = result[:, 27]
    if (np.isfinite(tail) & ~((tail == 0) | ((tail >= 10) & (tail < 888)))).any():
        raise ValueError("Ten-glass alcohol tail is invalid")
    for index, levels in enumerate(NEW_LEVELS, start=28):
        value = result[:, index]
        if not (np.isnan(value) | (value == 9) | np.isin(value, levels)).all():
            raise ValueError(f"Unknown new category: {RAW38_NAMES[index]}")
    history = result[:, 37]
    if (((result[:, 4] == 1) & np.isfinite(history) & (history != 8) & (history != 9)) |
            ((result[:, 4] == 2) & (history == 8))).any():
        raise ValueError("Pregnancy-history structural code disagrees with sex")
    return result


class ClinicalV22Predictor:
    """One model from a sealed fold-1 research example package."""

    def __init__(self, model: str = "binary4_BCE_EPF", *, root: Path | None = None,
                 device: str = "cpu") -> None:
        self.root = Path(root) if root is not None else HERE
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device is unavailable")
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("version") != "clinical_followup_v22_public/1" or
                set(manifest.get("models", {})) != set(EXAMPLE_IDS) or model not in EXAMPLE_IDS):
            raise ValueError("Unknown v22 public example or manifest version")
        if manifest.get("raw38_feature_names") != list(RAW38_NAMES) or manifest.get("encoded_dim") != 114:
            raise ValueError("Public raw/encoded schema changed")
        hashes = json.loads((self.root / "HASHES.json").read_text(encoding="utf-8"))
        required_files = {"__init__.py", "predict.py", *CORE_FILES,
                          "preprocessor.npz", "manifest.json", "PUBLIC_EXPORT_SOURCES.json",
                          *(name + ".npz" for name in EXAMPLE_IDS)}
        if set(hashes) not in (required_files, required_files | {"verification.json"}):
            raise RuntimeError("Published package file list differs from the fixed examples")
        if {path.name for path in self.root.iterdir() if path.is_file()} != set(hashes) | {"HASHES.json"}:
            raise RuntimeError("Published package contains an unexpected file")
        if any(Path(name).name != name or
                             not (self.root / name).is_file() or
                             _sha256(self.root / name) != expected
                             for name, expected in hashes.items()):
            raise RuntimeError("Published package integrity mismatch")
        self.spec, self.name, self.device = manifest["models"][model], model, device
        mode, family, seed = EXAMPLE_SPECS[model]
        if any(self.spec.get(key) != value for key, value in {
            "mode": mode, "family": family, "seed": seed, "fold": 1,
            "arm": "full", "panel": "expanded", "policy": "BCE",
            "state_file": model + ".npz",
        }.items()):
            raise ValueError("Public example identity differs from its fixed contract")
        self.pre = _numeric_npz(self.root / "preprocessor.npz", PREPROCESS_KEYS)
        for key, shape in {
            "base_median": (4,), "base_mean": (4,), "base_scale": (4,),
            "base_knots": (4, 3), "extra_median": (2,), "extra_mean": (2,),
            "extra_scale": (2,), **{f"new_{key}": (9,) for key in
                ("lower", "upper", "median", "mean", "scale", "all_missing")},
        }.items():
            if self.pre[key].shape != shape:
                raise ValueError("Public preprocessor statistic shape changed")
        if any((self.pre[key] <= 0).any() for key in ("base_scale", "extra_scale", "new_scale")):
            raise ValueError("Nonpositive preprocessor scale")
        if not np.isin(self.pre["new_all_missing"], (0, 1)).all():
            raise ValueError("Invalid all-missing fit flags")
        state_file = self.spec.get("state_file")
        if not isinstance(state_file, str) or Path(state_file).name != state_file:
            raise ValueError("Unsafe public state filename")
        state = _numeric_npz(self.root / state_file)
        if self.spec["family"] == "LR":
            if set(state) != {"coef", "bias"} or state["coef"].shape != (4, 114) or state["bias"].shape != (4,):
                raise ValueError("Invalid public LR state")
            self.coef, self.bias, self.network = state["coef"], state["bias"], None
        else:
            options = self.spec["config"]
            self.network = V22MultiTask(
                self.spec["family"], self.spec["mode"], 114, seed=self.spec["seed"],
                **options["architecture"], dropout=options["dropout"],
                feature_norm="fit", normalization="layer", residual_gate="trainable",
                plastic_readout=options["plastic_readout"],
            )
            self.network.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()}, strict=True)
            if (not bool(self.network.core.fit_initialized) or
                    not bool(self.network.anchor_initialized) or
                    not bool(self.network.loss_initialized) or
                    sum(parameter.numel() for parameter in self.network.parameters()) != self.spec["stored_parameters"]):
                raise ValueError("Public model initialization or topology differs")
            self.network.to(device).eval()
        self._validate_calibration()
        if self.spec["mode"] == "full10":
            transform = self.spec.get("target_transform", {})
            mean = np.asarray(transform.get("mean"), dtype=np.float64)
            scale = np.asarray(transform.get("scale"), dtype=np.float64)
            if (mean.shape != (5,) or scale.shape != (5,) or
                    not np.isfinite(mean).all() or not np.isfinite(scale).all() or
                    (scale <= 0).any()):
                raise ValueError("Invalid fit-only continuous target transform")
        expected_risks = ({"raw_any_precal", "raw_any", "adjusted_any", "direct_any_raw", "direct_any_calibrated"}
                          if self.spec["mode"] == "full10" else
                          {"direct_any_raw", "direct_any_calibrated"}
                          if self.spec["mode"] == "state16" else
                          {"raw_any_precal", "raw_any", "adjusted_any"})
        if set(self.spec["thresholds"]) != expected_risks:
            raise ValueError("Public screening cutoff stages changed")
        for stage, policies in self.spec["thresholds"].items():
            if set(policies) != set(RISK_POLICIES) or any(
                not np.isfinite(policies[target]) or not 0 <= policies[target] <= 1
                for target in RISK_POLICIES
            ):
                raise ValueError(f"Invalid development threshold: {stage}")

    def _validate_calibration(self) -> None:
        calibration = self.spec.get("calibration", {})
        if self.spec["mode"] == "state16":
            if set(calibration) != {"joint_temperature"}:
                raise ValueError("State16 requires joint temperature only")
            item = calibration["joint_temperature"]
            if (item.get("method") != "one_positive_joint_temperature_v22" or
                    not np.isfinite(item.get("temperature", np.nan)) or
                    not .2 <= item["temperature"] <= 5.):
                raise ValueError("Invalid joint temperature")
        else:
            required = {"component", "direct_union"} if self.spec["mode"] == "full10" else {"component"}
            if set(calibration) != required:
                raise ValueError("Unexpected calibration heads")
            item = calibration["component"]
            coef = np.asarray(item.get("component_coefficients"), dtype=np.float64)
            if (item.get("method") != "v18_monotone_components_then_frechet_bounded_union_weighted_bce" or
                    coef.shape != (4, 2) or not np.isfinite(coef).all() or
                    ((coef[:, 0] < 0) | (coef[:, 0] > 20)).any() or
                    (np.abs(coef[:, 1]) > 20).any() or
                    not np.isfinite(item.get("gamma", np.nan)) or
                    not -1 <= item["gamma"] <= 1):
                raise ValueError("Invalid component calibration")
            if "direct_union" in calibration:
                union = calibration["direct_union"]
                if (union.get("method") != "bounded_positive_slope_union_logit_v22" or
                        not np.isfinite(union.get("slope", np.nan)) or
                        not np.isfinite(union.get("bias", np.nan)) or
                        not 0 <= union["slope"] <= 20 or not -20 <= union["bias"] <= 20):
                    raise ValueError("Invalid direct-union calibration")

    def transform(self, raw: Mapping | np.ndarray) -> np.ndarray:
        x = _raw_matrix(raw)
        base = self.pre
        initial = x[:, :4]
        missing = np.isnan(initial)
        z = (np.where(missing, base["base_median"], initial) - base["base_mean"]) / base["base_scale"]
        blocks: list[np.ndarray] = [z, missing.astype(np.float64)]
        for index, levels in enumerate(BASE_CATEGORIES, start=4):
            column = x[:, index]
            blocks.extend(((column[:, None] == np.asarray(levels[1:])).astype(np.float64),
                           np.isnan(column)[:, None].astype(np.float64)))
        blocks.extend((np.maximum(z[:, :, None] - base["base_knots"][None, :, :], 0).reshape(len(x), 12),
                       z * (x[:, 4] == 1)[:, None]))
        extra = x[:, 10:12]
        extra_missing = np.isnan(extra)
        blocks.extend(((np.where(extra_missing, base["extra_median"], extra) -
                        base["extra_mean"]) / base["extra_scale"],
                       extra_missing.astype(np.float64)))
        for index, levels in enumerate(EXTRA_CATEGORIES, start=12):
            column = x[:, index]
            blocks.extend(((column[:, None] == np.asarray(levels[1:])).astype(np.float64),
                           np.isnan(column)[:, None].astype(np.float64)))
        first72 = np.concatenate(blocks, axis=1)
        newest = x[:, 19:28]
        new_missing = np.isnan(newest) | base["new_all_missing"].astype(bool)[None, :]
        clipped = np.clip(newest, base["new_lower"], base["new_upper"])
        filled = np.where(new_missing, base["new_median"], clipped)
        scaled = (filled - base["new_mean"]) / base["new_scale"]
        scaled[:, base["new_all_missing"].astype(bool)] = 0.
        blocks = [first72, scaled, new_missing.astype(np.float64)]
        for index, levels in enumerate(NEW_LEVELS, start=28):
            column = x[:, index].copy()
            column[column == 9] = np.nan
            blocks.extend(((column[:, None] == np.asarray(levels[1:])).astype(np.float64),
                           np.isnan(column)[:, None].astype(np.float64)))
        encoded = np.concatenate(blocks, axis=1).astype(np.float32)
        if encoded.shape != (len(x), 114) or not np.isfinite(encoded).all():
            raise ValueError("Public engineered114 encoder output is invalid")
        return encoded

    def infer(self, raw: Mapping | np.ndarray) -> dict[str, np.ndarray | dict]:
        encoded = self.transform(raw)
        if self.network is None:
            outputs = {"binary_logits": encoded.astype(np.float64) @ self.coef.T + self.bias}
        else:
            collected: dict[str, list[np.ndarray]] = {}
            with torch.inference_mode():
                for start in range(0, len(encoded), 256):
                    block = torch.as_tensor(encoded[start:start + 256], device=self.device)
                    parts = self.network.forward_parts(block)
                    for key in ("binary_logits", "aux_z", "union_logit", "state_logits"):
                        value = parts.get(key)
                        if value is not None:
                            collected.setdefault(key, []).append(value.detach().cpu().numpy())
            outputs = {key: np.concatenate(value).astype(np.float64)
                       for key, value in collected.items()}
        result: dict[str, np.ndarray | dict] = {}
        calibration = self.spec["calibration"]
        if "state_logits" in outputs:
            logits = outputs["state_logits"]
            raw_joint = _softmax(logits)
            bits = ((np.arange(16)[:, None] >> np.arange(4)) & 1)
            raw_components = raw_joint @ bits
            joint = _softmax(logits / calibration["joint_temperature"]["temperature"])
            # Match the frozen evaluator: preserve exact marginal ranks, since
            # clipping and converting to logits can create ties at extremes.
            result.update(state_logits=logits, raw_scores=raw_components.copy(),
                          raw_state_probabilities=raw_joint,
                          raw_components=raw_components,
                          direct_any_raw=1 - raw_joint[:, 0],
                          calibrated_state_probabilities=joint,
                          calibrated_components=joint @ bits,
                          direct_any_calibrated=1 - joint[:, 0])
        else:
            logits = outputs["binary_logits"]
            coeff = np.asarray(calibration["component"]["component_coefficients"], dtype=np.float64)
            raw_components = _sigmoid(logits)
            calibrated = _sigmoid(logits * coeff[:, 0] + coeff[:, 1])
            raw_any_precal, _ = _bounded_union(raw_components, 0.)
            raw_any, adjusted = _bounded_union(calibrated, float(calibration["component"]["gamma"]))
            result.update(raw_scores=logits, raw_components=raw_components,
                          raw_any_precal=raw_any_precal, calibrated_components=calibrated,
                          raw_any=raw_any, adjusted_any=adjusted)
            if "union_logit" in outputs:
                direct = outputs["union_logit"]
                union = calibration["direct_union"]
                result.update(direct_union_logit=direct, direct_any_raw=_sigmoid(direct),
                              direct_any_calibrated=_sigmoid(direct * union["slope"] + union["bias"]))
            if "aux_z" in outputs:
                z = outputs["aux_z"]
                mean = np.asarray(self.spec["target_transform"]["mean"], dtype=np.float64)
                scale = np.asarray(self.spec["target_transform"]["scale"], dtype=np.float64)
                original = z * scale + mean
                original[:, 3] = np.expm1(original[:, 3])
                result.update(continuous_z=z, continuous_original=original)
        for key, value in result.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite public inference output: {key}")
        result["screening"] = {
            stage: {policy: result[stage] >= float(cutoff)
                    for policy, cutoff in policies.items()}
            for stage, policies in self.spec["thresholds"].items()
        }
        return result

    def predict(self, state: Mapping) -> dict:
        if not isinstance(state, Mapping):
            raise ValueError("One JSON object with raw38 fields is required")
        result = self.infer(state)
        response = {"version": "clinical_followup_v22_public/1", "model": self.name,
                    "source_fold": 1, "source_seed": self.spec["seed"],
                    "raw_component_probabilities": dict(zip(COMPONENTS, result["raw_components"][0].tolist())),
                    "calibrated_component_probabilities": dict(zip(COMPONENTS, result["calibrated_components"][0].tolist())),
                    "screening": {
                        stage: {policy: {"cutoff": float(self.spec["thresholds"][stage][policy]),
                                         "positive": bool(decision[0])}
                                for policy, decision in rules.items()}
                        for stage, rules in result["screening"].items()},
                    "research_only": True, "not_full_cohort_refit": True,
                    "note": "Threshold-role targets are not held-out guarantees or diagnoses."}
        for name in ("raw_any_precal", "raw_any", "adjusted_any",
                     "direct_any_raw", "direct_any_calibrated"):
            if name in result:
                response[name] = float(result[name][0])
        if "state_logits" in result:
            response["raw_state_probabilities"] = result["raw_state_probabilities"][0].tolist()
            response["calibrated_state_probabilities"] = result["calibrated_state_probabilities"][0].tolist()
            response["state_probability_definition"] = "one temperature-scaled softmax16"
        if "continuous_original" in result:
            response["predicted_measurements"] = dict(zip(
                ("glucose_mg_dL", "systolic_bp_mmHg", "diastolic_bp_mmHg",
                 "triglycerides_mg_dL", "hdl_mg_dL"), result["continuous_original"][0].tolist()))
        return response


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True, help="Local raw38 JSON object")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args(argv)
    torch.set_num_threads(1)
    def reject_constant(value: str):
        raise ValueError(f"Nonstandard JSON numeric constant: {value}")
    state = json.loads(arguments.input.read_text(encoding="utf-8"),
                       parse_constant=reject_constant)
    value = ClinicalV22Predictor(arguments.model, device=arguments.device).predict(state)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
