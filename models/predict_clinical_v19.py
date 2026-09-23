"""Local clinical v19 inference from 19 non-invasive inputs; no network calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .event_field_v19 import EventPlasticFieldV19, MLPControlV19


HERE = Path(__file__).resolve().parent
FEATURES = (
    "age", "HE_BMI", "HE_wc", "WHtR", "sex", "incm", "edu", "sm_presnt",
    "dr_month", "pa_aerobic", "sedentary_hours", "walking_days", "stress_level",
    "employed", "alcohol_frequency", "alcohol_amount", "strength_days",
    "family_history", "living_alone",
)
BASE_CATEGORIES = ((1, 2), (1, 2, 3, 4), (1, 2, 3, 4), (0, 1), (0, 1), (0, 1))
EXTRA_CATEGORIES = ((1, 2, 3, 4), (0, 1), (0, 1, 2, 3, 4, 5),
                    (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5), (0, 1), (0, 1))
COMPONENTS = ("glucose", "bp", "tg", "low_hdl")
MEASUREMENTS = ("glucose_mg_dL", "sbp_mmHg", "dbp_mmHg", "tg_mg_dL", "hdl_mg_dL")


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _validate_raw(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 19 or len(x) == 0 or np.isinf(x).any():
        raise ValueError("Expected nonempty raw [n,19] inputs with finite-or-missing values")
    if not np.isfinite(x[:, 0]).all() or ((x[:, 0] < 19) | (x[:, 0] > 39)).any():
        raise ValueError("Age must be 19-39")
    if not np.isin(x[:, 4], (1, 2)).all():
        raise ValueError("Sex must be 1 or 2")
    body = x[:, 1:4]
    if (np.isfinite(body) & (body <= 0)).any():
        raise ValueError("Available body measurements must be positive")
    for j, categories in enumerate(BASE_CATEGORIES):
        column = x[:, 4 + j]
        if not (np.isnan(column) | np.isin(column, categories)).all():
            raise ValueError("Unknown base category")
    for j, categories in enumerate(EXTRA_CATEGORIES):
        column = x[:, 12 + j]
        if not (np.isnan(column) | np.isin(column, categories)).all():
            raise ValueError("Unknown extra category")
    if ((np.isfinite(x[:, 10])) & ((x[:, 10] < 0) | (x[:, 10] > 24))).any():
        raise ValueError("Sedentary hours must be 0-24")
    if ((np.isfinite(x[:, 11])) & (~np.isin(x[:, 11], np.arange(8)))).any():
        raise ValueError("Walking days must be 0-7")
    return x


def _union_risks(p, gamma):
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid component probabilities")
    lower = np.max(p, axis=1)
    upper = np.minimum(p.sum(axis=1), 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = -np.expm1(np.log1p(-p).sum(axis=1))
    raw = np.clip(raw, lower, upper)
    adjusted = ((1 + gamma) * raw - gamma * lower if gamma < 0
                else (1 - gamma) * raw + gamma * upper)
    return raw, np.clip(adjusted, lower, upper)


class ClinicalV19Predictor:
    def __init__(self, model="EPF_basic_clean_native", device="cpu", root=None):
        self.root = Path(root) if root is not None else HERE / "clinical_v19_checkpoints"
        self.metadata = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.metadata.get("version") != "clinical_factorial_v19" or model not in self.metadata["models"]:
            raise ValueError("Unknown clinical v19 model")
        expected_base = self.metadata.get("public_base_sha256")
        actual_base = hashlib.sha256((HERE / "event_plastic_field.py").read_bytes()).hexdigest()
        if expected_base != actual_base:
            raise RuntimeError("The public event-plastic base changed after export")
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device is unavailable")
        self.name = model
        self.device = device
        self.spec = self.metadata["models"][model]
        self.base = _npz(self.root / "base_preprocessor.npz")
        self.extra = _npz(self.root / "extra_preprocessor.npz")
        self.target = _npz(self.root / "target_transform.npz")
        if str(self.base["basis"]) != "hinge" or not bool(self.extra["expanded"]):
            raise ValueError("Unexpected preprocessor topology")
        calibration = self.spec["calibration"]
        self.coefficients = np.asarray(calibration["component_coefficients"], dtype=np.float64)
        self.gamma = float(calibration["gamma"])
        if (self.coefficients.shape != (4, 2) or not np.isfinite(self.coefficients).all()
                or not np.isfinite(self.gamma) or not -1 <= self.gamma <= 1):
            raise ValueError("Invalid calibration parameters")
        self.mode = self.spec["mode"]
        self.basis = self.spec["basis"]
        self.input_dim = {"basic": 56, "engineered": 72}[self.basis]
        if self.spec["input_dim"] != self.input_dim:
            raise ValueError("Input basis dimension mismatch")
        name = self.spec["state_file"]
        if Path(name).name != name or "/" in name or "\\" in name or not name.endswith(".npz"):
            raise ValueError("Invalid state file name")
        weights = _npz(self.root / name)
        if self.spec["family"] == "LR":
            self.coef = np.asarray(weights["coef"], dtype=np.float64)
            self.bias = np.asarray(weights["bias"], dtype=np.float64)
            if self.coef.shape != (4, self.input_dim) or self.bias.shape != (4,):
                raise ValueError("Invalid linear model shapes")
            self.model = None
        else:
            arch = self.spec["architecture"]
            output_dim = {"classification": 4, "regression": 5, "joint": 9}[self.mode]
            if self.spec["family"] in ("EPF", "EPF_field_only"):
                self.model = EventPlasticFieldV19(
                    self.input_dim, output_dim, dimension=arch["dimension"],
                    steps=arch["steps"], head_width=arch["head_width"],
                    dropout=self.spec["dropout"], feature_norm="fit", seed=self.spec["seed"],
                    initialization=self.spec["initialization"],
                    use_linear_skip=self.spec["family"] == "EPF")
            elif self.spec["family"] == "MLP":
                self.model = MLPControlV19(
                    self.input_dim, output_dim, head_width=arch["head_width"],
                    hidden_layers=arch["hidden_layers"], normalization="layer",
                    dropout=self.spec["dropout"], seed=self.spec["seed"],
                    initialization=self.spec["initialization"])
            else:
                raise ValueError("Unknown model family")
            self.model.load_state_dict({key: torch.as_tensor(value) for key, value in weights.items()}, strict=True)
            self.model.to(device).eval()

    def transform(self, raw):
        x = _validate_raw(raw)
        n = x[:, :4]
        missing = np.isnan(n)
        z = (np.where(missing, self.base["median"], n) - self.base["mean"]) / self.base["scale"]
        blocks = [z, missing.astype(np.float64)]
        for j, categories in enumerate(BASE_CATEGORIES):
            column = x[:, 4 + j]
            blocks.extend(((column[:, None] == np.asarray(categories[1:])).astype(np.float64),
                           np.isnan(column)[:, None].astype(np.float64)))
        blocks.extend((np.maximum(z[:, :, None] - self.base["knots"][None, :, :], 0).reshape(len(x), 12),
                       z * (x[:, 4] == 1)[:, None]))
        extra = x[:, 10:12]
        extra_missing = np.isnan(extra)
        blocks.extend(((np.where(extra_missing, self.extra["median"], extra) - self.extra["mean"]) / self.extra["scale"],
                       extra_missing.astype(np.float64)))
        for j, categories in enumerate(EXTRA_CATEGORIES):
            column = x[:, 12 + j]
            blocks.extend(((column[:, None] == np.asarray(categories[1:])).astype(np.float64),
                           np.isnan(column)[:, None].astype(np.float64)))
        encoded = np.concatenate(blocks, axis=1)
        if encoded.shape[1] != 72:
            raise RuntimeError("Unexpected encoded feature count")
        return encoded[:, np.r_[0:24, 40:72]] if self.basis == "basic" else encoded

    def _scores(self, output, sex):
        if self.mode != "regression":
            return output[:, :4]
        cutoff = np.column_stack((np.full(len(output), 100.), np.full(len(output), 130.),
                                  np.full(len(output), 85.), np.full(len(output), 150.),
                                  np.where(sex == 1, 40., 50.)))
        cutoff[:, 3] = np.log1p(cutoff[:, 3])
        standardized = (cutoff - self.target["mean"]) / self.target["scale"]
        delta = output - standardized
        return np.column_stack((delta[:, 0], np.maximum(delta[:, 1], delta[:, 2]),
                                delta[:, 3], -delta[:, 4]))

    def infer(self, raw):
        original = _validate_raw(raw)
        x = self.transform(original).astype(np.float32)
        if self.model is None:
            output = x @ self.coef.T + self.bias
        else:
            with torch.inference_mode():
                output = self.model(torch.as_tensor(x, device=self.device)).cpu().numpy()
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite model output")
        score = self._scores(output, original[:, 4])
        p = expit(score * self.coefficients[:, 0] + self.coefficients[:, 1])
        raw_any, adjusted_any = _union_risks(p, self.gamma)
        risks = {"components": p, "raw_any": raw_any, "adjusted_any": adjusted_any}
        screening = {kind: {policy: risks[kind] >= float(self.spec["thresholds"][kind][policy])
                            for policy in ("0.9", "0.95")}
                     for kind in ("raw_any", "adjusted_any")}
        continuous = None
        if self.mode != "classification":
            values = output if self.mode == "regression" else output[:, 4:]
            continuous = values * self.target["scale"] + self.target["mean"]
            continuous = np.array(continuous, copy=True)
            continuous[:, 3] = np.expm1(continuous[:, 3])
            if not np.isfinite(continuous).all():
                raise FloatingPointError("Nonfinite measurement estimate")
        return {**risks, "screening": screening, "continuous": continuous}

    def predict(self, state):
        if not isinstance(state, dict) or set(state) - set(FEATURES):
            raise ValueError("Only the 19 documented non-invasive features are accepted")
        if all(state.get(key) is None for key in ("HE_BMI", "HE_wc", "WHtR")):
            raise ValueError("At least one body measurement is required for a patient request")
        raw = [np.nan if state.get(key) is None else state[key] for key in FEATURES]
        values = self.infer(raw)
        result = {
            "version": "clinical_factorial_v19", "model": self.name,
            "component_probabilities": dict(zip(COMPONENTS, values["components"][0].tolist())),
            "risk_raw": float(values["raw_any"][0]),
            "risk_adjusted": float(values["adjusted_any"][0]),
            "screening": {kind: {policy: {"cutoff": float(self.spec["thresholds"][kind][policy]),
                                            "positive": bool(values["screening"][kind][policy][0])}
                                   for policy in ("0.9", "0.95")}
                          for kind in ("raw_any", "adjusted_any")},
            "missing_inputs": [key for key, value in zip(FEATURES, raw) if np.isnan(value)],
            "research_only": True, "source_fold": 1,
            "source_seed": self.spec["seed"],
            "single_shared_field": self.spec["family"] in ("EPF", "EPF_field_only"),
            "external_model_calls": 0,
            "note": "Development 90/95 sensitivity policies are not test guarantees or diagnoses. Continuous estimates are predictions, not measured values.",
        }
        if values["continuous"] is not None:
            result["estimated_measurements"] = dict(zip(MEASUREMENTS, values["continuous"][0].tolist()))
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EPF_basic_clean_native")
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    predictor = ClinicalV19Predictor(arguments.model, arguments.device)
    state = json.loads(Path(arguments.input).read_text(encoding="utf-8"))
    print(json.dumps(predictor.predict(state), ensure_ascii=False, indent=2, allow_nan=False))
