"""Local v21 research inference for five frozen fold-1 examples."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from .event_field_v21 import EventPlasticFieldV21, MLPControlV21


HERE = Path(__file__).resolve().parent
FEATURES = ("age", "HE_BMI", "HE_wc", "WHtR", "sex", "incm", "edu", "sm_presnt",
            "dr_month", "pa_aerobic", "sedentary_hours", "walking_days", "stress_level",
            "employed", "alcohol_frequency", "alcohol_amount", "strength_days",
            "family_history", "living_alone")
BASE_CATEGORIES = ((1, 2), (1, 2, 3, 4), (1, 2, 3, 4), (0, 1), (0, 1), (0, 1))
EXTRA_CATEGORIES = ((1, 2, 3, 4), (0, 1), (0, 1, 2, 3, 4, 5),
                    (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5), (0, 1), (0, 1))
COMPONENTS = ("glucose", "bp", "tg", "low_hdl")
RISK_KINDS = ("raw_any_precal", "raw_any", "adjusted_any")
POLICIES = ("0.9", "0.95")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if any(value.dtype.kind == "O" for value in arrays.values()):
        raise ValueError("Object array in public state")
    return arrays


def _validate_raw(raw):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 19 or len(x) == 0 or np.isinf(x).any():
        raise ValueError("Expected nonempty [n,19] finite-or-missing raw inputs")
    if not np.isfinite(x[:, 0]).all() or ((x[:, 0] < 19) | (x[:, 0] > 39)).any():
        raise ValueError("Age must be 19-39")
    if not np.isin(x[:, 4], (1, 2)).all():
        raise ValueError("Sex must be 1 or 2")
    if (np.isfinite(x[:, 1:4]) & (x[:, 1:4] <= 0)).any():
        raise ValueError("Available body measurements must be positive")
    for j, categories in enumerate(BASE_CATEGORIES):
        col = x[:, 4 + j]
        if not (np.isnan(col) | np.isin(col, categories)).all():
            raise ValueError("Unknown base category")
    for j, categories in enumerate(EXTRA_CATEGORIES):
        col = x[:, 12 + j]
        if not (np.isnan(col) | np.isin(col, categories)).all():
            raise ValueError("Unknown extra category")
    if ((np.isfinite(x[:, 10])) & ((x[:, 10] < 0) | (x[:, 10] > 24))).any():
        raise ValueError("Sedentary hours must be 0-24")
    if ((np.isfinite(x[:, 11])) & (~np.isin(x[:, 11], np.arange(8)))).any():
        raise ValueError("Walking days must be 0-7")
    return x


def _union(probabilities, gamma):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 4 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid four-label probabilities")
    lower = np.max(p, axis=1)
    upper = np.minimum(p.sum(axis=1), 1.)
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = -np.expm1(np.log1p(-p).sum(axis=1))
    q0 = np.clip(q0, lower, upper)
    adjusted = ((1 + gamma) * q0 - gamma * lower if gamma < 0 else
                (1 - gamma) * q0 + gamma * upper)
    return q0, np.clip(adjusted, lower, upper)


class ClinicalV21Predictor:
    def __init__(self, model="enhanced_AP_EPF", device="cpu", root=None):
        self.root = Path(root) if root is not None else HERE
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != "clinical_condition_v21" or model not in manifest["models"]:
            raise ValueError("Unknown v21 example")
        if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("Requested device is unavailable")
        hashes = json.loads((self.root / "HASHES.json").read_text(encoding="utf-8")) if (self.root / "HASHES.json").exists() else None
        if hashes is not None:
            for name, expected in hashes.items():
                if Path(name).name != name or _sha256(self.root / name) != expected:
                    raise RuntimeError("Published package integrity mismatch")
        self.name, self.spec, self.device = model, manifest["models"][model], device
        self.base = _npz(self.root / "base_preprocessor.npz")
        self.extra = _npz(self.root / "extra_preprocessor.npz")
        if set(self.base) != {"basis", "median", "mean", "scale", "knots"} or set(self.extra) != {"expanded", "median", "mean", "scale"}:
            raise ValueError("Invalid fit preprocessor archive")
        if str(self.base["basis"]) != "hinge" or not bool(self.extra["expanded"]):
            raise ValueError("Expected fit-only engineered encoder")
        if (self.base["median"].shape != (4,) or self.base["mean"].shape != (4,) or
                self.base["scale"].shape != (4,) or self.base["knots"].shape != (4, 3) or
                self.extra["median"].shape != (2,) or self.extra["mean"].shape != (2,) or
                self.extra["scale"].shape != (2,)):
            raise ValueError("Invalid fit preprocessor dimensions")
        if (not np.isfinite(self.base["scale"]).all() or not np.isfinite(self.extra["scale"]).all() or
                (self.base["scale"] <= 0).any() or (self.extra["scale"] <= 0).any()):
            raise ValueError("Invalid fit preprocessor scale")
        calibration = self.spec["calibration"]
        self.coefficients = np.asarray(calibration["component_coefficients"], dtype=np.float64)
        self.gamma = float(calibration["gamma"])
        if (self.coefficients.shape != (4, 2) or not np.isfinite(self.coefficients).all() or
                not np.isfinite(self.gamma) or not -1 <= self.gamma <= 1):
            raise ValueError("Invalid calibration")
        self.thresholds = self.spec["thresholds"]
        if set(self.thresholds) != set(RISK_KINDS) or any(set(self.thresholds[k]) != set(POLICIES) for k in RISK_KINDS):
            raise ValueError("Missing threshold stage or policy")
        if any(not np.isfinite(self.thresholds[k][policy]) or not 0 <= self.thresholds[k][policy] <= 1
               for k in RISK_KINDS for policy in POLICIES):
            raise ValueError("Invalid screening cutoff")
        weights = _npz(self.root / self.spec["state_file"])
        if self.spec["family"] == "LR":
            if set(weights) != {"coef", "bias"}:
                raise ValueError("Unexpected LR state keys")
            self.coef = np.asarray(weights["coef"], dtype=np.float64)
            self.bias = np.asarray(weights["bias"], dtype=np.float64)
            if self.coef.shape != (4, 72) or self.bias.shape != (4,):
                raise ValueError("Invalid LR coefficients")
            self.model = None
        else:
            opts = self.spec["model_options"]
            common = dict(dropout=self.spec["dropout"], seed=self.spec["seed"],
                          initialization=opts["initialization"],
                          use_linear_skip=opts["use_linear_skip"],
                          residual_gate=opts["residual_gate"],
                          numeric_embedding=opts["numeric_embedding"])
            if self.spec["family"] == "EPF":
                self.model = EventPlasticFieldV21(72, 4, **self.spec["architecture"],
                    feature_norm="fit", plasticity=opts["plasticity"],
                    quartic=opts["quartic"], plastic_readout=opts["plastic_readout"],
                    event_mode=opts["event_mode"], **common)
            elif self.spec["family"] == "MLP":
                self.model = MLPControlV21(72, 4, **self.spec["architecture"],
                    normalization="layer", **common)
            else:
                raise ValueError("Unknown model family")
            self.model.load_state_dict({key: torch.as_tensor(value) for key, value in weights.items()}, strict=True)
            if sum(p.numel() for p in self.model.parameters()) != self.spec["stored_parameters"]:
                raise ValueError("Stored parameter count differs from model topology")
            self.model.to(device).eval()

    def transform(self, raw):
        x = _validate_raw(raw)
        n = x[:, :4]
        missing = np.isnan(n)
        z = (np.where(missing, self.base["median"], n) - self.base["mean"]) / self.base["scale"]
        blocks = [z, missing.astype(np.float64)]
        for j, categories in enumerate(BASE_CATEGORIES):
            col = x[:, 4 + j]
            blocks.extend(((col[:, None] == np.asarray(categories[1:])).astype(np.float64),
                           np.isnan(col)[:, None].astype(np.float64)))
        blocks.extend((np.maximum(z[:, :, None] - self.base["knots"][None, :, :], 0).reshape(len(x), 12),
                       z * (x[:, 4] == 1)[:, None]))
        other = x[:, 10:12]
        other_missing = np.isnan(other)
        blocks.extend(((np.where(other_missing, self.extra["median"], other) -
                        self.extra["mean"]) / self.extra["scale"],
                       other_missing.astype(np.float64)))
        for j, categories in enumerate(EXTRA_CATEGORIES):
            col = x[:, 12 + j]
            blocks.extend(((col[:, None] == np.asarray(categories[1:])).astype(np.float64),
                           np.isnan(col)[:, None].astype(np.float64)))
        encoded = np.concatenate(blocks, axis=1)
        if encoded.shape != (len(x), 72) or not np.isfinite(encoded).all():
            raise ValueError("Invalid engineered encoder output")
        return encoded.astype(np.float32)

    def infer(self, raw):
        x = self.transform(raw)
        if self.model is None:
            logits = x.astype(np.float64) @ self.coef.T + self.bias
        else:
            pieces = []
            with torch.inference_mode():
                for start in range(0, len(x), 256):
                    block = torch.as_tensor(x[start:start + 256], device=self.device)
                    pieces.append(self.model(block).detach().cpu().numpy())
            logits = np.concatenate(pieces).astype(np.float64)
        if logits.shape != (len(x), 4) or not np.isfinite(logits).all():
            raise FloatingPointError("Nonfinite or malformed model logits")
        raw_components = expit(logits)
        raw_any_precal, _ = _union(raw_components, 0.)
        calibrated = expit(logits * self.coefficients[:, 0] + self.coefficients[:, 1])
        raw_any, adjusted_any = _union(calibrated, self.gamma)
        risk = {"raw_any_precal": raw_any_precal, "raw_any": raw_any,
                "adjusted_any": adjusted_any}
        screening = {kind: {policy: risk[kind] >= float(self.thresholds[kind][policy])
                            for policy in POLICIES} for kind in RISK_KINDS}
        return {"raw_logits": logits, "raw_components": raw_components,
                "calibrated_components": calibrated, **risk, "screening": screening}

    def predict(self, state):
        if not isinstance(state, dict) or set(state) - set(FEATURES):
            raise ValueError("Only the 19 non-invasive inputs are accepted")
        if all(state.get(key) is None for key in ("HE_BMI", "HE_wc", "WHtR")):
            raise ValueError("At least one body measurement is required")
        raw = [np.nan if state.get(key) is None else state[key] for key in FEATURES]
        values = self.infer(raw)
        return {"version": "clinical_condition_v21", "model": self.name,
                "raw_component_probabilities": dict(zip(COMPONENTS, values["raw_components"][0].tolist())),
                "calibrated_component_probabilities": dict(zip(COMPONENTS, values["calibrated_components"][0].tolist())),
                "risk_raw_precal": float(values["raw_any_precal"][0]),
                "risk_raw_after_component_calibration": float(values["raw_any"][0]),
                "risk_adjusted": float(values["adjusted_any"][0]),
                "screening": {kind: {policy: {"cutoff": float(self.thresholds[kind][policy]),
                                                 "positive": bool(values["screening"][kind][policy][0])}
                                       for policy in POLICIES} for kind in RISK_KINDS},
                "source_fold": 1, "source_seed": self.spec["seed"],
                "not_full_cohort_refit": True, "research_only": True,
                "note": "Development thresholds are not held-out guarantees or diagnoses."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("enhanced_AP_EPF", "enhanced_AP_MLP",
                                             "plain_AP_EPF", "LR_AP", "LR_BCE"),
                        default="enhanced_AP_EPF")
    parser.add_argument("--input", required=True, help="JSON object of the 19 documented inputs")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    state = json.loads(Path(arguments.input).read_text(encoding="utf-8"))
    predictor = ClinicalV21Predictor(arguments.model, arguments.device)
    print(json.dumps(predictor.predict(state), ensure_ascii=False, indent=2, allow_nan=False))
