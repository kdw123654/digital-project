"""Exercise the ten v28 cells on synthetic encoded inputs, without training."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from .clinical_v23_distribution import JointTargetTransform
from . import experiment as fx


def synthetic_inputs(seed: int = 2801) -> dict:
    """Return an in-memory example of the prepared-array contract; no real rows."""
    rng = np.random.default_rng(seed)
    count, fit_n, val_n = 64, 48, 8
    x = rng.normal(size=(count, 114)).astype(np.float32)
    sex = rng.integers(1, 3, count)
    glucose = np.exp(rng.normal(np.log(100.), .13, count))
    dbp = np.exp(rng.normal(np.log(78.), .11, count))
    pulse = np.exp(rng.normal(np.log(45.), .12, count))
    tg = np.exp(rng.normal(np.log(150.), .3, count))
    hdl = np.exp(rng.normal(np.log(46.), .17, count))
    numbers = np.column_stack((glucose, dbp + pulse, dbp, tg, hdl))
    labels = np.column_stack((glucose >= 100., (dbp + pulse >= 130.) | (dbp >= 85.),
                              tg >= 150., hdl < np.where(sex == 1, 40., 50.))).astype(np.float32)
    weights = rng.uniform(.5, 1.5, count)
    transform = JointTargetTransform().fit(numbers[:fit_n], weights[:fit_n])
    z = transform.transform(numbers).astype(np.float32)
    item = SimpleNamespace(fit_x=x[:fit_n], fit_weights=weights[:fit_n], fit_n=fit_n,
                           fit_z=z[:fit_n], target_transform=transform,
                           validation_x=x[fit_n:fit_n + val_n],
                           validation_weights=weights[fit_n:fit_n + val_n])
    roles = {}
    for name, rows in (("validation", slice(fit_n, fit_n + val_n)),
                       ("calibration", slice(fit_n + val_n, count))):
        roles[name] = {"x": x[rows], "z": z[rows], "y": labels[rows],
                       "w": weights[rows], "sex": sex[rows]}
    # These fixed random anchors are synthetic examples, not estimated study anchors.
    baseline = {"coef": rng.normal(0., .01, (5, 114)), "bias": np.zeros(5),
                "residual_cov": .8 * np.eye(5) + .2 * np.ones((5, 5))}
    logit_anchor = (rng.normal(0., .01, (4, 114)), np.zeros(4), [None] * 4)
    return {"item": item, "baseline": baseline, "logit_anchor": logit_anchor,
            "labels_fit": labels[:fit_n], "roles": roles}


def main() -> None:
    torch.set_num_threads(1)
    inputs = synthetic_inputs()
    row = inputs["roles"]["validation"]
    result = {"synthetic_only": True, "trained_model": False, "input_columns": 114,
              "note": "Synthetic initialization and forward pass only; no study performance claim.",
              "cells": {}}
    for cell in fx.CELLS:
        design = cell.split("_", 1)[0]
        model = fx.build(cell, 42, inputs, torch.device("cpu")).eval()
        out = fx.predict(design, model, row["x"], row["sex"],
                         row["y"] if design == "binary" else row["z"],
                         inputs["item"].target_transform, torch.device("cpu"))
        result["cells"][cell] = {
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "p4_shape": list(out["p4"].shape), "p16_shape": list(out["p16"].shape),
            "finite": bool(all(np.isfinite(value).all() for value in out.values())),
            "maximum_state_sum_error": float(np.max(np.abs(out["p16"].sum(1) - 1.))),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
