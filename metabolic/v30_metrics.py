"""Frozen v30 metrics, PSU bootstrap and FDR.

AP is the survey-weighted one-vs-rest average precision (sklearn definition).
A group with no weighted positive or no weighted negative has AP = NA and makes
the 6-group macro AP NA ("insufficient support"); NA is never replaced by 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .v30_data import GROUPS6

ECE_BINS = 15
BOOTSTRAP_SEED = 20260929


class FastAP:
    """Weighted AP for fixed scores under many reweightings (ties grouped as sklearn)."""

    def __init__(self, y, p):
        self.order = np.argsort(-p, kind="mergesort")
        scores = p[self.order]
        self.y = y[self.order].astype(np.float64)
        self.ends = np.r_[np.flatnonzero(np.diff(scores)), len(p) - 1]

    def __call__(self, w):
        weights = w[self.order]
        tp = np.cumsum(weights * self.y)[self.ends]
        total = np.cumsum(weights)[self.ends]
        if tp[-1] <= 0 or total[-1] - tp[-1] <= 0:
            return np.nan
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        return float(np.sum(np.diff(np.r_[0., tp]) * precision) / tp[-1])


def weighted_ap(y, p, w) -> float:
    y = np.asarray(y)
    if (w[y == 1].sum() <= 0) or (w[y == 0].sum() <= 0):
        return np.nan
    return float(average_precision_score(y, p, sample_weight=w))


def weighted_auc(y, p, w) -> float:
    y = np.asarray(y)
    if (w[y == 1].sum() <= 0) or (w[y == 0].sum() <= 0):
        return np.nan
    return float(roc_auc_score(y, p, sample_weight=w))


def ece(y, p, w, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = w.sum()
    value = 0.
    for b in range(bins):
        m = index == b
        if w[m].sum() > 0:
            value += w[m].sum() / total * abs(np.average(y[m], weights=w[m]) - np.average(p[m], weights=w[m]))
    return float(value)


def check_prob(p, k: int = 6) -> np.ndarray:
    p = np.asarray(p, np.float64)
    if p.ndim != 2 or p.shape[1] != k or not np.isfinite(p).all() or (p < -1e-12).any() or \
            not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError(f"Invalid {k}-class probability matrix")
    return np.clip(p, 0, 1)


def six_group_metrics(group, p6, w) -> dict:
    """Primary and secondary metrics for exclusive six-group probabilities."""
    p6 = check_prob(p6)
    group = np.asarray(group, int)
    w = np.asarray(w, np.float64)
    onehot = np.eye(6)[group]
    per = []
    for k in range(6):
        y, p = onehot[:, k], p6[:, k]
        q = np.clip(p, 1e-12, 1 - 1e-12)
        per.append({"group": GROUPS6[k], "n_pos": int(y.sum()), "weighted_prevalence": float(np.average(y, weights=w)),
                    "ap": weighted_ap(y, p, w), "auroc": weighted_auc(y, p, w),
                    "ovr_nll": float(np.average(-(y * np.log(q) + (1 - y) * np.log(1 - q)), weights=w)),
                    "brier": float(np.average((y - p) ** 2, weights=w)), "ece": ece(y, p, w),
                    "low_support": bool(y.sum() < 10)})
    aps = [r["ap"] for r in per]
    macro_ap = float(np.mean(aps)) if all(np.isfinite(aps)) else np.nan
    pred = p6.argmax(1)
    recalls, f1s = [], []
    for k in range(6):
        tp = w[(pred == k) & (group == k)].sum()
        pos = w[group == k].sum()
        predicted = w[pred == k].sum()
        recall = tp / pos if pos > 0 else np.nan
        precision = tp / predicted if predicted > 0 else 0.
        recalls.append(recall)
        f1s.append(0. if (precision + (recall if np.isfinite(recall) else 0)) == 0 else
                   2 * precision * recall / (precision + recall))
    confusion = np.zeros((6, 6))
    confusion_n = np.zeros((6, 6), dtype=int)
    np.add.at(confusion, (group, pred), w)
    np.add.at(confusion_n, (group, pred), 1)
    true_prob = np.clip(p6[np.arange(len(group)), group], 1e-12, 1)
    return {"macro_ap": macro_ap, "macro_ap_status": "ok" if np.isfinite(macro_ap) else "NA: insufficient support",
            "per_group": per, "macro_auroc": float(np.mean([r["auroc"] for r in per])) if all(
                np.isfinite([r["auroc"] for r in per])) else np.nan,
            "macro_f1": float(np.nanmean(f1s)), "balanced_accuracy": float(np.nanmean(recalls)),
            "multiclass_nll": float(np.average(-np.log(true_prob), weights=w)),
            "multiclass_brier": float(np.average(((onehot - p6) ** 2).sum(1), weights=w)),
            "confusion_weighted_rowshare": (confusion / np.maximum(confusion.sum(1, keepdims=True), 1e-300)).tolist(),
            "confusion_counts": confusion_n.tolist(), "n": int(len(group))}


def four_flag_metrics(flags, p4, w) -> dict:
    flags = np.asarray(flags)
    aps = [weighted_ap(flags[:, j], p4[:, j], w) for j in range(4)]
    return {"macro_ap4": float(np.mean(aps)) if all(np.isfinite(aps)) else np.nan, "ap4": aps,
            "auroc4": [weighted_auc(flags[:, j], p4[:, j], w) for j in range(4)]}


def state16_nll(state, p16, w) -> float:
    p = np.clip(np.asarray(p16)[np.arange(len(state)), np.asarray(state, int)], 1e-12, 1)
    return float(np.average(-np.log(p), weights=w))


def prevalence_baseline(group_fit, w_fit, n) -> np.ndarray:
    prior = np.array([w_fit[group_fit == k].sum() for k in range(6)]) / w_fit.sum()
    return np.tile(prior, (n, 1))


# --------------------------------------------------------------------------- bootstrap
class PSUBootstrap:
    """Rao-Wu rescaled PSU bootstrap within (year, stratum); all survey PSUs used.

    Singleton strata are held fixed. Draw multipliers once and reuse them for
    every model so contrasts are paired.
    """

    def __init__(self, units: pd.DataFrame, rows: pd.DataFrame, replicates: int, seed: int = BOOTSTRAP_SEED):
        units = units[units.survey_year.isin(rows.survey_year.unique())].drop_duplicates("psu_key")
        units = units.reset_index(drop=True)
        lookup = {k: i for i, k in enumerate(units.psu_key)}
        strata = [g.index.to_numpy() for _, g in units.groupby(["survey_year", "kstrata"])]
        rng = np.random.default_rng(seed)
        multipliers = np.ones((replicates, len(units)))
        for b in range(replicates):
            for ix in strata:
                n = len(ix)
                if n == 1:
                    continue
                draw = rng.choice(ix, size=n - 1, replace=True)
                counts = np.bincount(draw, minlength=len(units))[ix]
                multipliers[b, ix] = counts * n / (n - 1)
        self.row_unit = np.array([lookup[k] for k in rows.psu_key])
        self.multipliers = multipliers
        self.audit = {"units": len(units), "strata": len(strata),
                      "singleton_strata": int(sum(len(ix) == 1 for ix in strata)),
                      "replicates": replicates, "seed": seed,
                      "method": "Rao-Wu n_h-1 with replacement, multiplicity*n_h/(n_h-1), within (year, stratum), full survey PSUs, no FPC"}

    def weights(self, base_weight: np.ndarray) -> np.ndarray:
        return base_weight[None, :] * self.multipliers[:, self.row_unit]


def macro_ap_draws(group, p6, weight_draws) -> np.ndarray:
    fast = [FastAP((group == k).astype(float), p6[:, k]) for k in range(6)]
    out = np.empty(len(weight_draws))
    for b, w in enumerate(weight_draws):
        values = [f(w) for f in fast]
        out[b] = np.mean(values) if all(np.isfinite(values)) else np.nan
    return out


def per_group_ap_draws(group, p6, weight_draws) -> np.ndarray:
    fast = [FastAP((group == k).astype(float), p6[:, k]) for k in range(6)]
    return np.array([[f(w) for f in fast] for w in weight_draws])


def bootstrap_p(diff_draws: np.ndarray) -> float:
    d = diff_draws[np.isfinite(diff_draws)]
    if not len(d):
        return np.nan
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return float(max(min(p, 1.), 1 / (len(d) + 1)))


def ci(draws: np.ndarray, level: float = .95) -> list[float]:
    d = draws[np.isfinite(draws)]
    if not len(d):
        return [np.nan, np.nan]
    a = (1 - level) / 2
    return np.quantile(d, (a, 1 - a)).tolist()


def bh_fdr(p_values, q: float) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_values, np.float64)
    finite = np.isfinite(p)
    adjusted = np.full(len(p), np.nan)
    rejected = np.zeros(len(p), dtype=bool)
    if finite.any():
        pf = p[finite]
        order = np.argsort(pf)
        m = len(pf)
        ranked = pf[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        adj = np.empty(m)
        adj[order] = np.minimum(ranked, 1)
        adjusted[finite] = adj
        rejected[finite] = adj <= q
    return adjusted, rejected
