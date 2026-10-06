"""PSU-grouped splits for the four v30 evaluation roles (T, P, LC, R).

Every split is a set of row indices into the cohort frame (full-cohort order).
Seeds derive from the split id so reruns reproduce the same partition. The
nutrition arm filters the same indices by participation; it never re-splits.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .v30_common import PERIODS
from .v30_data import GROUPS6

INNER_VALIDATION_SHARE = 0.20
OUTER_FOLDS = 5
LC_FRACTIONS = (0.25, 0.50, 0.75)
MIN_FIT_PER_GROUP = 10
MIN_VALIDATION_PER_GROUP = 3


def split_seed(split_id: str, salt: str = "") -> int:
    return int(hashlib.sha256((split_id + "|" + salt).encode()).hexdigest()[:8], 16)


@dataclass
class Split:
    split_id: str
    family: str
    fit: np.ndarray
    validation: np.ndarray
    heldout: np.ndarray
    history_years: tuple[int, ...]
    heldout_years: tuple[int, ...]
    meta: dict = field(default_factory=dict)

    def roles(self) -> dict[str, np.ndarray]:
        return {"fit": self.fit, "validation": self.validation, "heldout": self.heldout}

    def restrict(self, mask: np.ndarray) -> "Split":
        """Same partition restricted to rows where mask is true (nutrition arm)."""
        keep = lambda idx: idx[mask[idx]]
        return Split(self.split_id, self.family, keep(self.fit), keep(self.validation),
                     keep(self.heldout), self.history_years, self.heldout_years,
                     {**self.meta, "restricted": True})


def _permuted_psus(frame: pd.DataFrame, rows: np.ndarray, year: int, seed: int) -> list[str]:
    psus = sorted(frame.psu_key.iloc[rows][frame.survey_year.iloc[rows].to_numpy() == year].unique())
    rng = np.random.default_rng(seed + year)
    return [psus[i] for i in rng.permutation(len(psus))]


def inner_split(frame: pd.DataFrame, rows: np.ndarray, split_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Per year, 20% of PSUs (rounded, >=1) to validation, the rest to inner fit."""
    seed = split_seed(split_id, "inner")
    years = sorted(frame.survey_year.iloc[rows].unique())
    validation_psus = set()
    for year in years:
        order = _permuted_psus(frame, rows, year, seed)
        k = max(1, int(round(INNER_VALIDATION_SHARE * len(order))))
        validation_psus.update(order[:k])
    in_val = frame.psu_key.iloc[rows].isin(validation_psus).to_numpy()
    return np.sort(rows[~in_val]), np.sort(rows[in_val])


def _years_rows(frame: pd.DataFrame, years) -> np.ndarray:
    return np.flatnonzero(frame.survey_year.isin(list(years)).to_numpy())


def temporal_splits(frame: pd.DataFrame, available: list[int], full_study: bool) -> list[Split]:
    """Expanding windows. Full study: tests 2019-2024 with history from 2015.
    Pilot: every available year with >=2 contiguous available history years."""
    out = []
    years = sorted(available)
    for test in years:
        history = [y for y in years if y < test]
        if full_study:
            if test < 2019 or history != list(range(2015, test)):
                continue
        elif len(history) < 2 or history != list(range(history[0], test)):
            continue
        dev = _years_rows(frame, history)
        fit, val = inner_split(frame, dev, f"T_{test}")
        out.append(Split(f"T_{test}", "T", fit, val, _years_rows(frame, [test]),
                         tuple(history), (test,)))
    return out


def period_splits(frame: pd.DataFrame, available: list[int]) -> tuple[list[Split], dict]:
    out, status = [], {}
    for period, years in PERIODS.items():
        present = [y for y in years if y in available]
        missing = [y for y in years if y not in available]
        if not present:
            status[period] = {"status": "unsupported: data missing", "missing_years": missing}
            continue
        status[period] = {"status": "complete" if not missing else "partial/pilot",
                          "present_years": present, "missing_years": missing}
        rows = _years_rows(frame, present)
        fold_of_psu = {}
        seed = split_seed(period, "outer")
        for year in present:
            order = _permuted_psus(frame, rows, year, seed)
            for rank, psu in enumerate(order):
                fold_of_psu[psu] = rank % OUTER_FOLDS
        fold = frame.psu_key.iloc[rows].map(fold_of_psu).to_numpy()
        for f in range(OUTER_FOLDS):
            held = np.sort(rows[fold == f])
            dev = np.sort(rows[fold != f])
            split_id = f"{period.split('_')[0]}_f{f + 1}"
            fit, val = inner_split(frame, dev, split_id)
            out.append(Split(split_id, "P", fit, val, held, tuple(present), tuple(present),
                             {"period": period, "outer_fold": f + 1}))
    return out, status


def _psu_prefix(frame: pd.DataFrame, rows: np.ndarray, split_id: str, target_rows_by_year: dict) -> np.ndarray:
    """Nested PSU prefix per year (fixed order) until the year's row target is met."""
    seed = split_seed(split_id, "subsample")
    keep = []
    for year, target in target_rows_by_year.items():
        order = _permuted_psus(frame, rows, year, seed)
        year_rows = rows[frame.survey_year.iloc[rows].to_numpy() == year]
        sizes = frame.psu_key.iloc[year_rows].value_counts()
        chosen, total = [], 0
        for psu in order:
            if total >= target:
                break
            chosen.append(psu)
            total += int(sizes[psu])
        keep.append(year_rows[frame.psu_key.iloc[year_rows].isin(chosen).to_numpy()])
    return np.sort(np.concatenate(keep)) if keep else np.zeros(0, dtype=int)


def learning_curve_splits(frame: pd.DataFrame, temporal: list[Split]) -> list[Split]:
    out = []
    for base in temporal:
        years = sorted(frame.survey_year.iloc[base.fit].unique())
        for q in LC_FRACTIONS:
            n_by_year = {int(y): int(math.ceil(q * int((frame.survey_year.iloc[base.fit] == y).sum())))
                         for y in years}
            # the same seed for every fraction keeps subsamples nested
            fit = _psu_prefix(frame, base.fit, base.split_id + "_LC", n_by_year)
            out.append(Split(f"LC_{base.heldout_years[0]}_q{int(q * 100)}", "LC", fit, base.validation,
                             base.heldout, base.history_years, base.heldout_years,
                             {"fraction": q, "base": base.split_id}))
    return out


def recency_splits(frame: pd.DataFrame, temporal: list[Split]) -> list[Split]:
    out = []
    for base in temporal:
        test = base.heldout_years[0]
        recent_rows = _years_rows(frame, [test - 1])
        fit_r, val_r = inner_split(frame, recent_rows, f"R_{test}_recent")
        out.append(Split(f"R_{test}_recent", "R", fit_r, val_r, base.heldout, (test - 1,), (test,),
                         {"base": base.split_id, "design": "recent year only"}))
        fraction = len(fit_r) / len(base.fit)
        years = sorted(frame.survey_year.iloc[base.fit].unique())
        n_by_year = {int(y): int(round(fraction * int((frame.survey_year.iloc[base.fit] == y).sum())))
                     for y in years}
        fit_s = _psu_prefix(frame, base.fit, f"R_{test}_sizematched", n_by_year)
        out.append(Split(f"R_{test}_sizematched", "R", fit_s, base.validation, base.heldout,
                         base.history_years, (test,),
                         {"base": base.split_id, "design": "all history, PSU-subsampled to recent-only fit size",
                          "target_fit_rows": len(fit_r)}))
    return out


def role_summary(frame: pd.DataFrame, rows: np.ndarray, weight_col: str = "wt_itvex") -> dict:
    part = frame.iloc[rows]
    return {"n": int(len(rows)), "psu": int(part.psu_key.nunique()), "households": int(part.hh_key.nunique()),
            "years": {str(k): int(v) for k, v in part.survey_year.value_counts().sort_index().items()},
            "weight_sum": float(part[weight_col].sum()),
            "group6": {GROUPS6[k]: int((part.group6 == k).sum()) for k in range(6)},
            "state16": {str(k): int((part.state16 == k).sum()) for k in range(16)}}


def leakage_check(frame: pd.DataFrame, split: Split) -> dict:
    roles = split.roles()
    result = {}
    names = list(roles)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = set(frame.psu_key.iloc[roles[a]]), set(frame.psu_key.iloc[roles[b]])
            ha, hb = set(frame.hh_key.iloc[roles[a]]), set(frame.hh_key.iloc[roles[b]])
            ra, rb = set(roles[a].tolist()), set(roles[b].tolist())
            result[f"{a}|{b}"] = {"psu_overlap": len(pa & pb), "household_overlap": len(ha & hb),
                                  "row_overlap": len(ra & rb)}
    test_years = set(split.heldout_years)
    dev_years = set(frame.survey_year.iloc[np.r_[split.fit, split.validation]].unique())
    if split.family in ("T", "LC", "R"):
        result["heldout_year_in_development"] = bool(test_years & dev_years)
    if any(v["psu_overlap"] or v["household_overlap"] or v["row_overlap"]
           for k, v in result.items() if isinstance(v, dict)) or result.get("heldout_year_in_development"):
        raise RuntimeError(f"Leakage in split {split.split_id}: {result}")
    return result


def support_status(frame: pd.DataFrame, split: Split) -> dict:
    fit = np.bincount(frame.group6.iloc[split.fit], minlength=6)
    val = np.bincount(frame.group6.iloc[split.validation], minlength=6)
    held = np.bincount(frame.group6.iloc[split.heldout], minlength=6)
    trainable = bool((fit >= MIN_FIT_PER_GROUP).all() and (val >= MIN_VALIDATION_PER_GROUP).all())
    return {"trainable": trainable, "fit_min": int(fit.min()), "validation_min": int(val.min()),
            "heldout_group_counts": held.tolist(),
            "heldout_low_support_groups": [GROUPS6[k] for k in range(6) if held[k] < 10],
            "heldout_unsupported_groups": [GROUPS6[k] for k in range(6) if held[k] == 0]}


def build_all(frame: pd.DataFrame, available: list[int], full_study: bool) -> tuple[list[Split], dict]:
    temporal = temporal_splits(frame, available, full_study)
    period, status = period_splits(frame, available)
    lc = learning_curve_splits(frame, temporal)
    recency = recency_splits(frame, temporal)
    return temporal + period + lc + recency, status
