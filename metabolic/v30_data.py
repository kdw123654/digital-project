"""v30 raw-data discovery, cohort flow, labels and survey keys.

Year-agnostic: any KNHANES year 2015-2024 whose SAV exists under
``data/raw/knhanes`` is discovered and hashed; missing years are reported,
never assumed. The cohort reproduces the historical ``unaware`` definition of
``metabolic.data.cohort`` with two explicit changes fixed in the v30 protocol:

* pregnancy: confirmed non-pregnant (women ``HE_prg==0``; men ``HE_prg==8``)
  instead of ``HE_dprg`` missing; the historical rule is kept as a sensitivity
  flag (``legacy_pregnancy_rule``).
* unknown diagnosis/medication/pregnancy/fasting status is excluded, never
  coded as negative.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .v30_common import ALL_YEARS, RAW_DIR, sha256_file

TARGETS5 = ("HE_glu", "HE_sbp", "HE_dbp", "HE_TG", "HE_HDL_st2")
FLAGS = ("elevated_glucose", "elevated_bp", "elevated_tg", "low_hdl")
BIT_WEIGHTS = (1, 2, 4, 8)
BITS16 = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.int64)
GROUPS6 = ("normal", "glucose_only", "bp_only", "tg_only", "low_hdl_only", "complex")
GROUPS6_KO = ("정상", "혈당 단독", "혈압 단독", "중성지방 단독", "저HDL 단독", "복합(2개 이상)")
STATE_TO_GROUP6 = np.array([0 if s == 0 else ({1: 1, 2: 2, 4: 3, 8: 4}[s] if s in (1, 2, 4, 8) else 5)
                            for s in range(16)], dtype=np.int64)
M16_TO_6 = np.zeros((16, 6), dtype=np.float64)
M16_TO_6[np.arange(16), STATE_TO_GROUP6] = 1.
DIAGNOSIS = ("DI1_dg", "DI2_dg", "DE1_dg")
MEDICATION_ALLOWED = {"DI1_2": (5, 8), "DI2_2": (5, 8), "DE1_31": (0, 8), "DE1_32": (0, 8)}
SCREEN_COLUMNS = ("ID", "ID_fam", "psu", "kstrata", "wt_itvex", "wt_tot", "age", "sex",
                  "HE_prg", "HE_dprg", "HE_fst", *DIAGNOSIS, *MEDICATION_ALLOWED, *TARGETS5)
AGE_BANDS = (("19", 19, 19), ("20-29", 20, 29), ("30-39", 30, 39))
FASTING_HOURS = 12


def state_names() -> list[str]:
    names = []
    for state in range(16):
        on = [FLAGS[j] for j in range(4) if (state >> j) & 1]
        names.append("none" if not on else "+".join(on))
    return names


def discover_years(raw_dir: Path = RAW_DIR) -> dict[int, Path | None]:
    """Find exactly one SAV per year; names differ in case (HN21_all.sav)."""
    found: dict[int, Path | None] = {}
    candidates = list(Path(raw_dir).rglob("*.sav")) + list(Path(raw_dir).rglob("*.SAV"))
    for year in ALL_YEARS:
        pattern = re.compile(rf"^hn{year % 100:02d}_all\.sav$", re.IGNORECASE)
        matches = sorted({p.resolve() for p in candidates if pattern.match(p.name)})
        if len(matches) > 1:
            raise RuntimeError(f"Multiple SAV files for {year}: {matches}")
        found[year] = matches[0] if matches else None
    return found


def year_source_record(year: int, path: Path) -> dict:
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    return {"year": year, "path": str(path.relative_to(RAW_DIR.parents[2])).replace("\\", "/"),
            "file_name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "rows": int(meta.number_rows), "columns": int(meta.number_columns),
            "column_names": list(meta.column_names)}


def load_year(year: int, path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)
    available = set(meta.column_names)
    alias = {"ID": "id"} if ("ID" not in available and "id" in available) else {}  # 2015: lower-case "id"
    use = None if columns is None else [alias.get(c, c) for c in dict.fromkeys(columns)
                                        if alias.get(c, c) in available]
    frame, _ = pyreadstat.read_sav(str(path), apply_value_formats=False, usecols=use)
    if "ID" not in frame and "id" in frame:
        frame = frame.rename(columns={"id": "ID"})
    if columns is not None:
        for name in columns:
            if name not in frame:
                frame[name] = np.nan
    frame["survey_year"] = int(year)
    frame["psu_key"] = str(year) + ":" + frame["psu"].astype(str)
    frame["hh_key"] = frame["psu_key"] + ":" + frame["ID_fam"].astype(str)
    frame["person_key"] = str(year) + ":" + frame["ID"].astype(str)
    return frame


def age_band(age: pd.Series) -> pd.Series:
    out = pd.Series("other", index=age.index, dtype=object)
    for name, low, high in AGE_BANDS:
        out[age.between(low, high)] = name
    return out


def cohort_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """Sequential v30 eligibility stages (cumulative application)."""
    f = frame
    diag_known = f[list(DIAGNOSIS)].isin((0, 1)).all(axis=1)
    no_diag = f[list(DIAGNOSIS)].eq(0).all(axis=1)
    meds = np.logical_and.reduce([f[c].isin(v) for c, v in MEDICATION_ALLOWED.items()])
    pregnancy = ((f.sex.eq(2) & f.HE_prg.eq(0)) | (f.sex.eq(1) & f.HE_prg.eq(8)))
    fasting = f.HE_fst.ge(FASTING_HOURS)
    complete = f[list(TARGETS5)].notna().all(axis=1) & f.sex.isin((1, 2))
    design = f.wt_itvex.gt(0) & f.psu.notna() & f.kstrata.notna()
    positive = complete & (f[list(TARGETS5)] > 0).all(axis=1) & (f.HE_sbp > f.HE_dbp)
    return [("age_19_39", f.age.between(19, 39)),
            ("diagnosis_status_known", diag_known),
            ("no_prior_diagnosis_unaware", no_diag),
            ("medication_status_known_none", pd.Series(meds, index=f.index)),
            ("confirmed_not_pregnant", pregnancy),
            ("fasting_ge_12h", fasting),
            ("complete_five_measurements", complete),
            ("valid_survey_design", design),
            ("joint_transform_feasible", positive)]


def labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Four flags, 16-state code and exclusive six groups from raw measurements."""
    f = frame
    if f[list(TARGETS5)].isna().any().any() or not f.sex.isin((1, 2)).all():
        raise ValueError("Incomplete measurements or unknown sex cannot be labelled")
    flags = np.column_stack([
        f.HE_glu.ge(100), f.HE_sbp.ge(130) | f.HE_dbp.ge(85), f.HE_TG.ge(150),
        f.HE_HDL_st2.lt(np.where(f.sex.eq(1), 40, 50))]).astype(np.int64)
    state = flags @ np.asarray(BIT_WEIGHTS)
    out = pd.DataFrame(flags, columns=list(FLAGS), index=f.index)
    out["state16"] = state
    out["group6"] = STATE_TO_GROUP6[state]
    return out


def build_cohort(sources: dict[int, Path], extra_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (cohort rows with labels, flow table long format, design units).

    ``design units`` lists every (year, stratum, PSU) in the full survey so the
    PSU bootstrap can resample PSUs with zero eligible respondents.
    """
    frames, flows, units = [], [], []
    columns = list(dict.fromkeys([*SCREEN_COLUMNS, *extra_columns]))
    for year, path in sorted(sources.items()):
        full = load_year(year, path, columns)
        units.append(full[["survey_year", "kstrata", "psu_key"]].dropna().drop_duplicates())
        full["age_band"] = age_band(full.age)
        full["legacy_pregnancy_rule"] = full.HE_dprg.isna()
        keep = pd.Series(True, index=full.index)
        rows = [("all_respondents", keep.copy())]
        for name, mask in cohort_masks(full):
            keep &= mask.fillna(False).astype(bool)
            rows.append((name, keep.copy()))
        for stage_index, (name, mask) in enumerate(rows):
            part = full.loc[mask]
            for (sex, band), count in part.groupby(["sex", "age_band"]).size().items():
                flows.append({"survey_year": year, "stage_index": stage_index, "stage": name,
                              "sex": int(sex) if pd.notna(sex) else -1, "age_band": band,
                              "n": int(count)})
        eligible = full.loc[keep].copy()
        # rows passing the old pregnancy rule but not the explicit one (and vice versa)
        eligible["label_ok"] = True
        frames.append(eligible)
    cohort = pd.concat(frames, ignore_index=True)
    lab = labels(cohort)
    cohort = pd.concat([cohort, lab], axis=1)
    return cohort, pd.DataFrame(flows), pd.concat(units, ignore_index=True)


def legacy_pregnancy_comparison(sources: dict[int, Path]) -> dict:
    """Count people whose eligibility differs between the explicit and legacy pregnancy rules."""
    result = {}
    for year, path in sorted(sources.items()):
        full = load_year(year, path, list(SCREEN_COLUMNS))
        stages = cohort_masks(full)
        base = pd.Series(True, index=full.index)
        for name, mask in stages:
            if name != "confirmed_not_pregnant":
                base &= mask.fillna(False).astype(bool)
        explicit = base & dict(stages)["confirmed_not_pregnant"]
        legacy = base & full.HE_dprg.isna()
        result[str(year)] = {"explicit_rule_n": int(explicit.sum()), "legacy_rule_n": int(legacy.sum()),
                             "explicit_only": int((explicit & ~legacy).sum()),
                             "legacy_only": int((legacy & ~explicit).sum())}
    return result


def key_audit(cohort: pd.DataFrame) -> dict:
    return {"person_key_duplicates": int(cohort.person_key.duplicated().sum()),
            "local_ID_reused_across_years": int(cohort.groupby("ID").survey_year.nunique().gt(1).sum()),
            "psu_keys": int(cohort.psu_key.nunique()), "household_keys": int(cohort.hh_key.nunique()),
            "households_spanning_psu": int(cohort.groupby("hh_key").psu_key.nunique().gt(1).sum()),
            "psu_in_multiple_strata": int(cohort.groupby("psu_key").kstrata.nunique().gt(1).sum()),
            "nonpositive_weights": int((cohort.wt_itvex <= 0).sum())}


def pooled_weight(cohort: pd.DataFrame) -> np.ndarray:
    """wt_itvex divided by the number of pooled survey years (KNHANES pooling rule).

    The constant factor cancels in every weighted metric; it is kept so pooled
    weighted totals estimate an average annual population.
    """
    years = cohort.survey_year.nunique()
    return cohort.wt_itvex.to_numpy(dtype=np.float64) / years
