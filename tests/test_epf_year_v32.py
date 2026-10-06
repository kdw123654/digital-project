import numpy as np
import pandas as pd
from metabolic.epf_year_v32 import year_splits, YEARS
from metabolic.peer_comparison_v31 import FEATURES
from metabolic.v30_data import TARGETS5


def test_each_year_is_held_out_once_and_never_in_development():
    rows=[]
    for year in YEARS:
        for psu in range(20):
            for group in range(5):
                rows.append({'survey_year':year,'psu_key':f'{year}:{psu}',
                             'hh_key':f'{year}:{psu}:{group}','group5':group})
    f=pd.DataFrame(rows);splits=year_splits(f)
    assert len(splits)==5
    for s in splits:
        dev=np.r_[s.fit,s.validation]
        assert set(f.survey_year.iloc[dev])==set(YEARS)-set(s.heldout_years)
        assert set(f.survey_year.iloc[s.heldout])==set(s.heldout_years)
        assert not set(f.psu_key.iloc[s.fit]) & set(f.psu_key.iloc[s.validation])
    assert np.array_equal(np.sort(np.concatenate([s.heldout for s in splits])),np.arange(len(f)))


def test_laboratory_targets_are_not_prediction_inputs():
    assert len(FEATURES)==15
    assert not set(FEATURES) & set(TARGETS5)


def test_year_support_failure_is_explicit():
    f=pd.DataFrame([{'survey_year':y,'psu_key':f'{y}:{p}','hh_key':f'{y}:{p}',
                     'group5':0} for y in YEARS for p in range(20)])
    import pytest
    with pytest.raises(ValueError,match='class support'):year_splits(f)
