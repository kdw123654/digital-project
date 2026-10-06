import numpy as np
import pandas as pd
from metabolic.peer_comparison_v31 import M16_TO_5, PeerEncoder, peer_features, FEATURES, BODY_INDICES


def test_five_group_lipid_union_is_not_six_group_complex():
    assert M16_TO_5.shape == (16, 5)
    assert np.array_equal(M16_TO_5.sum(1), np.ones(16))
    assert M16_TO_5[12].argmax() == 3  # TG and HDL together: lipid-only, per peer definition.
    assert M16_TO_5[5].argmax() == 4   # glucose plus TG: two risk domains.
    assert M16_TO_5[1].argmax() == 2 and M16_TO_5[2].argmax() == 1


def _frame():
    return pd.DataFrame({'age':[20.,30.,39.], 'HE_BMI':[21.,25.,29.], 'HE_wc':[70.,85.,100.],
        'HE_ht':[160.,170.,180.], 'sex':[1,2,1], 'incm':[1,2,9], 'edu':[2,3,9],
        'sm_presnt':[0,1,9], 'dr_month':[0,1,9], 'pa_aerobic':[0,1,9], 'cfam':[1,3,9],
        'BE8_1':[5,10,99], 'L_DN_TO':[1,2,3], 'BO1_1':[1,2,3], 'BM1_8':[0,1,9]})


def test_codes_unknown_and_structural_are_not_reversed():
    f=peer_features(_frame())
    assert f.weight_gain.tolist() == [0,0,1]
    assert f.solo_dinner.tolist() == [0,1,2]
    assert f.no_brush_bed.iloc[0] == 1 and f.no_brush_bed.iloc[1] == 0
    assert np.isnan(f.no_brush_bed.iloc[2]) and np.isnan(f.sedentary_hours.iloc[2])
    assert np.isnan(f.living_alone.iloc[2]) and np.isnan(f.incm.iloc[2])


def test_fit_only_encoder_preserves_epf_body_columns():
    raw=peer_features(_frame()); enc=PeerEncoder().fit(raw.iloc[:2],np.ones(2))
    mean=enc.mean.copy(); changed=raw.copy(); changed.loc[2,'age']=1000
    x=enc.transform(changed)
    assert len(FEATURES) == 15 and np.array_equal(enc.mean,mean)
    assert tuple(enc.columns[i] for i in BODY_INDICES) == ('age','HE_BMI','HE_wc','WHtR','sex==2')
    assert x[1,8] == 1 and x[0,8] == 0 and np.isfinite(x).all()
