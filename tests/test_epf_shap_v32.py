import numpy as np
from metabolic.epf_shap_math_v32 import shapley_fold




def test_antithetic_shap_recovers_two_feature_interaction_and_additivity():
    raw=np.array([[1.,1.],[2.,3.]])
    def fn(a):return (a[:,0]*a[:,1])[None,:,None]
    permutations=np.array([[[0,1],[1,0],[1,0],[0,1]]])
    background=np.zeros((1,4),dtype=int)
    result=shapley_fold(fn,raw,np.array([1]),permutations,background,chunk_rows=1)
    assert np.allclose(result['phi'][0,0,:,0],[2.,3.])
    assert result['additivity_max_abs_error']<1e-12


def test_same_paths_explain_seed_probability_mean_without_surrogate():
    raw=np.array([[0.,0.],[2.,3.]])
    def fn(a):return np.stack([a[:,0]+a[:,1],2*a[:,0]+a[:,1]])[:,:,None]
    permutations=np.array([[[0,1],[1,0],[1,0],[0,1]]]);background=np.zeros((1,4),int)
    result=shapley_fold(fn,raw,np.array([1]),permutations,background,chunk_rows=1)
    assert np.allclose(result['phi'].mean(0)[0,:,0],[3.,3.])
