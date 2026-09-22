"""Classical candidate definitions used in both comparisons (sklearn 1.7.1)."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier

def make_estimator(family,trial=0,seed=42):
    if trial not in [0,1]:raise ValueError('trial must be0 or1')
    if family in ['LR_linear','LR_basis']:
        return LogisticRegression(C=[.1,1.][trial],max_iter=3000,random_state=seed)
    if family=='RandomForest':
        return RandomForestClassifier(n_estimators=120,max_depth=[5,None][trial],min_samples_leaf=10,max_features=.8,n_jobs=1,random_state=seed)
    if family=='HistGradientBoosting':
        return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=[7,15][trial],learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=seed)
    raise ValueError(family)

def fit_four(family,x,y,sample_weight,trial=0,seed=42):
    """x is train-fitted encoded input; y has four simultaneous binary labels.

    LR_linear uses columns0..23 and40..71; other models use all72.
    Fit preprocessing/calibration and select trials only on their assigned roles.
    """
    columns=list(range(24))+list(range(40,72)) if family=='LR_linear' else list(range(72))
    models=[make_estimator(family,trial,seed).fit(x[:,columns],y[:,j],sample_weight=sample_weight) for j in range(4)]
    return models,columns
