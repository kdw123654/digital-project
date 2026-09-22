"""Explicit input-completeness/disagreement states, with shrunk local stackers."""
import numpy as np
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from .hybrid_ensembles import _fit_arrays,_probabilities

STATE_NAMES=['complete_agree','complete_disagree','missing_agree','missing_disagree']

class StatewiseStack:
    def __init__(self,C=.1,shrink_people=300,min_positive=20):self.C=C;self.shrink_people=shrink_people;self.min_positive=min_positive
    def _disagreement(self,p):return np.std(logit(np.clip(p,1e-7,1-1e-7)),axis=1)
    def state(self,p,raw):
        p=_probabilities(p,self.experts);raw=np.asarray(raw,float)
        if raw.shape!=(len(p),19) or np.isinf(raw).any():raise ValueError('Invalid raw inputs')
        incomplete=np.isnan(raw[:,[j for j in range(19) if j not in [0,4]]]).any(1)
        disagreement=self._disagreement(p)>self.cutoff
        return 2*incomplete[:,None].astype(int)+disagreement.astype(int)
    def _features(self,p):return (logit(np.clip(p,1e-7,1-1e-7)).reshape(len(p),-1)-self.mean)/self.scale
    def fit(self,p,raw,y,w,person_ids=None):
        p,y,w=_fit_arrays(p,y,w);self.experts=p.shape[1]
        ids=np.arange(len(p)) if person_ids is None else np.asarray(person_ids)
        if ids.shape!=(len(p),):raise ValueError('Wrong participant identifiers')
        disagreement=self._disagreement(p);self.cutoff=np.zeros(4)
        for j in range(4):
            order=np.argsort(disagreement[:,j]);self.cutoff[j]=disagreement[order[min(np.searchsorted(np.cumsum(w[order]),.75*w.sum()),len(w)-1)],j]
        z=logit(np.clip(p,1e-7,1-1e-7)).reshape(len(p),-1);self.mean=np.average(z,axis=0,weights=w);self.scale=np.sqrt(np.average((z-self.mean)**2,axis=0,weights=w));self.scale[self.scale<1e-6]=1
        z=self._features(p);states=self.state(p,raw);self.global_models=[];self.local={};self.support={}
        for j in range(4):
            model=LogisticRegression(C=self.C,max_iter=3000).fit(z,y[:,j],sample_weight=w);self.global_models.append(model)
            for state in range(4):
                mask=states[:,j]==state;people=len(np.unique(ids[mask]));positive=len(np.unique(ids[mask&(y[:,j]==1)]));negative=len(np.unique(ids[mask&(y[:,j]==0)]))
                key=(j,state);self.support[key]={'people':people,'positive':positive,'negative':negative,'active':False,'weight':0.}
                if min(positive,negative)<self.min_positive:continue
                local=LogisticRegression(C=self.C,max_iter=3000).fit(z[mask],y[mask,j],sample_weight=w[mask])
                alpha=people/(people+self.shrink_people);self.local[key]=(local,alpha);self.support[key].update(active=True,weight=alpha)
        return self
    def predict(self,p,raw):
        p=_probabilities(p,self.experts);z=self._features(p);states=self.state(p,raw)
        scores=np.column_stack([m.decision_function(z) for m in self.global_models])
        for (j,state),(model,alpha) in self.local.items():
            mask=states[:,j]==state
            if mask.any():scores[mask,j]=(1-alpha)*scores[mask,j]+alpha*model.decision_function(z[mask])
        return expit(scores)
