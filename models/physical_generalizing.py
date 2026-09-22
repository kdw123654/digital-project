"""Control variance of physical novelty without changing the physical dynamics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .physical_improved import ImprovedPhysical
from .native_candidates import physics_module

class PhysicalGeneralizing(ImprovedPhysical):
    def __init__(self,C=.1,feature_scale=.1,adaptation=.003,seed=42,feature_rank=6):
        super().__init__(C,feature_scale,adaptation,seed);self.feature_rank=feature_rank

    def fit(self,x,y,w):
        x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
        if x.ndim!=2 or y.shape!=(len(x),4) or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isin(y,[0,1]).all() or w.shape!=(len(x),) or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid fit data')
        wn=w/w.sum();mod=physics_module();self.medium=mod.Medium.random(n=24,input_dim=x.shape[1],seed=self.seed)
        self.medium.drive/=np.sqrt(x.shape[1]);before=self.medium.coupling.copy()
        self.input_mean=x.mean(0);self.input_scale=x.std(0);self.input_scale[self.input_scale<1e-8]=1
        self._response(x,adapt=True);physics=self._response(x)
        self.reg_mean=np.average(x,axis=0,weights=w);self.reg_scale=np.sqrt(np.average((x-self.reg_mean)**2,axis=0,weights=w));self.reg_scale[self.reg_scale<1e-8]=1
        z=np.column_stack([np.ones(len(x)),(x-self.reg_mean)/self.reg_scale]);penalty=np.eye(z.shape[1])*1e-4;penalty[0,0]=0
        self.linear_projection=np.linalg.solve(z.T@(wn[:,None]*z)+penalty,z.T@(wn[:,None]*physics))
        residual=physics-z@self.linear_projection;self.residual_mean=np.average(residual,axis=0,weights=w);centered=residual-self.residual_mean
        eigen,vec=np.linalg.eigh(centered.T@(wn[:,None]*centered));idx=np.argsort(eigen)[::-1];idx=idx[eigen[idx]>max(float(eigen.max())*1e-3,1e-10)][:self.feature_rank]
        if not len(idx):raise RuntimeError('No stable physical features')
        self.eigenvectors=vec[:,idx]
        # Do not amplify weak directions to unit variance: ridge-whitening floor.
        self.eigen_scale=np.sqrt(np.maximum(eigen[idx],eigen.max()*.25))
        features=self._features(x);self.readout=[LogisticRegression(C=self.C,max_iter=3000).fit(features,y[:,j],sample_weight=w) for j in range(4)]
        self.diagnostics={'mechanism':'same quartic medium,BAOAB,local adaptation,transient/free decay',
          'coupling_relative_change':float(np.linalg.norm(self.medium.coupling-before)/np.linalg.norm(before)),
          'retained_novel_dim':len(idx),'new_feature_scale':self.feature_scale,'original_basis_unchanged':bool(np.array_equal(features[:,:x.shape[1]],x)),
          'new_feature_std':np.std(features[:,x.shape[1]:],axis=0).tolist(),'weak_direction_floor_fraction':.25}
        return self
