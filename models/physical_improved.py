"""Physics-preserving transient reservoir with train-only local adaptation.

The original quartic medium and BAOAB integrator are reused. Labels only train
the readout. State is reset per person; simulation time is not health time.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from .native_candidates import physics_module


class ImprovedPhysical:
    def __init__(self,C=.1,feature_scale=.3,adaptation=.003,seed=42):
        self.C=C;self.feature_scale=feature_scale;self.adaptation=adaptation;self.seed=seed

    def _response(self,x,adapt=False):
        mod=physics_module();sim=mod.Simulator(self.medium,dt=.04,seed=self.seed,batch=len(x))
        drive=np.tanh((x-self.input_mean)/self.input_scale)
        values=[];kinetic=[]
        # Observe driven response and free decay, not redundant x/v at one instant.
        for step in range(64):
            sim.step(drive if step<32 else np.zeros_like(drive),temperature=0.)
            if adapt and self.adaptation and step%8==7:sim.adapt(rate=self.adaptation,threshold=.005)
            if step in [7,31,63]:values.append(sim.x.copy())
            kinetic.append(sim.v**2)
        values.append(np.mean(kinetic,axis=0))
        return np.concatenate(values,axis=1)

    def _features(self,x):
        physics=self._response(x)
        z=np.column_stack([np.ones(len(x)),(x-self.reg_mean)/self.reg_scale])
        residual=physics-z@self.linear_projection
        novel=(residual-self.residual_mean)@self.eigenvectors/self.eigen_scale
        novel=np.clip(novel,-6,6)*self.feature_scale
        # Preserve the original LR basis scale and regularization geometry.
        return np.column_stack([x,novel])

    def fit(self,x,y,w):
        x=np.asarray(x,float);y=np.asarray(y,float);w=np.asarray(w,float)
        if x.ndim!=2 or y.shape!=(len(x),4) or w.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<=0).any():raise ValueError('Invalid fit data')
        wn=w/w.sum();mod=physics_module()
        self.medium=mod.Medium.random(n=24,input_dim=x.shape[1],seed=self.seed)
        self.medium.drive/=np.sqrt(x.shape[1]);before=self.medium.coupling.copy()
        self.input_mean=x.mean(0);self.input_scale=x.std(0);self.input_scale[self.input_scale<1e-8]=1
        self._response(x,adapt=True)
        physics=self._response(x)
        self.reg_mean=np.average(x,axis=0,weights=w)
        self.reg_scale=np.sqrt(np.average((x-self.reg_mean)**2,axis=0,weights=w));self.reg_scale[self.reg_scale<1e-8]=1
        z=np.column_stack([np.ones(len(x)),(x-self.reg_mean)/self.reg_scale]);penalty=np.eye(z.shape[1])*1e-5;penalty[0,0]=0
        self.linear_projection=np.linalg.solve(z.T@(wn[:,None]*z)+penalty,z.T@(wn[:,None]*physics))
        residual=physics-z@self.linear_projection;self.residual_mean=np.average(residual,axis=0,weights=w)
        centered=residual-self.residual_mean;cov=centered.T@(wn[:,None]*centered)
        eigen,vec=np.linalg.eigh(cov);idx=np.argsort(eigen)[::-1];idx=idx[eigen[idx]>max(float(eigen.max())*1e-4,1e-10)][:24]
        if not len(idx):raise RuntimeError('No nonredundant physical features survived')
        self.eigenvectors=vec[:,idx];self.eigen_scale=np.sqrt(eigen[idx])
        features=self._features(x)
        self.readout=[LogisticRegression(C=self.C,max_iter=3000).fit(features,y[:,j],sample_weight=w) for j in range(4)]
        self.diagnostics={'coupling_relative_change':float(np.linalg.norm(self.medium.coupling-before)/np.linalg.norm(before)),
             'physical_raw_dim':physics.shape[1],'retained_novel_dim':len(idx),'linear_residual_variance_fraction':float(np.var(residual)/max(np.var(physics),1e-12)),
             'feature_scale':self.feature_scale,'no_backpropagation_in_medium':True,'adaptation_uses_labels':False}
        return self

    def predict(self,x):
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.input_mean) or not np.isfinite(x).all():raise ValueError('Invalid prediction matrix')
        features=self._features(x)
        return np.column_stack([m.predict_proba(features)[:,1] for m in self.readout])
