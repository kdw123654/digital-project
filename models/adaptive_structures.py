"""Missing-input adaptations: frozen baseline plus bounded structural residuals."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)
PAIR_BITS = np.stack([BITS[:,a]*BITS[:,b] for a in range(4) for b in range(a+1,4)],axis=1)

RAW_GROUPS=[[1,2,3],[0,4,5,6,13,17,18],[7,8,9,10,11,12,14,15,16]]

def reliability(raw):
    raw=np.asarray(raw,float)
    return np.column_stack([np.isfinite(raw[:,ix]).mean(1) for ix in RAW_GROUPS]).astype('float32')

def masked_raw(raw,seed,rate=.2):
    raw=np.array(raw,dtype=float,copy=True);rng=np.random.default_rng(seed)
    mask=rng.random(raw.shape)<rate;mask[:,[0,4]]=False
    # Waist and waist/height ratio are not independently available.
    coupled=mask[:,2]|mask[:,3];mask[:,2]=coupled;mask[:,3]=coupled
    raw[mask]=np.nan
    return raw

class AdaptiveNet(nn.Module):
    def __init__(self,dimension,groups,kind,width=6):
        super().__init__();self.kind=kind;self.dimension=dimension;self.width=width
        self.register_buffer('bits',torch.tensor(BITS.copy()))
        self.register_buffer('pair_bits',torch.tensor(PAIR_BITS.copy()))
        self.anchor=nn.Linear(dimension,4)
        for p in self.anchor.parameters():p.requires_grad=False
        self.layers=nn.ModuleList([nn.Linear(len(ix),width) for ix in groups])
        for i,ix in enumerate(groups):self.register_buffer(f'group_{i}',torch.tensor(ix,dtype=torch.long))
        self.gate=nn.Parameter(torch.full((4,),-2.))
        if kind=='phase_reliable':
            self.amplitudes=nn.ModuleList([nn.Linear(len(ix),width) for ix in groups]);size=9*width
        elif kind=='sparse_reliable':
            n=3*width;mask=(torch.rand(n,n)<.2)&~torch.eye(n,dtype=torch.bool)
            self.register_buffer('edges',mask.nonzero(as_tuple=False));self.edge_weights=nn.Parameter(torch.zeros(int(mask.sum())))
            self.leak=nn.Parameter(torch.full((n,),-1.));size=n
        elif kind=='relax_label':
            size=3*width;self.relation=nn.Parameter(torch.zeros(4,4));self.step=nn.Parameter(torch.tensor(-2.))
        else:raise ValueError(kind)
        self.output=nn.Linear(size,4);self.pair=nn.Linear(size,6)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.pair.weight);nn.init.zeros_(self.pair.bias)

    def initialize(self,models):
        with torch.no_grad():
            self.anchor.weight.copy_(torch.tensor(np.stack([m.coef_[0] for m in models]),dtype=torch.float32))
            self.anchor.bias.copy_(torch.tensor([m.intercept_[0] for m in models],dtype=torch.float32))

    def forward(self,packed):
        x=packed[:,:self.dimension];r=packed[:,self.dimension:];w=self.width
        inputs=[x[:,getattr(self,f'group_{i}')] for i in range(3)]
        states=[torch.tanh(layer(z)) for layer,z in zip(self.layers,inputs)]
        if self.kind=='phase_reliable':
            amplitude=[torch.sigmoid(layer(z))*r[:,i:i+1] for i,(layer,z) in enumerate(zip(self.amplitudes,inputs))]
            phase=[np.pi*z for z in states]
            real=[a*torch.cos(p) for a,p in zip(amplitude,phase)];imag=[a*torch.sin(p) for a,p in zip(amplitude,phase)]
            interaction=[real[i]*real[j]+imag[i]*imag[j] for i,j in [(0,1),(0,2),(1,2)]]
            h=torch.cat(real+imag+interaction,dim=1)
        elif self.kind=='sparse_reliable':
            drive=torch.cat([s*r[:,i:i+1] for i,s in enumerate(states)],dim=1);h=drive
            matrix=x.new_zeros((3*w,3*w));matrix[self.edges[:,0],self.edges[:,1]]=torch.tanh(self.edge_weights)
            matrix=matrix/(1+matrix.abs().sum(1,keepdim=True));leak=.5*torch.sigmoid(self.leak)
            for _ in range(4):h=h+leak*(torch.tanh(drive+h@matrix.T)-h)
        else:h=torch.cat([s*r[:,i:i+1] for i,s in enumerate(states)],dim=1)
        unary=self.anchor(x)+torch.sigmoid(self.gate)*torch.tanh(self.output(h))
        if self.kind=='relax_label':
            relation=(self.relation+self.relation.T)/2;relation=relation-torch.diag(torch.diag(relation))
            relation=relation/(1+torch.linalg.matrix_norm(relation,ord='fro'));q=torch.sigmoid(unary)
            # Four label states, not an arbitrary ordering of participants/features.
            for _ in range(4):q=q+torch.sigmoid(self.step)*(torch.sigmoid(unary+(q-.5)@relation)-q)
            unary=torch.logit(q.clamp(1e-6,1-1e-6))
        score=unary@self.bits.T+.1*torch.tanh(self.pair(h))@self.pair_bits.T
        return F.log_softmax(score,dim=1)
