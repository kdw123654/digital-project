"""Small, stateless multilabel adaptations of the user's research ideas.

The internal recurrence index is computation, never participant/time ordering.
These are independent tabular adaptations, not biological/physical simulators.
"""
from itertools import combinations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(np.float32)

PAIRS = list(combinations(range(4), 2))
PAIR_BITS = np.stack([BITS[:, a]*BITS[:, b] for a,b in PAIRS], axis=1)
ARCHITECTURES = ['independent_mlp','powerset_mlp','group_hurdle','pair_energy',
                 'phase_pair','phase_no_interference','sparse_pair','sparse_no_recurrence',
                 'relax_pair','relax_uncoupled']


class ResearchNet(nn.Module):
    def __init__(self, dimension, groups, architecture, width=12):
        super().__init__()
        if architecture not in ARCHITECTURES:raise ValueError(architecture)
        self.architecture=architecture;self.width=width
        self.register_buffer('bits',torch.tensor(BITS.copy()))
        self.register_buffer('pair_bits',torch.tensor(PAIR_BITS.copy()))
        self.linear=nn.Linear(dimension,4)
        self.hidden=nn.Linear(dimension,width)
        self.residual=nn.Linear(width,4,bias=False)
        if architecture in ['group_hurdle','pair_energy']:
            self.group_layers=nn.ModuleList([nn.Linear(len(g),4) for g in groups])
            for i,g in enumerate(groups):self.register_buffer(f'group_{i}',torch.tensor(g,dtype=torch.long))
            self.group_output=nn.Linear(20,4,bias=False)
            # No unused common hidden branch in the grouped controls.
            del self.hidden,self.residual
        if architecture=='group_hurdle':self.union=nn.Linear(dimension+20,1)
        if architecture=='powerset_mlp':self.powerset=nn.Linear(width,16)
        if architecture not in ['independent_mlp','powerset_mlp','group_hurdle']:
            self.pair=nn.Linear(20 if architecture=='pair_energy' else width,6)
            nn.init.zeros_(self.pair.weight);nn.init.zeros_(self.pair.bias)
        if architecture.startswith('phase'):
            self.amplitude=nn.Linear(dimension,width)
            self.phase=nn.Linear(dimension,width)
        if architecture.startswith('sparse'):
            # Seeded 25%-density wiring. Actual parameters only on selected edges.
            mask=(torch.rand(width,width)<.25)&~torch.eye(width,dtype=torch.bool)
            edges=mask.nonzero(as_tuple=False)
            self.register_buffer('edges',edges)
            if architecture=='sparse_pair':
                self.edge_weight=nn.Parameter(torch.zeros(len(edges)))
                self.leak=nn.Parameter(torch.zeros(width))
        if architecture.startswith('relax'):
            self.tau=nn.Parameter(torch.zeros(width))
            if architecture=='relax_pair':self.coupling=nn.Parameter(torch.zeros(width,width))
        if hasattr(self,'residual'):nn.init.zeros_(self.residual.weight)
        if hasattr(self,'group_output'):nn.init.zeros_(self.group_output.weight)

    def initialize(self, lr):
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(np.stack([m.coef_[0] for m in lr]),dtype=torch.float32))
            self.linear.bias.copy_(torch.tensor([m.intercept_[0] for m in lr],dtype=torch.float32))

    def representation(self,x):
        a=self.architecture
        if a in ['group_hurdle','pair_energy']:
            body,context,behavior=[torch.tanh(layer(x[:,getattr(self,f'group_{i}')])) for i,layer in enumerate(self.group_layers)]
            return torch.cat([body,context,behavior,body*context,body*behavior],dim=1)
        drive=torch.tanh(self.hidden(x))
        if a.startswith('phase'):
            amp=torch.sigmoid(self.amplitude(x));phase=np.pi*torch.tanh(self.phase(x))
            real=amp*torch.cos(phase);imag=amp*torch.sin(phase)
            # Interference between adjacent latent channels, within each person.
            interference=real*torch.roll(real,1,1)+imag*torch.roll(imag,1,1)
            return drive+interference if a=='phase_pair' else drive+amp.square()
        if a=='sparse_pair':
            matrix=x.new_zeros((self.width,self.width))
            matrix[self.edges[:,0],self.edges[:,1]]=torch.tanh(self.edge_weight)
            matrix=matrix/(matrix.abs().sum(1,keepdim=True)+1)
            h=torch.zeros_like(drive);leak=torch.sigmoid(self.leak)
            for _ in range(4):h=(1-leak)*h+leak*torch.tanh(drive+h@matrix.T-.1*h.mean(1,keepdim=True))
            return h
        if a.startswith('relax'):
            h=torch.zeros_like(drive);step=.5*torch.sigmoid(self.tau)
            matrix=x.new_zeros((self.width,self.width))
            if a=='relax_pair':
                matrix=(self.coupling+self.coupling.T)/2
                matrix=matrix/(1+torch.linalg.matrix_norm(matrix,ord='fro'))
            for _ in range(4):h=h+step*(-h+torch.tanh(drive+h@matrix))
            return h
        return drive

    def forward(self,x):
        h=self.representation(x);a=self.architecture
        if a=='powerset_mlp':return F.log_softmax(self.linear(x)@self.bits.T+self.powerset(h),dim=1)
        unaries=self.linear(x)+(self.group_output(h) if hasattr(self,'group_output') else self.residual(h))
        if a=='group_hurdle':
            union=self.union(torch.cat([x,h],dim=1))
            conditional=F.log_softmax(unaries@self.bits[1:].T,dim=1)
            return torch.cat([F.logsigmoid(-union),F.logsigmoid(union)+conditional],dim=1)
        score=unaries@self.bits.T
        if hasattr(self,'pair'):score=score+.5*torch.tanh(self.pair(h))@self.pair_bits.T
        return F.log_softmax(score,dim=1)


def objective(log_joint,labels,joint_labels,weight):
    marginal=(log_joint.exp()@torch.tensor(BITS.copy(),device=log_joint.device)).clamp(1e-6,1-1e-6)
    nll=F.nll_loss(log_joint,joint_labels,reduction='none')/4
    bce=F.binary_cross_entropy(marginal,labels,reduction='none').mean(1)
    return ((nll+.2*bce)*weight).sum()/weight.sum()
