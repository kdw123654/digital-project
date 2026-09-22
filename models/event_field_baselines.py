"""Comparison controls, never called by the proposed field model."""
import torch
from torch import nn
from .event_plastic_field import EventPlasticField

class MatchedResidualMLP(nn.Module):
    def __init__(self,input_dim,output_dim,parameter_target,seed=42):
        super().__init__();torch.manual_seed(seed)
        # Count the common trainable linear skip as part of every model.
        skip=(input_dim+1)*output_dim
        width=max(4,round((parameter_target-skip-output_dim)/(input_dim+output_dim+1)))
        self.linear_skip=nn.Linear(input_dim,output_dim)
        self.readout=nn.Sequential(nn.Linear(input_dim,width),nn.SiLU(),nn.Linear(width,output_dim))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
    def forward(self,x):return self.linear_skip(x)+self.readout(x)
    set_linear=EventPlasticField.set_linear

class HistoryGRU(nn.Module):
    """Strong history-window GRU control with learned attention readout.

    Receives exactly the same64 observed values as field/MLP. Attention may use
    every within-window hidden state; no final-state compression disadvantage.
    This is named GRU+attention, not reported as a vanilla streaming GRU.
    """
    def __init__(self,input_dim,output_dim,parameter_target,seed=42,heads=8):
        super().__init__();torch.manual_seed(seed);self.input_dim=input_dim;self.heads=heads
        def count(h):return 3*h*2+3*h*h+6*h+h*heads+heads+input_dim*heads+(h*heads+1)*64+(64+1)*output_dim+(input_dim+1)*output_dim
        hidden=min(range(4,65),key=lambda h:abs(count(h)-parameter_target))
        self.gru=nn.GRU(2,hidden,batch_first=True);self.content=nn.Linear(hidden,heads)
        locations=torch.linspace(-1,1,input_dim);centers=torch.linspace(-1,1,heads)
        self.position=nn.Parameter(-10*(locations[:,None]-centers[None,:]).square())
        self.register_buffer('positions',locations)
        self.readout=nn.Sequential(nn.Linear(heads*hidden,64),nn.SiLU(),nn.Linear(64,output_dim));self.linear_skip=nn.Linear(input_dim,output_dim)
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
    def forward(self,x):
        sequence=torch.stack([x,self.positions.expand(len(x),-1)],2);hidden,_=self.gru(sequence)
        weights=torch.softmax(self.content(hidden)+self.position,dim=1)
        pooled=torch.einsum('nth,ntk->nkh',hidden,weights).flatten(1)
        return self.linear_skip(x)+self.readout(pooled)
    set_linear=EventPlasticField.set_linear
