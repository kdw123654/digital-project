"""One event-plastic complex-field cell, not an ensemble of three networks.

The same complex field propagates through a Hermitian Cayley operator, evolves
under an onsite quartic phase flow, fires/resets, and changes its own next-step
coupling through an antisymmetric timing trace. i*antisymmetric(real) is Hermitian.
All predictive parameters are trained together. Fast plasticity is input-driven
state, with no labels in forward inference. Academic priority is not asserted.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

class SurrogateEvent(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):ctx.save_for_backward(x);return (x>=0).to(x.dtype)
    @staticmethod
    def backward(ctx,gradient):
        (x,)=ctx.saved_tensors
        return gradient/(1+5*x.abs()).square()

def inverse_softplus(x):return math.log(math.expm1(x))

class EventPlasticField(nn.Module):
    def __init__(self,input_dim,output_dim,dimension=16,steps=6,plasticity=True,quartic=True,seed=42,head_width=64):
        super().__init__();torch.manual_seed(seed)
        self.input_dim=input_dim;self.output_dim=output_dim;self.dimension=dimension;self.steps=steps;self.plasticity=plasticity;self.quartic=quartic
        self.drive=nn.Linear(input_dim,2*dimension)
        self.h_real=nn.Parameter(torch.randn(dimension,dimension)/math.sqrt(dimension))
        self.h_imag=nn.Parameter(torch.randn(dimension,dimension)*.05)
        self.decay_raw=nn.Parameter(torch.full((dimension,),2.5))
        self.quartic_raw=nn.Parameter(torch.full((dimension,),inverse_softplus(.5)))
        initial_threshold=torch.linspace(.08,.3,dimension)
        self.threshold_raw=nn.Parameter(torch.log(torch.expm1(initial_threshold)))
        self.reset_raw=nn.Parameter(torch.full((dimension,),-.4))
        self.trace_raw=nn.Parameter(torch.full((dimension,),1.4))
        self.homeostasis_raw=nn.Parameter(torch.full((dimension,),inverse_softplus(1.)))
        self.plastic_decay_raw=nn.Parameter(torch.tensor(2.))
        self.plastic_gain_raw=nn.Parameter(torch.tensor(-1.5))
        self.dt_raw=nn.Parameter(torch.tensor(inverse_softplus(.2)))
        self.readout=nn.Sequential(nn.Linear(5*dimension,head_width),nn.SiLU(),nn.Linear(head_width,output_dim))
        self.linear_skip=nn.Linear(input_dim,output_dim)
        # Shared comparison initialization: fitted train-only linear function plus
        # zero initial nonlinear correction. Nothing stays frozen during training.
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
        self.last_diagnostics={}

    def initial_state(self,batch,device=None):
        device=self.drive.weight.device if device is None else device;d=self.dimension
        return (torch.zeros(batch,d,dtype=torch.complex64,device=device),torch.zeros(batch,d,device=device),torch.zeros(batch,d,d,device=device))

    def hamiltonian(self,plastic):
        real=(self.h_real+self.h_real.T)/2;imag=(self.h_imag-self.h_imag.T)/2
        if self.plasticity:imag=imag+plastic
        return torch.complex(real.expand(plastic.shape[0],-1,-1),imag.expand(plastic.shape[0],-1,-1) if imag.ndim==2 else imag)

    def advance(self,x,state,base_unitary=None):
        z,trace,plastic=state;dt=F.softplus(self.dt_raw).clamp(.025,.5)
        eye=torch.eye(self.dimension,dtype=torch.complex64,device=x.device)
        # Solve instead of explicitly inverting the Cayley denominator.
        if base_unitary is None:
            h=self.hamiltonian(plastic)
            propagated=torch.linalg.solve(eye+1j*dt*h,((eye-1j*dt*h)@z.unsqueeze(-1))).squeeze(-1)
        else:propagated=z@base_unitary.T
        # Trainable linear injection avoids saturating away input magnitudes.
        encoded=.1*self.drive(x).reshape(-1,self.dimension,2)
        driven=(.5+.499*torch.sigmoid(self.decay_raw))*propagated+torch.view_as_complex(encoded.contiguous())
        energy=driven.abs().square()
        coefficient=F.softplus(self.quartic_raw) if self.quartic else torch.zeros_like(self.quartic_raw)
        evolved=driven*torch.exp(-1j*dt*coefficient*energy)
        threshold=(F.softplus(self.threshold_raw)+.01)*(1+F.softplus(self.homeostasis_raw)*trace)
        event=SurrogateEvent.apply((energy-threshold)/(threshold.detach()+.1))
        # A spike removes a threshold-sized intensity budget, retaining excess
        # signal; multiplying all suprathreshold amplitudes by one constant lost it.
        reset=.05+.9*torch.sigmoid(self.reset_raw)
        # Forward is unchanged for actual spikes (intensity >= threshold).
        # Bound the surrogate's counterfactual gradient below threshold instead
        # of dividing its nonzero derivative by nearly zero field intensity.
        denominator=torch.maximum(energy,threshold.detach()).clamp_min(1e-6)
        fraction=(1-reset*threshold*event/denominator).clamp_min(.01)
        next_z=evolved*torch.sqrt(fraction)
        trace_decay=.5+.499*torch.sigmoid(self.trace_raw);next_trace=trace_decay*trace+(1-trace_decay)*event
        timing=trace.unsqueeze(2)*event.unsqueeze(1)-event.unsqueeze(2)*trace.unsqueeze(1)
        plastic_decay=.5+.499*torch.sigmoid(self.plastic_decay_raw);gain=.2*torch.sigmoid(self.plastic_gain_raw)
        next_plastic=.5*torch.tanh((plastic_decay*plastic+gain*timing)/.5) if self.plasticity else torch.zeros_like(plastic)
        return (next_z,next_trace,next_plastic),event,energy

    def forward(self,x,return_diagnostics=False):
        if x.ndim!=2 or x.shape[1]!=self.input_dim:raise ValueError('Wrong field input shape')
        state=self.initial_state(len(x));events=[];energies=[]
        # Plasticity is identically zero for the first two transitions: initial
        # trace is zero, hence the first antisymmetric timing update is zero.
        h=self.hamiltonian(state[2][:1])[0];dt=F.softplus(self.dt_raw).clamp(.025,.5)
        eye=torch.eye(self.dimension,dtype=torch.complex64,device=x.device)
        base_unitary=torch.linalg.solve(eye+1j*dt*h,eye-1j*dt*h)
        for step in range(self.steps):
            state,event,energy=self.advance(x,state,base_unitary if step<2 or not self.plasticity else None);events.append(event);energies.append(energy)
        z,trace,plastic=state
        # Every nonlinear correction passes through this single field state.
        # The common linear skip preserves information without an MLP bypass.
        features=torch.cat([z.real,z.imag,z.abs().square(),trace,energies[-1]],1)
        output=self.linear_skip(x)+self.readout(features)
        if return_diagnostics:
            return output,{'event_rate':torch.stack(events).mean(),'field_energy':torch.stack(energies).mean(),'plastic_norm':plastic.norm(dim=(1,2)).mean(),'field_features':features}
        return output

    def set_linear(self,coef,bias):
        with torch.no_grad():
            self.linear_skip.weight.copy_(torch.as_tensor(coef,dtype=self.linear_skip.weight.dtype,device=self.linear_skip.weight.device))
            self.linear_skip.bias.copy_(torch.as_tensor(bias,dtype=self.linear_skip.bias.dtype,device=self.linear_skip.bias.device))

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
