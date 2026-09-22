"""Numerically identical field with one CUDA solve-status check per forward."""
import torch
from torch.nn import functional as F
from .event_plastic_field import EventPlasticField

class FastEventPlasticField(EventPlasticField):
    def advance(self,x,state,base_unitary=None):
        if base_unitary is not None or x.device.type!='cuda':return super().advance(x,state,base_unitary)
        z,trace,plastic=state;dt=F.softplus(self.dt_raw).clamp(.025,.5);h=self.hamiltonian(plastic)
        eye=torch.eye(self.dimension,dtype=torch.complex64,device=x.device)
        propagated,info=torch.linalg.solve_ex(eye+1j*dt*h,(eye-1j*dt*h)@z.unsqueeze(-1),check_errors=False)
        if getattr(self,'_inside_forward',False):self._solve_infos.append(info)
        elif bool(info.ne(0).any()):raise FloatingPointError('Cayley solve failed')
        # Reuse the entire original physical/event/plastic update. The identity
        # supplies the already propagated field without a second Cayley solve.
        return super().advance(x,(propagated.squeeze(-1),trace,plastic),eye)

    def forward(self,x,return_diagnostics=False):
        self._solve_infos=[];self._inside_forward=True
        try:result=super().forward(x,return_diagnostics)
        finally:self._inside_forward=False
        if self._solve_infos:
            invalid=torch.cat([x.reshape(-1) for x in self._solve_infos]).ne(0).any()
            self._solve_infos=[]
            if bool(invalid):raise FloatingPointError('Cayley solve failed before optimizer update')
        return result
