"""Allowlisted JSON/NPZ model archives; never import a class named by a file."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from .native_candidates import physics_module
from .vortex_improved import ImprovedVortex
from .vortex_alternating import VortexAlternating
from .physical_generalizing import PhysicalGeneralizing
from .portia_generalizing import GeneralizingPortia
from .hybrid_ensembles import UniformProbabilityEnsemble,UniformLogitEnsemble,LogisticStackingEnsemble
from .conditional_hybrid import InputContext,ContextualLogitStack,ContextualMixture
from .statewise_hybrid import StatewiseStack

def registry():
    types=[LogisticRegression,VortexAlternating,PhysicalGeneralizing,GeneralizingPortia,
           UniformProbabilityEnsemble,UniformLogitEnsemble,LogisticStackingEnsemble,
           InputContext,ContextualLogitStack,ContextualMixture,StatewiseStack,physics_module().Medium]
    return {cls.__name__:cls for cls in types}

def save_archive(value,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);arrays={};allowed=registry()
    def encode(v):
        if v is None or isinstance(v,(str,bool,int,float)):return v
        if isinstance(v,np.generic):return encode(v.item())
        if isinstance(v,(np.ndarray,torch.Tensor)):
            arr=v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v
            if arr.dtype.hasobject:raise TypeError('Object arrays forbidden')
            key=f'a{len(arrays):04d}';arrays[key]=arr
            return {'type':'tensor' if isinstance(v,torch.Tensor) else 'array','key':key}
        if isinstance(v,(list,tuple)):return {'type':'tuple' if isinstance(v,tuple) else 'list','items':[encode(a) for a in v]}
        if isinstance(v,dict):return {'type':'dict','items':[[encode(k),encode(a)] for k,a in v.items()]}
        if isinstance(v,ImprovedVortex):
            return {'type':'wave','config':{'input_dim':v.input_dim,'latent_dim':v.latent_dim,'steps':v.steps,'dropout':0.},'state':encode(dict(v.state_dict()))}
        name=type(v).__name__
        if allowed.get(name) is not type(v):raise TypeError(f'Unapproved object: {name}')
        omitted={'history','diagnostics','diagnostics_','_projection_t','_residual_mean_t','_whitener_t'}
        return {'type':'model','class':name,'attributes':encode({k:a for k,a in vars(v).items() if k not in omitted})}
    schema={'format':'allowlisted-model-npz-v1','value':encode(value)}
    (folder/'model.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    np.savez_compressed(folder/'weights.npz',**arrays)

def load_archive(folder):
    folder=Path(folder);schema=json.loads((folder/'model.json').read_text(encoding='utf-8'))
    if schema.get('format')!='allowlisted-model-npz-v1':raise ValueError('Unknown archive format')
    with np.load(folder/'weights.npz',allow_pickle=False) as store:arrays={k:store[k] for k in store.files}
    allowed=registry()
    def decode(v):
        if not isinstance(v,dict):return v
        kind=v['type']
        if kind=='array':return arrays[v['key']].copy()
        if kind=='tensor':return torch.tensor(arrays[v['key']])
        if kind in ['list','tuple']:
            items=[decode(a) for a in v['items']];return tuple(items) if kind=='tuple' else items
        if kind=='dict':return {decode(k):decode(a) for k,a in v['items']}
        if kind=='wave':
            model=ImprovedVortex(**v['config']);model.load_state_dict(decode(v['state']));model.eval();return model
        if kind=='model':
            if v['class'] not in allowed:raise ValueError('Unknown model class')
            model=object.__new__(allowed[v['class']]);model.__dict__.update(decode(v['attributes']))
            if isinstance(model,VortexAlternating):model._set_tensor_transform()
            return model
        raise ValueError('Unknown archive value type')
    return decode(schema['value'])
