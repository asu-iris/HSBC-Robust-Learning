from flax import nnx
from flax import linen as nn
from flax.nnx import bridge
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import List, Dict, Union

class TestNetworkJax(nnx.Module):
    def __init__(self,input_dim=5,hidden_dim=64,num_hidden_layers=3,rngs=nnx.Rngs(0)):
        # Define your layers here
        self.activation=nnx.relu
        self.tanh = nnx.tanh
        #network for reward prediction
        self.fc_0=nnx.Linear(in_features=input_dim, out_features=hidden_dim,rngs=rngs)
        self.fch = nnx.Sequential(*[
            nnx.Sequential(
                nnx.Linear(in_features=hidden_dim, out_features=hidden_dim,rngs=rngs),  # Adjust input size to N_HIDDEN
                self.activation
            ) for _ in range(num_hidden_layers - 1)
        ])
        self.fc_n=nnx.Linear(in_features=hidden_dim, out_features=1,rngs=rngs)

    def __call__(self,x):
        r=self.fc_0(x)
        r=self.activation(r)
        r=self.fch(r)
        r=self.fc_n(r)
        r=self.tanh(r)
        return r
    
NestedDict = Dict[str, Union[jnp.ndarray, 'NestedDict']]

def stack_nested_dicts(dicts: List[NestedDict]) -> NestedDict:
    """
    Stack parameters from a list of nested dictionaries with the same structure.

    Args:
        dicts (List[NestedDict]): A list of nested dictionaries with the same structure,
                                  where leaf values are JAX arrays.

    Returns:
        NestedDict: A nested dictionary with the same structure, where the leaf arrays
                    are stacked along the first dimension.
    """
    def stack_fn(*arrays):
        """Stack arrays along the first dimension."""
        return jnp.stack(arrays, axis=0)

    def recursive_stack(keys, *values):
        """Recursively stack values in nested dictionaries."""
        if isinstance(values[0], dict):
            return {key: recursive_stack(key, *(v[key] for v in values)) for key in values[0].keys()}
        return stack_fn(*values)

    return recursive_stack(None, *dicts)

def get_ensemble_model(model_num=128,input_dim=5,hidden_dim=64,num_hidden_layers=3):
    model = bridge.to_linen(TestNetworkJax, input_dim, hidden_dim=hidden_dim,num_hidden_layers=num_hidden_layers)
    param_list = []
    data = jnp.zeros(shape=(10,input_dim))
    for i in range(model_num):
        key = jax.random.key(int(time.time()))
        state_dict = model.init(key, data)
        param_list.append(state_dict['params'])

    params =stack_nested_dicts(param_list)
    NNX_DATA = state_dict['nnx']

    return model, params, NNX_DATA

def forward_batched(model, batched_params, batched_inputs, NNX_DATA):
    """
    Forward pass for a batch of inputs using a batch of parameters.
    
    Args:
        batched_params: A batch of parameters (batch_size x original param structure).
        batched_inputs: A batch of inputs (batch_size x input_dim).
    
    Returns:
        Outputs for the batch.
    """
    def single_forward(params, x):
        return model.apply({'params': params , 'nnx':NNX_DATA}, x)
    
    # Vectorize the forward pass over the batch
    return jax.vmap(single_forward,(0,None),0)(batched_params, batched_inputs)