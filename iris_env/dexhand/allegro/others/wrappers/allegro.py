import mujoco
import jax

from mujoco import mjx
from jax import numpy as jnp


class AllegroEnvMJX:

    def __init__(self, param):

        # model path
        self.model_path = 'envs/allegro/models/allegro_right_hand.xml'

        # Initialize parameters and fetch batch size
        self.param_ = param
        self.batch_size = param.batch_size

        # system dimensions:
        self.n_qpos = 16
        self.n_qvel = 16
        self.n_action = 16

        # low-level control loop
        self.frame_skip = int(100)

        # Initialize model and data
        self.mj_model = mujoco.MjModel.from_xml_path(self.model_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # In put it on device
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        # Generate model batches
        self.mjx_model_batch = jax.tree_map(
            lambda x: jnp.tile(x[None], (self.batch_size,) + (1,) * (x.ndim)),
            self.mjx_model
        )
        
        # Generate data batches
        self.mjx_data_batch = jax.vmap(mjx.make_data)(self.mjx_model_batch)
        

    def reset(self, mjx_model: mjx.Model, init_qpos: jax.Array, init_qvel: jax.Array):
        
        mjx_data = mjx.make_data(mjx_model)
        mjx_data.replace(qpos=init_qpos, qvel=init_qvel)
        
        return mjx.forward(mjx_model,mjx_data)
    
    
    def step(self, mjx_model: mjx.Model, mjx_data:mjx.Data, action: jax.Array):
        
        # Calculate target joint positions for all batch elements
        target_jpos = mjx_data.qpos[0:self.n_qpos] + action

        # Define a single step function
        def single_step(model, data, target_jpos):
            data = data.replace(ctrl=target_jpos)
            return mjx.step(model, data)
               
        def f(data, _):
            return (
                mjx.step(mjx_model, data),
                None,
            )
            
        return jax.lax.scan(f, mjx_data, (), self.n_frames)[0]
                

