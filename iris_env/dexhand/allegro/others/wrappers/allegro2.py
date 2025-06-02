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
        

    def reset(self, init_qpos: jax.Array, init_qvel: jax.Array):
        if init_qpos.ndim ==1:
            init_qpos = jnp.tile(init_qpos, (self.batch_size, 1))
        if init_qvel.ndim==1:
            init_qvel = jnp.tile(init_qvel,(self.batch_size,1 ))

        self.mjx_data_batch = jax.vmap(lambda d, qp, qv: d.replace(qpos=qp, qvel=qv))(
            self.mjx_data_batch, init_qpos, init_qvel)

        self.mjx_data_batch = jax.vmap(mjx.forward)(
            self.mjx_model_batch, self.mjx_data_batch)

        return self.mjx_data_batch
    
    
    def step(self, action_batch: jax.Array):
        
        # get current joint positions for all batch elements
        curr_jpos_batch = self.mjx_data_batch.qpos[:,0:self.n_qpos]
        
        # Calculate target joint positions for all batch elements
        target_jpos_batch = curr_jpos_batch + action_batch

        # Define a single step function
        def single_step(model, data, target_jpos):
            data = data.replace(ctrl=target_jpos)
            return mjx.step(model, data)
        
        # Create a batched version of the single step function
        batched_single_step = jax.vmap(single_step, in_axes=(0, 0, 0))
        
        # Perform frame_skip steps
        def body_fun(_, loop_carry):
            return batched_single_step(self.mjx_model_batch, loop_carry, target_jpos_batch)

        self.mjx_data_batch = jax.lax.fori_loop(
            0, self.frame_skip, body_fun, self.mjx_data_batch
        )
        
        return self.mjx_data_batch
