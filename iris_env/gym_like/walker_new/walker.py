from typing import Callable, Dict, Optional, Tuple

from brax import base
from brax.envs.base import PipelineEnv, Wrapper
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco

import pathlib
import os

class Walker(PipelineEnv):

    def __init__(self, env_param, frame_skip=10, timestep=0.002,  **kwargs):

        # env param
        self.param = env_param
        filedir = pathlib.Path(__file__).resolve().parent

        # load model
        path = os.path.join(filedir,'model/walker.xml')
        self.mj_model=mujoco.MjModel.from_xml_path(path)
        self.mj_model.opt.timestep = timestep
        
        sys = mjcf.load_model(self.mj_model)
        super().__init__(sys=sys, backend='mjx', n_frames=frame_skip, **kwargs)
        self.obs_dim = None
        self.action_dim = 6


    def reset(self, qpos: jax.Array, qvel: jax.Array) -> base.State:
        pipeline_state = self.pipeline_init(qpos, qvel)
        return pipeline_state

    def step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """Runs one timestep of the environment's dynamics."""
        assert pipeline_state is not None

        pipeline_state = self.pipeline_step(pipeline_state, action)
        return pipeline_state

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        positions = pipeline_state.q[1:]
        velocity = jnp.clip(pipeline_state.qd, -10, 10)
        obs = jnp.concat((positions,velocity),axis = -1)
        return obs



def do_batching(state: base.State, batch_size: int):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=batch_size, axis=0)
    return jax.tree_util.tree_map(replicate_field,state)

def extract_from_batch(state_batch: base.State, index: int):
    def fun_body(x):
        return x[index]
    return jax.tree_util.tree_map(fun_body,state_batch)



def get_obs(pipeline_state: base.State) -> jax.Array:
    """Returns the environment observations."""
    positions = pipeline_state.q[1:]
    velocity = jnp.clip(pipeline_state.qd, -10, 10)
    obs = jnp.concat((positions,velocity),axis = -1)
    return obs

# for the above function,
# you don't have to maintain the position of batching dimention (using ....), just write like a non-batch function
# vmap will directly handle for you when you do vmap:
# print(jax.vmap(get_obs)(planner_state_batch))

class VmapWrapper(Wrapper):
  """Vectorizes bBrax env."""

  def __init__(self, env: PipelineEnv, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, qpos_batch: jax.Array, qvel_batch: jax.Array) -> base.State:
    if qpos_batch.ndim==1:
       qpos_batch=do_batching(qpos_batch, self.batch_size)
    if qvel_batch.ndim==1:
       qvel_batch=do_batching(qvel_batch, self.batch_size)

    return jax.vmap(self.env.reset)(qpos_batch, qvel_batch)

  def step(self, pipeline_state_batch: base.State, action_batch: jax.Array) -> base.State:
    if action_batch.ndim==1:
       action_batch=do_batching(action_batch, self.batch_size)

    return jax.vmap(self.env.step)(pipeline_state_batch, action_batch)
  
  def _get_obs(self, pipeline_state_batch: base.State) ->base.State:
    return jax.vmap(self.env._get_obs)(pipeline_state_batch)
  


class RolloutVmapWrapper(VmapWrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: PipelineEnv, episode_length: int, batch_size: Optional[int] = None):
    super().__init__(env, batch_size)
    self.episode_length = episode_length
    
  def rollout(self, init_pipeline_state_batch: base.State, action_batch_traj: jax.Array):
    
    if init_pipeline_state_batch.q.ndim ==1:
            init_pipeline_state_batch=do_batching(init_pipeline_state_batch, self.batch_size)
            
    def f(state_batch, action_batch):
      obs = jax.vmap(get_obs)(state_batch)
      model_input = jnp.concat((obs,action_batch),axis=-1)
      next_state_batch = self.step(state_batch, action_batch)
      
      return next_state_batch, model_input
  
    final_state_batch, model_input_traj = jax.lax.scan(f, init_pipeline_state_batch, action_batch_traj)
    
    return model_input_traj
