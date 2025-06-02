from typing import Callable, Dict, Optional, Tuple

from brax import base
from brax.envs.base import PipelineEnv, Wrapper
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco

class Allegro(PipelineEnv):

    def __init__(self, env_param, **kwargs):

        # env param
        self.param = env_param

        # load model
        path = 'dexhand/allegro/models/allegro_right_hand.xml'
        self.mj_model=mujoco.MjModel.from_xml_path(path)
        sys = mjcf.load_model(self.mj_model)
        super().__init__(sys=sys, backend='mjx', n_frames=self.param.frame_skip, **kwargs)


    def reset(self, qpos: jax.Array, qvel: jax.Array) -> base.State:
        pipeline_state = self.pipeline_init(qpos, qvel)
        return pipeline_state

    def step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """Runs one timestep of the environment's dynamics."""
        assert pipeline_state is not None
        target_jpos = pipeline_state.q + action

        pipeline_state = self.pipeline_step(pipeline_state, target_jpos)
        return pipeline_state

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd
        return jnp.concatenate((position, velocity))



def do_batching(state: base.State, batch_size: int):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=batch_size, axis=0)
    return jax.tree_util.tree_map(replicate_field,state)



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





def reward(state:base.State, action: jax.Array):
   pass
   return 0.0 

