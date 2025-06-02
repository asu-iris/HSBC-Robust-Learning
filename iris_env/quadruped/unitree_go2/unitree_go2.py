from typing import Callable, Dict, Optional, Tuple

from brax.mjx import base
from brax.envs.base import PipelineEnv, Wrapper
from brax.io import mjcf
from brax import math

import jax
from jax import numpy as jnp
import mujoco
import pathlib,os


class UnitreeGo2(PipelineEnv):

    def __init__(self, env_param, frame_skip=10, timestep=0.002, **kwargs):

        # env param
        self.param = env_param

        # load model
        filedir = pathlib.Path(__file__).resolve().parent

        # load model
        path = os.path.join(filedir,'models/scene_mjx.xml')
        self.mj_model = mujoco.MjModel.from_xml_path(path)
        self.mj_model.opt.timestep = timestep

        # wrapper of brax
        sys = mjcf.load_model(self.mj_model)
        super().__init__(sys=sys, backend='mjx', n_frames=frame_skip, **kwargs)

        self.action_dim = 12


    def reset(self, qpos: jax.Array, qvel: jax.Array) -> base.State:
        pipeline_state = self.pipeline_init(qpos, qvel)
        return pipeline_state

    def step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """Runs one timestep of the environment's dynamics."""
        assert pipeline_state is not None
        target_jpos = pipeline_state.q[-self.action_dim:] + action

        pipeline_state = self.pipeline_step(pipeline_state, target_jpos)
        return pipeline_state

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""
        position = pipeline_state.q
        velocity = pipeline_state.qd

        foots_z = pipeline_state.site_xpos[1:5,2]

        return jnp.concatenate((position, foots_z, velocity))


def do_batching(state: base.State, batch_size: int):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=batch_size, axis=0)
    return jax.tree_util.tree_map(replicate_field, state)


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
      obs = self._get_obs(state_batch)
      #qfrc = self._get_qfrc(state_batch,action_batch)
      model_input = jnp.concat((obs,action_batch),axis=-1)
      next_state_batch = self.step(state_batch, action_batch)
      
      return next_state_batch, model_input
  
    final_state_batch, model_input_traj = jax.lax.scan(f, init_pipeline_state_batch, action_batch_traj)
    
    return model_input_traj

def costFn(state: base.State, action: jax.Array):

    # define the goal object pose
    robot_target_quat = math.quat_rot_axis(
        axis=jnp.array([0., 0, 1.]), angle=jnp.pi/2)
    robot_target_pos = jnp.array([2.0, 2.0, 0.45])
    robot_target_height=0.45


    # get the object pose
    robot_pos = state.q[0:3]
    robot_quat = state.q[3:7]
    robot_height=robot_pos[-1]

    # position cost
    # cost_quat = 1 - jnp.dot(robot_quat, robot_target_quat) ** 2
    cost_pos = jnp.linalg.norm(robot_target_pos-robot_pos)**2

    # target height cost
    cost_height= jnp.linalg.norm(robot_target_height-robot_height)**2


    # control cost
    cost_control = jnp.dot(action, action)

    return 5*cost_height+0.01*cost_control
