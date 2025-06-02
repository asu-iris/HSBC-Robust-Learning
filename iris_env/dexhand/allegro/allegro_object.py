from typing import Callable, Dict, Optional, Tuple

from brax.mjx import base
from brax.envs.base import PipelineEnv, Wrapper
from brax.io import mjcf
from brax import math

import jax
from jax import numpy as jnp
import mujoco
import os,pathlib


class AllegroObject(PipelineEnv):

    def __init__(self, env_param, frame_skip=10, timestep=0.002, **kwargs):

        # env param
        self.param = env_param
        self.object_name = env_param.object_name

        # load model
        filedir = pathlib.Path(__file__).resolve().parent

        # load model
        path = os.path.join(filedir,'models/allegro_'+self.object_name+'.xml')

        self.mj_model = mujoco.MjModel.from_xml_path(path)
        self.mj_model.opt.timestep = timestep

        # wrapper of brax
        sys = mjcf.load_model(self.mj_model)
        super().__init__(sys=sys, backend='mjx', n_frames=frame_skip, **kwargs)

        self.action_dim = 16
        
    

    def reset(self, qpos: jax.Array, qvel: jax.Array) -> base.State:
        pipeline_state = self.pipeline_init(qpos, qvel)
        return pipeline_state

    def step(self, pipeline_state: base.State, action: jax.Array) -> base.State:
        """Runs one timestep of the environment's dynamics."""
        assert pipeline_state is not None
        target_jpos = pipeline_state.q[0:self.action_size] + action

        pipeline_state = self.pipeline_step(pipeline_state, target_jpos)
        return pipeline_state


def do_batching(state: base.State, batch_size: int):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=batch_size, axis=0)
    return jax.tree_util.tree_map(replicate_field, state)


def get_obs(pipeline_state: base.State) -> jax.Array:
    """Returns the environment observations."""
    qpos=pipeline_state.q
    qvel=pipeline_state.qd

    pos_ff_tip = pipeline_state.site_xpos[0]
    pos_mf_tip = pipeline_state.site_xpos[1]
    pos_rf_tip = pipeline_state.site_xpos[2]
    pos_th_tip = pipeline_state.site_xpos[3]

    return jnp.concat((qpos, qvel, pos_ff_tip, pos_mf_tip, pos_rf_tip, pos_th_tip),axis=-1)


class VmapWrapper(Wrapper):
    """Vectorizes bBrax env."""

    def __init__(self, env: PipelineEnv, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, qpos_batch: jax.Array, qvel_batch: jax.Array) -> base.State:
        if qpos_batch.ndim == 1:
            qpos_batch = do_batching(qpos_batch, self.batch_size)
        if qvel_batch.ndim == 1:
            qvel_batch = do_batching(qvel_batch, self.batch_size)

        return jax.vmap(self.env.reset)(qpos_batch, qvel_batch)

    def step(self, pipeline_state_batch: base.State, action_batch: jax.Array) -> base.State:
        assert pipeline_state_batch.q.ndim>1
        if action_batch.ndim == 1:
            action_batch = do_batching(action_batch, self.batch_size)

        return jax.vmap(self.env.step)(pipeline_state_batch, action_batch)


class RolloutVmapWrapper(VmapWrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: PipelineEnv, horizon: int, batch_size: Optional[int] = None):
    super().__init__(env, batch_size)
    self.horizon = horizon
    

  def rollout(self, init_pipeline_state_batch: base.State, action_batch_traj: jax.Array):
    
    if init_pipeline_state_batch.q.ndim ==1:
            init_pipeline_state_batch=do_batching(init_pipeline_state_batch, self.batch_size)
            
    def f(state_batch, action_batch):
      obs_batch=jax.vmap(get_obs)(state_batch)
      obs_action_batch=jnp.concat((obs_batch,action_batch),axis=-1)
      next_state_batch = self.step(state_batch, action_batch)
      return next_state_batch, obs_action_batch
  
    final_state_batch, obs_action_batch_traj = jax.lax.scan(f, init_pipeline_state_batch, action_batch_traj)
    
    return obs_action_batch_traj




# def costFn(state: base.State, action: jax.Array):

#     # define the goal object pose
#     obj_target_quat = math.quat_rot_axis(axis=jnp.array([0., 0, 1.]), angle=1.9*jnp.pi/2)
#     obj_target_pos=jnp.array([0.05, -0.00, 0.05])

#     # get the object pose
#     obj_pos = state.q[-7:-4]
#     obj_quat = state.q[-4:]


#     # object orientation cost
#     cost_quat = 1 - jnp.dot(obj_quat, obj_target_quat) ** 2
#     cost_pos=jnp.linalg.norm(obj_target_pos-obj_pos)**2

#     # control cost
#     cost_control = jnp.dot(action, action)

#     # contact cost
#     pos_ff_tip = state.site_xpos[0]
#     pos_mf_tip = state.site_xpos[1]
#     pos_rf_tip = state.site_xpos[2]
#     pos_th_tip = state.site_xpos[3]
#     cost_contact = jnp.linalg.norm(pos_ff_tip-obj_pos)**2+jnp.linalg.norm(
#         pos_mf_tip-obj_pos)**2+jnp.linalg.norm(pos_rf_tip-obj_pos)**2+jnp.linalg.norm(pos_th_tip-obj_pos)**2


#     # grasp cost
#     obj_v0 = (pos_ff_tip - obj_pos)
#     obj_v1 = (pos_rf_tip - obj_pos)
#     obj_v2 = (pos_th_tip - obj_pos)
#     tri_vec=obj_v0 / jnp.linalg.norm(obj_v0) + obj_v1 / jnp.linalg.norm(obj_v1) + obj_v2 / jnp.linalg.norm(obj_v2)
#     cost_grasp = jnp.dot(tri_vec,tri_vec)



#     return 50*cost_quat+10*cost_pos+0.01*cost_control+ 1000*cost_contact




