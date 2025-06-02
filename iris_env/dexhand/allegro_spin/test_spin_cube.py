import numpy as np
import jax.numpy as jnp
import time
import jax
from brax.io import html
from mujoco import mjx
import mujoco
from brax import math

import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from spin_object import AllegroObject, RolloutVmapWrapper
from planning.jax_mppi import MPPI



class env_param:

    def __init__(self):

        self.object_name = 'cube'
        self.init_robot_qpos = np.array([
            0.183885, 1.13392, 1.10524, 0.361177,
            0.0865814, 0.771298, 1.00078, 0.888168,
            0.120795, 0.945232, 1.46352, 0.0791672,
            0.841227, 1.3188, 1.01457, 0.6449,
        ])

        self.init_object_pos = np.array([-0.011549, -0.0224281, 0.101794])
        self.init_object_quat = np.array(
            [-0.456287, -0.822746, 0.312223, -0.131939])
        self.init_object_qpos = np.hstack(
            (self.init_object_pos, self.init_object_quat))


# define cost function
def costFn(obs: jax.Array, action: jax.Array):

    # initial finger pose and object pos and quat
    param = env_param()
    robot_init_qpos = param.init_robot_qpos
    obj_init_pos = param.init_object_pos

    # desired angular velocity
    obj_target_vquat=jnp.array([0,0,1.0])


    # parse the observation
    qpos = obs[0:23]
    qvel = obs[23:22+23]

    # robot pose
    robot_qpos = qpos[0:16]

    # current object pose and finger pose
    obj_pos = qpos[-7:-4]
    obj_quat = qpos[-4:]

    # obj velocity
    obj_vel = qvel[-6:-3]
    obj_vquat = qvel[-3:]

    # fingertip position
    robot_ff_tip = obs[45:48]
    robot_mf_tip = obs[48:51]
    robot_rf_tip = obs[51:54]
    robot_th_tip = obs[54:57]


    # write the cost terms, please refer to https://github.com/HaozhiQi/penspin/blob/f6f2c87ed6a427edd8929751e8feb4bc1744b197/penspin/tasks/allegro_hand_hora.py#L520

    cost_obj_linvel = jnp.linalg.norm(obj_vel, ord=1)
    cost_obj_rotate = jnp.linalg.norm(obj_vquat-obj_target_vquat)
    cost_obj_position = jnp.linalg.norm(obj_pos-obj_init_pos, ord=1)

    cost_robot_diff_qpos = jnp.linalg.norm(robot_qpos-robot_init_qpos)**2
    cost_robot_effort = jnp.dot(action, action)

    cost_contact = jnp.linalg.norm(robot_ff_tip-obj_pos)**2+jnp.linalg.norm(
        robot_mf_tip-obj_pos)**2+jnp.linalg.norm(robot_rf_tip-obj_pos)**2+jnp.linalg.norm(robot_th_tip-obj_pos)**2

    return 0*cost_obj_linvel+1*cost_obj_rotate+2000*cost_obj_position +\
        10*cost_robot_diff_qpos+0.01*cost_robot_effort +\
        2000*cost_contact


def cost_fn(obs_action: jax.Array):
    obs = obs_action[0:57]
    action = obs_action[-16:]
    return costFn(obs, action)


vmapcost_fn = jax.vmap(cost_fn)
rollout_vmapcost_fn = jax.vmap(vmapcost_fn)


# init single environment
param = env_param()
env = AllegroObject(param, frame_skip=2)

# reset
init_qpos = jnp.array(np.concatenate(
    (param.init_robot_qpos, param.init_object_qpos)))
init_qvel = jnp.zeros(22)
env_state = env.reset(init_qpos, init_qvel)


# jit everything
env_reset = jax.jit(env.reset)
env_step = jax.jit(env.step)

# init vector environment for planner
batch_size = 1024
horizon = 8
batch_rollouter = RolloutVmapWrapper(AllegroObject(
    param, frame_skip=2), batch_size=batch_size, horizon=horizon)


# create the mppi_planner
planner_cfg = {'noise_sigma': 0.5, 'temperature': 1.0}
planner = MPPI(batch_rollouter=batch_rollouter,
               rollout_vmapcost_fn=rollout_vmapcost_fn, cfg=planner_cfg)
planner_state = planner.reset()
planner_act = jax.jit(planner.get_action)


env_traj = []
rollout_length = 500
for t in range(rollout_length):

    st = time.time()
    best_action, planner_state = planner_act(env_state, planner_state)
    print('mpc time:', time.time()-st)

    # env step forward
    env_state = env_step(env_state, best_action)
    env_traj.append(env_state)


with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_traj,height=960))
