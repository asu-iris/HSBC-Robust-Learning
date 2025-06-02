import numpy as np
import jax.numpy as jnp
import time
import jax
from brax.io import html
from mujoco import mjx
import mujoco
from brax import math


import sys
from pathlib import Path
path = Path('./')
sys.path.append(str(path.absolute()))
print(sys.path)

from allegro_object import AllegroObject, RolloutVmapWrapper
from planning.jax_mppi import MPPI



class env_param:

    def __init__(self):
        
        self.object_name='bunny'
        self.init_robot_qpos = np.array([
            0.0, -0.03, -0.01, 0,
            0.0, -0.03, -0.01, 0,
            0.0, -0.03, -0.01, 0,
            0.9, 0.9, 0.8, 0.7,
        ])

        self.init_object_pos=np.array([0.05, -0.0, 0.05])
        self.init_object_quat=np.array([1.0, 0.0, 0, 0])
        self.init_object_qpos=np.hstack((self.init_object_pos, self.init_object_quat))




# define cost function 
def costFn(obs: jax.Array, action: jax.Array):

    # define the goal object pose
    obj_target_quat = math.quat_rot_axis(
        axis=jnp.array([0., 0, 1.]), angle=1.9*jnp.pi/2)
    obj_target_pos = jnp.array([0.05, -0.00, 0.05])

    # get the object pose
    qpos = obs[0:23]
    qvel = obs[23:22+23]
    obj_pos = qpos[-7:-4]
    obj_quat = qpos[-4:]

    # object orientation cost
    cost_quat = 1 - jnp.dot(obj_quat, obj_target_quat) ** 2
    cost_pos = jnp.linalg.norm(obj_target_pos-obj_pos)**2

    # control cost
    cost_control = jnp.dot(action, action)
    # contact cost
    pos_ff_tip = obs[45:48]
    pos_mf_tip = obs[48:51]
    pos_rf_tip = obs[51:54]
    pos_th_tip = obs[54:57]
    cost_contact = jnp.linalg.norm(pos_ff_tip-obj_pos)**2+jnp.linalg.norm(
        pos_mf_tip-obj_pos)**2+jnp.linalg.norm(pos_rf_tip-obj_pos)**2+jnp.linalg.norm(pos_th_tip-obj_pos)**2

    return 50*cost_quat+10*cost_pos+0.01*cost_control + 1000*cost_contact

def cost_fn(obs_action: jax.Array):
    obs = obs_action[0:57]
    action = obs_action[-16:]
    return costFn(obs, action)

vmapcost_fn = jax.vmap(cost_fn)
rollout_vmapcost_fn = jax.vmap(vmapcost_fn)





# init single environment
param = env_param()
env = AllegroObject(param, frame_skip=10)

# reset
init_qpos = jnp.array(np.concatenate(
    (param.init_robot_qpos, param.init_object_qpos)))
init_qvel = jnp.zeros(22)
env_state = env.reset(init_qpos, init_qvel)


# jit everything
env_reset = jax.jit(env.reset)
env_step = jax.jit(env.step)

# init vector environment for planner
batch_size = 512
horizon = 5
batch_rollouter = RolloutVmapWrapper(AllegroObject(
    param, frame_skip=2, timestep=0.005), batch_size=batch_size, horizon=horizon)


# create the mppi_planner
planner_cfg = {'noise_sigma': 0.3, 'temperature': 1.0}
planner = MPPI(batch_rollouter=batch_rollouter,
               rollout_vmapcost_fn=rollout_vmapcost_fn, cfg=planner_cfg)
planner_state = planner.reset()
planner_act = jax.jit(planner.get_action)


env_traj = []
rollout_length = 50
for t in range(rollout_length):

    st = time.time()
    best_action, planner_state = planner_act(env_state, planner_state)
    print('mpc time:', time.time()-st)

    # env step forward
    env_state = env_step(env_state, best_action)
    env_traj.append(env_state)


with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_traj,height=960))
    
