import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html
from mujoco import mjx
import mujoco

from dexhand.allegro.allegro_object import AllegroObject
from planning.jax_mppi import MPPI


class env_param:

    def __init__(self):

        self.object_name = 'bunny'
        self.init_robot_qpos = np.array([
            0.0, -0.03, -0.01, 0,
            0.0, -0.03, -0.01, 0,
            0.0, -0.03, -0.01, 0,
            0.9, 0.9, 0.8, 0.7,
        ])

        self.init_object_pos = np.array([0.05, -0.02, 0.05])
        self.init_object_quat = np.array([1.0, 0.0, 0, 0])
        self.init_object_qpos = np.hstack(
            (self.init_object_pos, self.init_object_quat))


# init single environment
param = env_param()
env = AllegroObject(param, frame_skip=10)

# reset
init_qpos = jnx.array(np.concatenate(
    (param.init_robot_qpos, param.init_object_qpos)))
init_qvel = jnx.zeros(22)
env_state = env.reset(init_qpos, init_qvel)

print(env_state.contact.dim.size)
for i in range(env_state.contact.dim.size):
  geom1_id=env_state.contact.geom[i,0]
  geom2_id=env_state.contact.geom[i,1]
  print(mjx.id2name(env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,geom1_id), ' and ',
        mjx.id2name(env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,geom2_id),':',
        env_state.contact.dist[i])
  
with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, [env_state],height=960))