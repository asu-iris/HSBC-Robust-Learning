import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html

from allegro_object import AllegroObject, VmapWrapper


# set the environment 

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'




class env_param:

    def __init__(self):
        
        # frame skip
        self.frame_skip=10

        self.object_name='cube'

        # system initial condition
        self.init_robot_qpos = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])

        self.init_object_pos=np.array([-0.03, -0.01, 0.03])
        self.init_object_quat=np.array([1.0, 0.0, 0, 0])
        self.init_object_qpos=np.hstack((self.init_object_pos, self.init_object_quat))


# init single environment
param = env_param()
env = AllegroObject(param)
print(env.sys.opt.timestep)

# reset
init_qpos=jnx.array(np.concatenate((param.init_robot_qpos, param.init_object_qpos)))
init_qvel=jnx.zeros(22)
state=env.reset(init_qpos, init_qvel)


# init vector environment
planner=AllegroObject(param, timestep=0.01)
planner_num = 512
planner = VmapWrapper(planner, batch_size=planner_num)
planner_state=planner.reset(init_qpos, init_qvel)
print(planner.sys.opt.timestep)

# jit everything
reset = jax.jit(env.reset)
step = jax.jit(env.step)
planner_reset=jax.jit(planner.reset)
planner_step=jax.jit(planner.step)

# control action
action=0.1*jnx.ones((16,))

traj_state = []
for t in range(20):
    st=time.time()
    state=step(state,action)
    print('single env step time:',time.time()-st)

    st=time.time()
    planner_state=planner_step(planner_state,action)
    print('vector env step time:',time.time()-st)
        
    traj_state.append(state)

with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, traj_state))




