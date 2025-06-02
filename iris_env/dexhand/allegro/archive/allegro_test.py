import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html

from allegro import Allegro, VmapWrapper

class env_param:

    def __init__(self):
        # frame skip
        self.frame_skip=10

        # system initial condition
        self.init_robot_qpos = np.array([
            0.125, 1.13, 1.45, 1.24,
            -0.02, 0.445, 1.17, 1.5,
            -0.459, 1.54, 1.11, 1.23,
            0.638, 1.85, 1.5, 1.26
        ])


# init single environment
param = env_param()
env = Allegro(param)
# reset
init_qpos=jnx.array(param.init_robot_qpos)
init_qvel=jnx.zeros(16)
state=env.reset(init_qpos, init_qvel)


# init vector environment
planner=Allegro(param)
planner_num = 512
planner = VmapWrapper(planner, batch_size=planner_num)
planner_state=planner.reset(init_qpos, init_qvel)


# jit everything
reset = jax.jit(env.reset)
step = jax.jit(env.step)
planner_reset=jax.jit(planner.reset)
planner_step=jax.jit(planner.step)

# control action
action=0.1*jnx.ones((16,))

traj_state = []
for t in range(10):
    st=time.time()
    state=step(state,action)
    print('single env step time:',time.time()-st)

    st=time.time()
    planner_state=planner_step(planner_state,action)
    print('vector env step time:',time.time()-st)
        
    traj_state.append(state)

with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, traj_state))




