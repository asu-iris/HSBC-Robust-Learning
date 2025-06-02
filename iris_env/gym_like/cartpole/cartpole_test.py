import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html

from cartpole import Cartpole, VmapWrapper, get_obs

class env_param:

    def __init__(self):
        # frame skip
        self.frame_skip=5


# init single environment
param = env_param()
env = Cartpole(param, frame_skip=5, timestep=0.01)
# reset
init_qpos_arr = np.zeros(2)
init_qpos_arr[0] = np.random.uniform(-1.0,1.0)
init_qpos_arr[1] = np.random.uniform(-np.pi/2,np.pi/2)
init_qvel_arr = np.random.uniform([-1.,-1.],[1.,1.])
init_qpos=jnx.array(init_qpos_arr)
init_qvel=jnx.array(init_qvel_arr)
state=env.reset(init_qpos, init_qvel)


# init vector environment
planner=Cartpole(param, frame_skip=5, timestep=0.01)
planner_num = 256
planner = VmapWrapper(planner, batch_size=planner_num)
planner_state=planner.reset(init_qpos, init_qvel)

print(env._get_obs(state))

# you can either do:
# print(jax.vmap(get_obs)(planner_state))
# # or 
# print(planner._get_obs(planner_state))

# jit everything
reset = jax.jit(env.reset)
step = jax.jit(env.step)
planner_reset=jax.jit(planner.reset)
planner_step=jax.jit(planner.step)

# control action
action=0.1*jnx.ones((1,))

traj_state = []
for t in range(100):
    st=time.time()
    state=step(state,action)
    print('single env step time:',time.time()-st)

    st=time.time()
    planner_state=planner_step(planner_state,action)
    print('vector env step time:',time.time()-st)
    
    traj_state.append(state)

with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, traj_state))





