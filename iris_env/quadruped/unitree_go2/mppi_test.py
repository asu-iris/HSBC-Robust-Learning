import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html
from mujoco import mjx
import mujoco

from unitree_go2 import UnitreeGo2, RolloutVmapWrapper
from planning.jax_mppi import MPPI


class env_param:

    def __init__(self):
        
        # system initial condition
        self.init_robot_joints = np.zeros(12)
        self.init_robot_pos=np.array([-0.0, -0.000, 0.450])
        self.init_robot_quat=np.array([1.0, 0.0, 0, 0])
        self.init_robot_qpos=np.hstack((self.init_robot_pos, self.init_robot_quat))


# init single environment
param = env_param()
env = UnitreeGo2(param, frame_skip=5)

# reset
init_qpos=jnx.array(np.concatenate((param.init_robot_qpos, param.init_robot_joints)))
init_qvel=jnx.zeros(18)
env_state=env.reset(init_qpos, init_qvel)


# jit everything
env_reset = jax.jit(env.reset)
env_step = jax.jit(env.step)


# init vector environment for planner
batch_size = 512
horizon=5
batch_rollouter=RolloutVmapWrapper(UnitreeGo2(param, frame_skip=2, timestep=0.02), batch_size=batch_size, horizon=horizon)


# create the mppi_planner
planner_cfg={'noise_sigma': 2, 'temperature':1.0}
planner=MPPI(batch_rollouter=batch_rollouter, cfg=planner_cfg)
planner_state=planner.reset()
planner_act=jax.jit(planner.get_action)



env_traj=[]
rollout_length= 1000
for t in range(rollout_length):
    
    
    st=time.time()
    best_action, planner_state=planner_act(env_state,planner_state)
    print('mpc time:', time.time()-st)
    
    # env step forward
    env_state = env_step(env_state, best_action)
    env_traj.append(env_state)
    
    
with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_traj))
    
