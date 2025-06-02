import numpy as np
import jax.numpy as jnx
import time
import jax
from brax.io import html

from unitree_go2 import UnitreeGo2, RolloutVmapWrapper


# set the environment 
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=true'


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


# init vector environment for planner
planner=UnitreeGo2(param, frame_skip=2, timestep=0.02)
batch_size = 512
horizon=5
planner=RolloutVmapWrapper(planner, batch_size=batch_size, horizon=horizon)


# jit everything
env_reset = jax.jit(env.reset)
env_step = jax.jit(env.step)
planner_cost_episode=jax.jit(planner.rollout_accumcost)


env_traj=[]
rollout_length= 100
for t in range(rollout_length):
    
    # generate randome action sequences
    key = jax.random.PRNGKey(t)
    
    st=time.time()
    plan_action_batch_traj = 2*(jax.random.uniform(key, shape=(horizon, batch_size, env.act_dim))-0.5)
    plan_cost_batch=planner_cost_episode(env_state,plan_action_batch_traj)
    
    # pick the winer action sequence
    best_action = plan_action_batch_traj[0,jax.numpy.argmin(plan_cost_batch)]
    print('mpc time:', time.time()-st)
    
    # env step forward
    env_state = env_step(env_state, best_action)
    env_traj.append(env_state)
    
    
with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_traj))
    

