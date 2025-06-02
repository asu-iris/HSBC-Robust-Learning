import jax
import jax.numpy as jnp
from jax import random

import brax
from brax import envs
from brax.io import metrics
import matplotlib.pyplot as plt
import numpy as np
import time
from brax.io import html
from brax.envs.wrappers import training

# import ant environment (modified by IRIS lab)
import humanoid

# create environment
env=humanoid.Humanoid()
env_state = env.reset(rng=jnp.array(random.PRNGKey(0)))

env_step_fn = jax.jit(env.step)


# create MPC planner with a batch size
planner=humanoid.Humanoid()
planner_num = 512
planner = training.VmapWrapper(planner, batch_size=planner_num)
planner_state = planner.reset(rng=jnp.array(random.PRNGKey(0)))
planner_horizon = 10
planner_step_fn = jax.jit(planner.step)

# add a batch dim for any env/planner state
@jax.jit
def batch_state(state):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=planner_num, axis=0)
    return jax.tree_util.tree_map(replicate_field,state)


# env rolllout using sampling-based mpc controller
env_rollout_length = 500
env_pipeline_state_traj = []
for t in range(env_rollout_length):

    # mpc planning and selection
    # set the planner state using current env state
    planner_state=batch_state(env_state)
    # planner_state=planner_state.tree_replace({"pipeline_state": planner_state.pipeline_state})

    # generate randome action sequences
    key = jax.random.PRNGKey(t)
    planner_actions = 2*(jax.random.uniform(key, shape=(planner_num, planner_horizon, planner.action_size))-0.5)
    planner_sum_rewards = jax.numpy.zeros(planner_num)
    st = time.time()
    for k in range(planner_horizon):
        planner_state = planner_step_fn(planner_state, planner_actions[:, k, :])
        planner_sum_rewards = planner_sum_rewards+planner_state.reward
    print('mpc_time:', time.time()-st)
    
    # pick the winer action sequence
    best_actions = planner_actions[jax.numpy.argmax(planner_sum_rewards)]
    st = time.time()
    env_state = env_step_fn(env_state, best_actions[0])
    
    # save evn traj
    env_pipeline_state_traj.append(env_state.pipeline_state)


#   visualization
with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_pipeline_state_traj))
