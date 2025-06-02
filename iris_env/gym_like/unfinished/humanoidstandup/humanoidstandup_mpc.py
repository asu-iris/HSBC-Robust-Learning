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



action_dim = 17
mpc_dyn_num = 512
mpc_dyn = envs.create(env_name='humanoidstandup', batch_size=mpc_dyn_num,
                      episode_length=None,
                      auto_reset=False,
                      backend='mjx')
mpc_dyn_state = mpc_dyn.reset(rng=jnp.array(random.PRNGKey(0)))
mpc_horizon = 10
mpc_dyn_step_fn = jax.jit(mpc_dyn.step)


env = envs.create(env_name='humanoidstandup',
                  episode_length=None,
                  auto_reset=False,
                  backend='mjx')
env_state = env.reset(rng=jnp.array(random.PRNGKey(0)))
env_step_fn = jax.jit(env.step)

@jax.jit
def batching_anything(state):
    def replicate_field(x):
        return jnp.repeat(x[None, ...], repeats=mpc_dyn_num, axis=0)
    return jax.tree_util.tree_map(replicate_field,state)



env_rollout_length = 500
env_rollout_pipeline_state = []
for t in range(env_rollout_length):

    # mpc planning and selection
    # set the current mpc state with the env state
    batch_env_state=batching_anything(env_state)
    mpc_dyn_state=mpc_dyn_state.tree_replace({"pipeline_state": batch_env_state.pipeline_state})

    
    # generate rand action sequences
    key = jax.random.PRNGKey(t)
    mpc_rand_actions = 1*(jax.random.uniform(
        key, shape=(mpc_dyn_num, mpc_horizon, action_dim))-0.5)
    mpc_rewards = jax.numpy.zeros(mpc_dyn_num)
    st = time.time()
    for k in range(mpc_horizon):
        mpc_dyn_state = mpc_dyn_step_fn(
            mpc_dyn_state, mpc_rand_actions[:, k, :])
        mpc_rewards = mpc_rewards+mpc_dyn_state.reward
    print('mpc_time:', time.time()-st)
    
    # pick the best action
    best_action = mpc_rand_actions[jax.numpy.argmax(mpc_rewards)][0]
    st = time.time()
    env_state = env_step_fn(env_state, best_action)
    
    # safe for readering
    env_rollout_pipeline_state.append(env_state.pipeline_state)


#   visualization
with open("simulation_output.html", "w") as f:
    f.write(html.render(env.sys, env_rollout_pipeline_state))
