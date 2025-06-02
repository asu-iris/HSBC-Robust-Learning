
"""
This is an adaptation of repo: https://github.com/jlehtomaa/jax-mppi
by jlehtomaa (https://github.com/jlehtomaa)
"""

import numpy as np
import jax.numpy as jnp
import jax

from brax.mjx import base


class MPPI:
    def __init__(self, batch_rollouter, rollout_vmapcost_fn, cfg, warm_state=False):
        
        # this is the open loop batch_rollouter
        self.batch_rollouter=batch_rollouter
        self.rollout_vmapcost_fn=rollout_vmapcost_fn
        self.horizon=batch_rollouter.horizon
        self.batch_size=batch_rollouter.batch_size
        self.action_dim=batch_rollouter.action_dim
        
        # configure
        self.cfg=cfg
        
        # MPC warm start
        self.warm_start=warm_state
        
        self.plan_size=(self.horizon, self.batch_size, self.action_dim)
        self.reset()

    def reset(self):
        """Reset the control trajectory at the start of an episode.
        """
        return {'plan':jnp.zeros(self.plan_size), "rng_key":jax.random.PRNGKey(0)}

    def _sample_noise(self, rng_key):
        """Get noise for constructing perturbed action sequences.
        """
        return jax.random.normal(rng_key, shape=self.plan_size) * self.cfg["noise_sigma"]

    def get_action(self, env_state: base.State, planner_state: jax.Array):
        """Get the next optimal action based on current state observation.
        """

        acts = planner_state['plan'] + self._sample_noise(planner_state['rng_key']) # (horizon, n_samples, act_dim)
        # acts = np.clip(acts, self.act_min, self.act_max)

        obs_action_batch_traj=self.batch_rollouter.rollout(env_state,acts) #(horizon, num_samples,obs_dim+act_dim)
        cost_batch_traj=self.rollout_vmapcost_fn(obs_action_batch_traj) #(horizon, num_samples, 1)
        cost_batch=jnp.sum(cost_batch_traj,axis=0) #(num_samples, 1)

        exp_costs = jnp.exp(self.cfg["temperature"] * (np.min(cost_batch) - cost_batch))
        denom = jnp.sum(exp_costs) + 1e-10

        weighted_inputs = exp_costs[jnp.newaxis, :,  jnp.newaxis] * acts
        sol = np.sum(weighted_inputs, axis=1) / denom # (horizon, act_dim)

        if self.warm_start==True:
            # Update the initial plan, and only return the first action 
            shifted_sol = jnp.roll(sol, shift=-1, axis=0)
            shifted_sol = shifted_sol.at[-1].set(sol[-1]) # Repeat the last step.
            
            # update the planner state
            planner_state['plan']=shifted_sol[:,jnp.newaxis,:] # (horizon, 1, act_dim)
            planner_state['rng_key']=jax.random.split(planner_state['rng_key'])[1]
        

        return sol[0], planner_state