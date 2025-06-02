from iris_env.gym_like.cartpole.cartpole import Cartpole, RolloutVmapWrapper
import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j

from brax.io import html
import time

class env_param:
    def __init__(self):
        # frame skip
        self.frame_skip=5

class Cartpole_MPC(object):
    def __init__(self, Horizon=20, rollout_length=100, planner_num=256, sigma=0.4) -> None:
        self.Horizon = Horizon
        self.rollout_length = rollout_length
        self.planner_num = planner_num
        self.sigma = sigma

        param = env_param()
        self.env = Cartpole(env_param=None,frame_skip=5,timestep=0.01)

        # jit everything
        self.reset = jax.jit(self.env.reset)
        self.step = jax.jit(self.env.step)

        self.planner = RolloutVmapWrapper(self.env, batch_size=self.planner_num, episode_length=self.Horizon)
        self.planner_rollout_episode=jax.jit(self.planner.rollout)



    def generate_traj_cartpole(self,torch_model,render = False, init_pos = None, init_vel = None,  lam_mppi = 1.0, noiselevel = 0.0 , filename = 'simulation_output.html'):
        #planner:
        self.mppi = MY_MPPI(planner_num=self.planner_num,Horizon=self.Horizon,action_dim=self.env.action_dim,
                            rollout_fn=self.planner_rollout_episode,reward_fn=torch_model,lam=lam_mppi,sigma=self.sigma)


        if init_pos is not None:
            init_qpos=jnx.array(init_pos)
        else:
            init_qpos_arr = np.zeros(2)
            init_qpos_arr[0] = 0.01 * np.random.randn()
            init_qpos_arr[1] = np.pi + 0.01 * np.random.randn()
            init_qpos=jnx.array(init_qpos_arr)

        if init_vel is not None:
            init_qvel=jnx.array(init_vel)
        else:
            init_qvel_arr = 0.01 * np.random.randn(2)
            init_qvel=jnx.array(init_qvel_arr)

        state=self.env.reset(init_qpos, init_qvel)

        sample_center = jnx.zeros(self.env.action_dim)
        env_obss=[]
        env_actions=[]
        traj_state = []

        seed = int(time.time())
        key = jax.random.key(seed)

        for t in range(self.rollout_length):
            traj_state.append(state)
            #planning
            qpos,qvel = state.q,state.qd
            _,key = jax.random.split(key,num=2)
            action = self.mppi.command(state,key)

            action += noiselevel * jax.random.normal(key,shape=(self.env.action_dim,))
            action = jnx.clip(action,-1,1)
            #sample_center = action
            #step
            obs = self.env._get_obs(state)
            if t%10==0:
                print('obs',obs)
            env_obss.append(obs)
            env_actions.append(action)

            state=self.step(state,action)

        if render:
            with open(filename, "w") as f:
                f.write(html.render(self.env.sys, traj_state))

        return jnx.stack(env_obss,axis=0),jnx.stack(env_actions,axis=0)
    
    def generate_random_traj(self,init_pos = None, init_vel = None):
        if init_pos is not None:
            init_qpos=jnx.array(init_pos)
        else:
            init_qpos_arr = np.zeros(2)
            init_qpos_arr[0] = 0.01 * np.random.randn()
            init_qpos_arr[1] = np.pi + 0.01 * np.random.randn()
            init_qpos=jnx.array(init_qpos_arr)

        if init_vel is not None:
            init_qvel=jnx.array(init_vel)
        else:
            init_qvel_arr = 0.01 * np.random.randn(2)
            init_qvel=jnx.array(init_qvel_arr)

        state=self.env.reset(init_qpos, init_qvel)

        env_obss=[]
        env_actions=[]
        traj_state = []

        seed = int(time.time())
        key = jax.random.key(seed)

        for t in range(self.rollout_length):
            traj_state.append(state)
            #planning
            qpos,qvel = state.q,state.qd
            if t%10==9:
                _,key = jax.random.split(key,num=2)

            action = jax.random.uniform(key,shape=(self.env.action_dim,), minval=-1, maxval=1)

            #sample_center = action
            #step
            obs = self.env._get_obs(state)
            if t%10==0:
                print('obs',obs)
            env_obss.append(obs)
            env_actions.append(action)

            state=self.step(state,action)

        return jnx.stack(env_obss,axis=0),jnx.stack(env_actions,axis=0)
    
class MY_MPPI(object):
    def __init__(self,planner_num,Horizon,action_dim,rollout_fn,reward_fn,lam,sigma):
        self.planner_num = planner_num
        self.Horizon = Horizon
        self.action_dim = action_dim
        self.rollout_fn = rollout_fn
        self.reward_fn = reward_fn
        self.lam=lam
        self.sigma=sigma

        self.nominal_traj = jnx.zeros((self.Horizon,self.action_dim))

    def reset(self):
        self.nominal_traj = jnx.zeros((self.Horizon,self.action_dim))

    def command(self,state,key):
        action_sample = self.nominal_traj + self.sigma * jax.random.normal(key,shape=(self.planner_num, self.Horizon, self.action_dim))
        action_sample = jnx.clip(action_sample,-1,1)
        #st=time.time()
        plan_input_batch=self.rollout_fn(state,action_sample.transpose(1,0,2))
        assert not jnx.isnan(plan_input_batch).any()
        #print(time.time()-st)

        with torch.no_grad():
            step_rewards = self.reward_fn(j2t(plan_input_batch))
            plan_reward_batch = step_rewards.mean(dim=0).flatten()
            assert not plan_reward_batch.isnan().any()
            plan_reward_adjusted = plan_reward_batch.max() - plan_reward_batch
        
        weights = torch.exp(-plan_reward_adjusted/self.lam)
        weights_norm = weights/weights.sum()
        assert not weights_norm.isnan().any()
        #print(weights_norm)
        weights_jax = t2j(weights_norm.unsqueeze(-1).unsqueeze(-1))
        action_avg = (action_sample * weights_jax).sum(axis = 0)
        action = action_avg[0]
        self.nominal_traj = jnx.concat((action_avg[1:],jnx.zeros((1,self.action_dim))),axis=0)

        return action



