import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import time
from brax.io import html
import sys,os
from iris_env.dexhand.allegro.allegro_object import AllegroObject,RolloutVmapWrapper,get_obs

class env_param:
    def __init__(self):
        
        self.object_name='bunny'
        # self.init_robot_qpos = np.array([
        #     0.0, 0.3, 0.8, 0.8,
        #     0.0, 0.3, 0.8, 0.8,
        #     0.0, 0.3, 0.8, 0.8,
        #     0.8, 0.9, 0.8, 0.3,
        # ])
        # self.init_robot_qpos = np.array([
        #     0.125, 1.13, 1.45, 1.24,
        #     -0.02, 0.445, 1.17, 1.5,
        #     -0.459, 1.54, 1.11, 1.23,
        #     0.638, 1.85, 1.5, 1.26
        # ])

        self.init_robot_qpos = np.array([
            0.125, 0.0, 1.45, 0.0,
            -0.02, 0.0, 1.17, 0.0,
            -0.459, 0.0, 1.11, 0.0,
            0.638, 1.85, 1.5, 1.26
        ])

        self.init_object_pos=np.array([0.01, 0.0, 0.045])
        #self.init_object_pos=np.array([0.01, 0.0, 0.038])
        self.init_object_quat=np.array([1.0, 0.0, 0, 0])
        self.init_object_qpos=np.hstack((self.init_object_pos, self.init_object_quat))

    def set_pos(self, pos, quat = np.array([1.0, 0.0, 0, 0])):
        self.init_object_qpos = np.hstack((pos,quat))

CTRL_RANGE = 0.2
class Allegro_MPC(object):
    def __init__(self, obj_name, Horizon=40, rollout_length=150, planner_num=256, sigma=0.4, timestep = 0.01, frameskip = 4) -> None:
        self.Horizon = Horizon
        self.rollout_length = rollout_length
        self.planner_num = planner_num
        self.sigma = sigma

        self.init_pos_dict = {'cube': np.array([-0.0, 0.0, 0.045]), 'bunny': np.array([-0.0, 0.0, 0.05])}

        self.param = env_param()
        self.param.set_pos(self.init_pos_dict[obj_name]) 
        self.param.object_name = obj_name
        self.env = AllegroObject(self.param,frame_skip=frameskip, timestep = timestep)#0.005
        # jit everything
        self.reset = jax.jit(self.env.reset)
        self.step = jax.jit(self.env.step)

        self.planner = RolloutVmapWrapper(self.env, batch_size=self.planner_num, horizon=self.Horizon)
        self.planner_rollout_episode=jax.jit(self.planner.rollout)

    def generate_traj_allegro(self,torch_model,render = False, init_pos = None, init_vel = None, noiselevel = 0.0, lam_mppi = 1.0, filename = 'simulation_output.html'):
        #planner:
        self.mppi = MY_MPPI(planner_num=self.planner_num,Horizon=self.Horizon,action_dim=self.env.action_dim,
                            rollout_fn=self.planner_rollout_episode,reward_fn=torch_model,lam=lam_mppi,sigma=self.sigma)

        if init_pos is not None:
            init_qpos=jnx.array(init_pos)
        else:
            init_qpos_arr = np.concat((self.param.init_robot_qpos,self.param.init_object_qpos))
            init_qpos_arr[0:16] += 0.05*np.random.uniform(-1,1,size = 16)
            # init_qpos_arr[16:18] += 0.002*np.random.uniform(-1,1,size = 2)
            # init_qpos_arr[19:23] += 0.01*np.random.uniform(-1,1,size = 4)           
            init_qpos=jnx.array(init_qpos_arr)

        if init_vel is not None:
            init_qvel=jnx.array(init_vel)
        else:
            init_qvel_arr = np.zeros(22)
            #init_qvel_arr[0:16] = 0.01*np.random.uniform(-1,1,size = 16)
            init_qvel=jnx.array(init_qvel_arr)

        state=self.env.reset(init_qpos, init_qvel)

        env_obss=[]
        env_actions=[]
        traj_state = []

        seed = int(time.time())
        key = jax.random.key(seed)
        for t in range(self.rollout_length):
            #st=time.time()
            _,key = jax.random.split(key,num=2)
            traj_state.append(state)

            action = self.mppi.command(state,key)
            #print('planning_time',time.time()-st)
            
            # if np.random.rand()>=epsilon:
            #     seed = int(time.time())
            #     key = jax.random.key(seed)
            #     action += 4*self.sigma * jax.random.normal(key,shape=(self.env.action_dim,))
            #     #action = jax.random.uniform(key,shape=(self.env.action_dim,),minval=-1,maxval=1)
            
            action += noiselevel * jax.random.normal(key,shape=(self.env.action_dim,))
            action= jnx.clip(action,-1,1)
            # action = jnx.zeros(self.env.action_dim)
            action_lowlevel = action*CTRL_RANGE

            #sample_center = action
            #step
            obs = get_obs(state)
            #print('obs',obs)
            obj_pos = obs[16:19]
            obj_q = obs[19:23]
            obj_z = obs[18]

            pos_ff_tip = obs[...,45:48]
            pos_mf_tip = obs[...,48:51]
            pos_rf_tip = obs[...,51:54]
            pos_th_tip = obs[...,54:57]

            dist = (np.linalg.norm(pos_ff_tip - obj_pos), np.linalg.norm(pos_mf_tip - obj_pos), np.linalg.norm(pos_rf_tip - obj_pos), np.linalg.norm(pos_th_tip - obj_pos))

            if t%10 == 9:
                np.set_printoptions(precision=3)
                np.set_printoptions(suppress=True)
                print('t',t,'q',obj_q,'pos', obj_pos, 'd', np.array(dist))
                #print('a', action)
                #print('a lowlvl', action_lowlevel)
            #print('action',action)
            env_obss.append(obs)
            env_actions.append(action)

            state=self.step(state,action_lowlevel)
            #print('loop time',time.time()-st)

        if render:
            with open(filename, "w") as f:
                f.write(html.render(self.env.sys, traj_state, height=960))

        return jnx.stack(env_obss,axis=0),jnx.stack(env_actions,axis=0)
    
    def generate_random_traj(self,init_pos = None, init_vel = None):
        if init_pos is not None:
            init_qpos=jnx.array(init_pos)
        else:
            init_qpos_arr = np.concat((self.param.init_robot_qpos,self.param.init_object_qpos))
            init_qpos_arr[0:16] += 0.05*np.random.uniform(-1,1,size = 16)
            init_qpos_arr[16:18] += 0.002*np.random.uniform(-1,1,size = 2)
            init_qpos_arr[19:23] += 0.01*np.random.uniform(-1,1,size = 4)           
            init_qpos=jnx.array(init_qpos_arr)

        if init_vel is not None:
            init_qvel=jnx.array(init_vel)
        else:
            init_qvel_arr = np.zeros(22)
            #init_qvel_arr[0:16] = 0.01*np.random.uniform(-1,1,size = 16)
            init_qvel=jnx.array(init_qvel_arr)

        state=self.env.reset(init_qpos, init_qvel)

        env_obss=[]
        env_actions=[]
        traj_state = []

        seed = int(time.time())
        key = jax.random.key(seed)
        
        for t in range(self.rollout_length):
            #st=time.time()
            traj_state.append(state)
            
            _,key = jax.random.split(key,num=2)
            action = jax.random.uniform(key,shape=(self.env.action_dim,),minval=-1,maxval=1)
            action_lowlevel = action*CTRL_RANGE

            obs = get_obs(state)
            #print('obs',obs)
            obj_pos = obs[16:19]
            obj_q = obs[19:23]
            obj_z = obs[18]

            pos_ff_tip = obs[...,45:48]
            pos_mf_tip = obs[...,48:51]
            pos_rf_tip = obs[...,51:54]
            pos_th_tip = obs[...,54:57]

            dist = (np.linalg.norm(pos_ff_tip - obj_pos), np.linalg.norm(pos_mf_tip - obj_pos), np.linalg.norm(pos_rf_tip - obj_pos), np.linalg.norm(pos_th_tip - obj_pos))

            if t%10 == 9:
                print('t',t,'q',obj_q,'z', obj_z, 'd', np.array(dist))
                #print('a', action)
                #print('a lowlvl', action_lowlevel)
            #print('action',action)
            env_obss.append(obs)
            env_actions.append(action)

            state=self.step(state,action_lowlevel)
            #print('loop time',time.time()-st)


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
        # seed = int(time.time())
        # key = jax.random.key(seed)
        # _,key = jax.random.split(key,num=2)
        action_sample = self.nominal_traj + self.sigma * jax.random.normal(key,shape=(self.planner_num, self.Horizon, self.action_dim))
        action_sample = jnx.clip(action_sample,-1,1)
        #st=time.time()
        plan_input_batch=self.rollout_fn(state,action_sample.transpose(1,0,2)* CTRL_RANGE) #scaling of the control
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