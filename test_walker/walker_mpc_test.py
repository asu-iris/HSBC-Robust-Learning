import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.walker_mpc import Walker_MPC


os.environ['CUDA_VISIBLE_DEVICES'] ='2'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'

def torch_reward_gt(input): #input:(x,u)
    upright = (torch.cos(input[...,1]) + 1)/2
    #standing = torch.exp(-16*(input[...,0])**2)
    standing = torch.clip(1-1.0*torch.abs(input[...,0]),0,1)
    standing_reward = (3*standing + upright)/4

    #move_reward = torch.exp(-8*(1.0-input[...,8])**2)
    move_reward = torch.clip(input[...,8]/1.0,0.0,1)

    return (standing_reward*move_reward).unsqueeze(-1)

mpc = Walker_MPC(planner_num=512,Horizon=30,rollout_length=800,sigma=0.3)
obs_seq, act_seq = mpc.generate_traj_walker(torch_reward_gt,render=True,epsilon=1.0,lam_mppi=0.005) #0.01
print(obs_seq[-1])