import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.Humanoid_mpc import Humanoid_MPC


os.environ['CUDA_VISIBLE_DEVICES'] ='3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'

def torch_reward_gt(input): #input:(x,u)
    #standing = torch.exp(-(1.1-input[...,0])**2)
    standing = torch.clip(input[...,0]/1.2,0.0,1.0)
    #small_ctrl = torch.exp(-0.02 * torch.norm(input[...,-23:]))
    #torso_vel = torch.norm(input[...,-101:-99],p=2,dim=-1)
    torso_vel = torch.norm(input[...,22:24],p=2,dim=-1)
    #torso_w = torch.norm(input[...,-98:-95],p=2,dim=-1)
    torso_w = torch.abs(input[...,27])
    small_vel = torch.exp(-0.1 * torso_vel - 0.3 * torso_w)
    small_ctrl = 1.0

    return (standing * small_ctrl * small_vel).unsqueeze(-1)

mpc = Humanoid_MPC(planner_num=128,Horizon=25,rollout_length=300,sigma=0.25)
obs_seq, act_seq = mpc.generate_traj_humanoid(torch_reward_gt,render=True,noiselevel=0.0,lam_mppi=0.01) #0.01
#print(obs_seq[-1])