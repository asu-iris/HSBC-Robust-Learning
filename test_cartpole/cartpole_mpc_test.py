import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.cartpole_mpc import Cartpole_MPC

def torch_reward_gt(input): #input:(x,u)
    upright = (input[...,2] + 1)/2
    middle = torch.exp(-input[...,0]**2)
    small_ctrl = torch.exp(-4*input[...,-1]**2)
    small_ctrl = (4+small_ctrl)/5

    small_vel = torch.exp(-0.5*input[...,4]**2)
    small_vel = (1 + small_vel) / 2

    return (upright*middle*small_ctrl*small_vel).unsqueeze(-1)

mpc = Cartpole_MPC(rollout_length=400,sigma=0.8)
obs_seq, act_seq = mpc.generate_traj_cartpole(torch_reward_gt,render=True,epsilon=1.0)
print(obs_seq[-1])