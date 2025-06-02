import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import sys,os
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.unitree_mpc import Unitree_MPC


os.environ['CUDA_VISIBLE_DEVICES'] ='3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'

def torch_reward_gt(input): #input:(x,u)
    robot_pos = input[...,0:3]
    robot_quat = input[...,4:7]
    robot_height = robot_pos[...,-1]

    robot_vel = input[...,23:25]
    robot_angular_vel = input[...,26:29]

    target_height = 0.6
    target_foot_height = 1.2
    # target height cost
    cost_height= (robot_height-target_height)**2

    pads_z_front = input[...,19:21]
    pads_z_rear = input[...,21:23]
    pads_z_rear_1 = input[...,21]
    pads_z_rear_2 = input[...,22]

    #foot_cost =  torch.sum((pads_z_front - target_foot_height) ** 2,dim=-1) + pads_z_rear_1 ** 2 +  (pads_z_rear_2 - 0.5) ** 2
    foot_cost =  torch.sum((pads_z_front - target_foot_height) ** 2,dim=-1) + torch.sum(pads_z_rear ** 2,dim=-1)

    vel_cost = torch.sum(robot_vel ** 2,dim=-1)
    ang_vel_cost = torch.sum(robot_angular_vel ** 2,dim=-1)
    cost = 250*cost_height + 50*foot_cost + 0.0001* ang_vel_cost + 1.0*vel_cost #

    return 0.01*(-cost).unsqueeze(-1)

mpc = Unitree_MPC(planner_num=1024,Horizon=25,rollout_length=200,sigma=1.0)
obs_seq, act_seq = mpc.generate_traj_unitree(torch_reward_gt,render=True,noiselevel=0.0,lam_mppi=1e-3) #0.01
print(obs_seq.shape)
print(torch_reward_gt(j2t(obs_seq)).flatten())
#print(obs_seq[-1])