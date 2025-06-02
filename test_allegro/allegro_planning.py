import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os
import random

#jax.config.update("jax_default_device", jax.devices()[2])

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.Allegro_mpc import Allegro_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim
from itertools import combinations
import argparse
from brax import math

from scipy.spatial.transform import Rotation as R


def torch_reward_gt(input,verbose = False): #input:(x,u)
    # define the goal object pose
    
    obj_target_pos = jnx.array([0.02, -0.00, 0.00])
    obj_target_pos = j2t(obj_target_pos)
    obj_target_pos.requires_grad = False

    obj_pos = input[...,16:19]
    # object orientation cost
    cost_quat = input[...,19]
    #cost_quat =2 * torch.arccos(torch.abs((obj_quat*obj_target_quat).sum(dim=-1)))/torch.pi
    #cost_quat = torch.sqrt(cost_quat)
    cost_pos = (obj_target_pos-obj_pos)**2
    cost_pos = torch.norm(obj_target_pos-obj_pos,p=2,dim=-1)**2
    cost_pos[2]*=100

    pos_ff_tip = input[...,20:23]
    pos_mf_tip = input[...,23:26]
    pos_rf_tip = input[...,26:29]
    pos_th_tip = input[...,29:32]

    cost_contact = torch.norm(pos_ff_tip-obj_pos,p=2,dim=-1)**2+torch.norm(pos_mf_tip-obj_pos,p=2,dim=-1)**2+\
            torch.norm(pos_rf_tip-obj_pos,p=2,dim=-1)**2+  torch.norm(pos_th_tip-obj_pos,p=2,dim=-1)**2
    
    cost = 100*cost_quat+0.1*cost_pos + 5*cost_contact
    #cost =   5000*cost_contact #+ 5000*fallen_cost
    reward = -cost.unsqueeze(-1)*0.01

    return reward

def seperate_contact_reward(input):
    pos_ff_tip = input[...,20:23]
    pos_mf_tip = input[...,23:26]
    pos_rf_tip = input[...,26:29]
    pos_th_tip = input[...,29:32]

    obj_pos = input[...,16:19]
    cost_contact = torch.norm(pos_ff_tip-obj_pos,p=2,dim=-1)**2+torch.norm(pos_mf_tip-obj_pos,p=2,dim=-1)**2+\
            torch.norm(pos_rf_tip-obj_pos,p=2,dim=-1)**2+  torch.norm(pos_th_tip-obj_pos,p=2,dim=-1)**2
    
    return -0.2*cost_contact.unsqueeze(-1) 
    
def input_wrapper(input,mask,target_quat = torch.tensor([1,0.0,0.0,0.0],dtype = torch.float32,device = torch.device("cuda:0"))):
    input_reduced = input[...,mask]
    input_reduced[...,16:19]*=10
    input_reduced[...,23:35]*=10

    joints = input_reduced[...,0:16]
    obj_pos = input_reduced[...,16:19]
    obj_quat = input_reduced[...,19:23]
    ftps = input_reduced[...,23:35]

    quat_dist = 1 - (obj_quat*target_quat).sum(dim=-1) ** 2
    quat_dist = quat_dist.unsqueeze(-1)

    final_output = torch.concat((joints,obj_pos,quat_dist,ftps),dim=-1)

    return final_output

os.environ['CUDA_VISIBLE_DEVICES'] ='3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'

obj_target_quat = jnx.array(R.random(random_state=1).as_quat(scalar_first=True))

#obj_target_quat = math.quat_rot_axis(axis=jnx.array([0., 0., 1.]), angle = -0.5 * jnx.pi/2)
obj_target_quat = j2t(obj_target_quat)
obj_target_quat = obj_target_quat/(torch.norm(obj_target_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)
print('target', obj_target_quat)

device = torch.device("cuda:0")
ens = EnsembleModels(RewardFCModel,{'input_dim':32,"hidden_dim":64,"num_hidden_layers":3},16,device)
ens.stacked_params = torch.load('./Data/Allegro/Cube/RA/error_0/run_0/ensemble_30.pt')

mpc = Allegro_MPC(obj_name='cube',planner_num=1024,Horizon=8,rollout_length=200,sigma=1.0)

INPUT_MASK = np.full(73,True)
#INPUT_MASK[0:16] = False
INPUT_MASK[23:45] = False
INPUT_MASK[-16:] = False

model_fn = lambda x:ens.prediction(input_wrapper(x,INPUT_MASK,obj_target_quat)).mean(dim=0) + seperate_contact_reward(input_wrapper(x,INPUT_MASK,obj_target_quat))
model_fn_2 = lambda x:torch_reward_gt(input_wrapper(x,INPUT_MASK,obj_target_quat))
obs_seq, act_seq = mpc.generate_traj_allegro(model_fn,render=True,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
