import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os
from brax.io.torch import jax_to_torch 
import time
import copy

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.cartpole_mpc import Cartpole_MPC
from Trajectory.walker_mpc import Walker_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim

device = torch.device("cuda:0")
ens = EnsembleModels(RewardFCModel,{'input_dim':23,"hidden_dim":64,"num_hidden_layers":3},128,device)

#ens.stacked_params = torch.load('./Data/Walker_Robust/run_1_gradient_err_10_nn_64/ensemble_89.pt')
ens.stacked_params = torch.load('./Data/Walker_Robust/run_2_ours_err_20/ensemble_74.pt')
print(ens.stacked_params['fc_0.weight'].shape)

mpc = Walker_MPC(planner_num=512,rollout_length=300,Horizon=30,sigma=0.3)


model_fn = lambda x:ens.prediction(x).mean(dim=0)
obs_seq, act_seq = mpc.generate_traj_walker(model_fn,render=True,init_pos=None, init_vel=None,epsilon=1.0,lam_mppi=0.005)


