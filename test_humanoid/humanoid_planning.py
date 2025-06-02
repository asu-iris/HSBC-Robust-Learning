import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os
from brax.io.torch import jax_to_torch 
import time
import copy

os.environ['CUDA_VISIBLE_DEVICES'] ='1'

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.cartpole_mpc import Cartpole_MPC
from Trajectory.walker_mpc import Walker_MPC
from Trajectory.Humanoid_mpc import Humanoid_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim

device = torch.device("cuda:0")
ens = EnsembleModels(RewardFCModel,{'input_dim':62,"hidden_dim":128,"num_hidden_layers":3},16,device)

#ens.stacked_params = torch.load('./Data/Walker_Robust/run_1_gradient_err_10_nn_64/ensemble_99.pt')
ens.stacked_params = torch.load('./Data/Humanoid/RA/error_2/run_3/ensemble_149.pt')
print(ens.stacked_params['fc_0.weight'].shape)

mpc = Humanoid_MPC(planner_num=1024,rollout_length=300,Horizon=25,sigma=1.0)

INPUT_MASK = np.full(270,True)
INPUT_MASK[45:-17] = False
model_fn = lambda x:ens.prediction(x[...,INPUT_MASK]).mean(dim=0)
obs_seq, act_seq = mpc.generate_traj_humanoid(model_fn,render=True,init_pos=None, init_vel=None, noiselevel=0.0,lam_mppi=0.01)


