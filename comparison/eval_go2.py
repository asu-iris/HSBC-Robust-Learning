import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os
from brax.io.torch import jax_to_torch 
import time
import copy
import argparse



import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.Humanoid_mpc import Humanoid_MPC
from Trajectory.unitree_mpc import Unitree_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim
from test_go2.Go2_Robust_Align_free import torch_reward_gt_2, input_wrapper

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
    foot_cost =  torch.sum((pads_z_front - target_foot_height) ** 2,dim=-1) + torch.sum(pads_z_rear ** 2,dim=-1) 

    vel_cost = torch.sum(robot_vel ** 2,dim=-1)
    ang_vel_cost = torch.sum(robot_angular_vel ** 2,dim=-1)
    cost = 200*cost_height + 50*foot_cost + 0.0001* ang_vel_cost + 1e-6*vel_cost #

    return 0.01*(-cost).unsqueeze(-1)

def eval_ensemble(mpc_module,ens_model):
    # INPUT_MASK = np.full(270,True)
    # INPUT_MASK[45:-17] = False
    INPUT_MASK = np.full(53,True)
    INPUT_MASK[0:2] = False
    INPUT_MASK[-12:] = False

    model_fn = lambda x:ens_model.prediction(input_wrapper(x,INPUT_MASK)).mean(dim=0)
    success_flag = False
    while not success_flag:
        try: #make sure the planning is successful
            obs_seq, act_seq = mpc_module.generate_traj_unitree(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
            success_flag = True
        except AssertionError:
            print('failure: re-planning')
            success_flag = False
    traj = jnx.concat((obs_seq,act_seq),axis=1)
    traj_tensor = j2t(traj)

    reward_gt = torch_reward_gt_2(input_wrapper(traj_tensor,INPUT_MASK)).cpu().numpy()
    with torch.no_grad():
        reward_pred = model_fn(traj_tensor).cpu().numpy()

    reward_sum = reward_gt.sum()
    return reward_gt,reward_pred,reward_sum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='description for option1',type=str)
    parser.add_argument('--device', help='description for option1',type=str)
    parser.add_argument('--id', help='description for option1',type=str)
    parser.add_argument('--range', help='range',type=int,default=51)
    parser.add_argument('--freq', help='freq',type=int,default=5)
    args = parser.parse_args()
    args.id

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
    model_dir = args.dir
    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':53,"hidden_dim":64,"num_hidden_layers":3},16,device)
    mpc = Unitree_MPC(planner_num=1024,rollout_length=100,Horizon=25,sigma=0.75)
    result = {}
    for i in range(0,args.range,args.freq):
        ens.stacked_params = torch.load(os.path.join(model_dir,'ensemble_{}.pt'.format(i)))
        print(ens.stacked_params['fc_0.weight'].shape)
        #input()
        r_gt,r_pred,r_sum = eval_ensemble(mpc,ens)
        print('ensemble_{}.pt'.format(i),r_sum)
        result[str(i)] = r_sum
        result['r_gt'] = r_gt
        result['r_pred'] = r_pred

    np.savez(os.path.join(model_dir,'eval_result_{}.npz'.format(args.id)), **result)









