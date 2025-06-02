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
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim


def torch_reward_gt(input): #input:(x,u)
    #standing = torch.exp(-(1.1-input[...,0])**2)
    standing = torch.clip(input[...,0]/1.2,0.0,1.0)
    #small_ctrl = torch.exp(-0.02 * torch.norm(input[...,-23:]))
    torso_vel = torch.norm(input[...,22:24],p=2,dim=-1)
    torso_w = torch.abs(input[...,27])
    small_vel = torch.exp(-0.1 * torso_vel - 0.3 * torso_w)
    small_ctrl = 1.0

    return (standing * small_ctrl * small_vel).unsqueeze(-1)

def eval_ensemble(mpc_module,ens_model):
    INPUT_MASK = np.full(270,True)
    INPUT_MASK[45:-17] = False
    model_fn = lambda x:ens.prediction(x[...,INPUT_MASK]).mean(dim=0)
    success_flag = False
    while not success_flag:
        try: #make sure the planning is successful
            obs_seq, act_seq = mpc_module.generate_traj_humanoid(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
            success_flag = True
        except AssertionError:
            print('failure: re-planning')
            success_flag = False
    traj = jnx.concat((obs_seq,act_seq),axis=1)
    traj_tensor = j2t(traj)
    reward_gt = torch_reward_gt(traj_tensor).cpu().numpy()
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
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    model_dir = args.dir
    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':23,"hidden_dim":128,"num_hidden_layers":3},16,device)
    mpc = Humanoid_MPC(planner_num=1024,rollout_length=300,Horizon=25,sigma=0.75)
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









