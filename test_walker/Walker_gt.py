import torch
import numpy as np
from torch2jax import j2t,t2j
import sys,os
import argparse
import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.walker_mpc import Walker_MPC
from test_walker.Walker_Robust_Align import torch_reward_gt

def eval_ensemble(mpc_module):
    obs_seq, act_seq = mpc_module.generate_traj_walker(torch_reward_gt,render=True,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
    print('joints',obs_seq[-1,:8])
    exit()
    traj = jnx.concat((obs_seq,act_seq),axis=1)
    traj_tensor = j2t(traj)

    reward_gt = torch_reward_gt(traj_tensor).cpu().numpy()

    reward_sum = reward_gt.sum()
    return reward_sum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='description for option1',type=str)
    parser.add_argument('--device', help='description for option1',type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    model_dir = args.dir
    mpc = Walker_MPC(planner_num=1024,rollout_length=500,Horizon=30,sigma=0.75) #512

    r_sum = eval_ensemble(mpc)
    print('gt reward',r_sum)

    np.save(os.path.join(model_dir,'gt.npy'), r_sum)
