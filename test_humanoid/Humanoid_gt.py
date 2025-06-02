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
from Trajectory.Humanoid_mpc import Humanoid_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim

from Humanoid_Robust_Align import torch_reward_gt

def eval_ensemble(mpc_module):
    INPUT_MASK = np.full(270,True)
    INPUT_MASK[45:-17] = False
    success_flag = False
    while not success_flag:
        try: #make sure the planning is successful
            obs_seq, act_seq = mpc_module.generate_traj_humanoid(torch_reward_gt,render=False,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
            success_flag = True
        except AssertionError:
            print('failure: re-planning')
            success_flag = False
    print(obs_seq[-1,:23])
    traj = jnx.concat((obs_seq,act_seq),axis=1)
    traj_tensor = j2t(traj)
    reward_gt = torch_reward_gt(traj_tensor).cpu().numpy()
    r_sum = reward_gt.sum()
    return r_sum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='description for option1',type=str)
    parser.add_argument('--device', help='description for option1',type=str)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    model_dir = args.dir
    mpc = Humanoid_MPC(planner_num=1024,rollout_length=300,Horizon=25,sigma=0.75)
    print(mpc.env.sys.init_q.shape)

    r_sum = eval_ensemble(mpc)
    print('gt reward',r_sum)

    np.save(os.path.join(model_dir,'gt.npy'), r_sum)









