import torch
import numpy as np
from torch2jax import j2t,t2j
import sys,os
import argparse

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.unitree_mpc import Unitree_MPC

from test_go2.Go2_Robust_Align_free import torch_reward_gt_2, input_wrapper


def eval_ensemble(mpc_module):
    # INPUT_MASK = np.full(270,True)
    # INPUT_MASK[45:-17] = False
    INPUT_MASK = np.full(53,True)
    INPUT_MASK[0:2] = False
    INPUT_MASK[-12:] = False

    model_fn = lambda x:torch_reward_gt_2(input_wrapper(x,INPUT_MASK))
    success_flag = False
    while not success_flag:
        try: #make sure the planning is successful
            obs_seq, act_seq = mpc_module.generate_traj_unitree(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
            print(obs_seq[-1,:19])
            exit()
            success_flag = True
        except AssertionError:
            print('failure: re-planning')
            success_flag = False
    traj = jnx.concat((obs_seq,act_seq),axis=1)
    traj_tensor = j2t(traj)

    reward_gt = torch_reward_gt_2(input_wrapper(traj_tensor,INPUT_MASK)).cpu().numpy()

    reward_sum = reward_gt.sum()
    return reward_sum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='description for option1',type=str)
    parser.add_argument('--device', help='description for option1',type=str)
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
    model_dir = args.dir

    mpc = Unitree_MPC(planner_num=1024,rollout_length=100,Horizon=25,sigma=0.75)
    print(mpc.env.sys.init_q.shape)
    print(mpc.env.sys.init_q)
    exit()

    r_sum = eval_ensemble(mpc)
    print('gt_reward',r_sum)

    np.save(os.path.join(model_dir,'gt.npy'), r_sum)









