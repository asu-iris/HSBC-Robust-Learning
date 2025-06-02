import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os
import random

from scipy.spatial.transform import Rotation as R
#jax.config.update("jax_default_device", jax.devices()[2])

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.Allegro_mpc import Allegro_MPC

import argparse
from brax import math
from test_allegro.Allegro_Robust_Align_multitarget import torch_reward_gt, seperate_contact_reward, input_wrapper

def eval_ensemble(mpc_module,ens_model):
    INPUT_MASK = np.full(73,True)
    #INPUT_MASK[0:16] = False
    INPUT_MASK[23:45] = False
    INPUT_MASK[-16:] = False
    #Targets:
    # targets = R.random(5,random_state=RANDOM_SEED).as_quat(scalar_first=True)
    # target_quat_list = []
    # for i in range(targets.shape[0]):
    #     obj_target_quat = j2t(jnx.array(targets[i]))
    #     obj_target_quat = obj_target_quat/(torch.norm(obj_target_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)
    #     target_quat_list.append(obj_target_quat)

    axis_list = [jnx.array([1., 0., 0.]),jnx.array([0., 1., 0.]),jnx.array([0., 0., 1.])]
    angles_list = [1.0 * jnx.pi/2, -1.0 * jnx.pi/2] #[jnx.pi] 
    target_quat_list = []
    for axis in axis_list:
        for angle in angles_list:
            obj_target_quat = math.quat_rot_axis(axis=axis, angle = angle)
            obj_target_quat = j2t(obj_target_quat)
            obj_target_quat = obj_target_quat/(torch.norm(obj_target_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)
            target_quat_list.append(obj_target_quat)

    reward_targets = []
    for obj_target_quat in target_quat_list:
        print('current target',obj_target_quat)
        model_fn = lambda x:ens_model.prediction(input_wrapper(x,INPUT_MASK,obj_target_quat)).mean(dim=0) + seperate_contact_reward(input_wrapper(x,INPUT_MASK,obj_target_quat))
        success_flag = False
        while not success_flag:
            try: #make sure the planning is successful
                obs_seq, act_seq = mpc_module.generate_traj_allegro(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01)
                success_flag = True
                # check fallen condition
                obj_z = obs_seq[...,18]
                if jnx.any(obj_z<-0.01):
                    print('object fallen')
                    fall_idx = jnx.argwhere(obj_z<-0.01)[0,0]
                    print('fall idx',fall_idx)
                    if fall_idx<=60:
                        print('fallen too soon, re-planning')
                        success_flag = False

                    else:
                        obs_seq = obs_seq[:fall_idx]
                        act_seq = act_seq[:fall_idx]
                        success_flag = True

                else:
                    success_flag = True
            except AssertionError:
                print('failure: re-planning')
                success_flag = False
        traj = jnx.concat((obs_seq,act_seq),axis=1)
        traj_tensor = j2t(traj)
        reward_gt = torch_reward_gt(input_wrapper(traj_tensor,INPUT_MASK,obj_target_quat)).cpu().numpy()
        with torch.no_grad():
            reward_pred = model_fn(traj_tensor).cpu().numpy()

        reward_mean = reward_gt.mean()
        reward_targets.append(reward_mean)
    return np.mean(reward_targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='description for option1',type=str)
    parser.add_argument('--device', help='description for option1',type=str)
    parser.add_argument('--id', help='description for option1',type=str)
    parser.add_argument('--seed', help='random seed',type=int,default=256)
    parser.add_argument('--obj', help='obj',type=str,default='cube')
    parser.add_argument('--range', help='range',type=int,default=32)
    parser.add_argument('--freq', help='freq',type=int,default=2)
    args = parser.parse_args()
    print(args)
    args.id

    RANDOM_SEED = args.seed

    #os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    )

    model_dir = args.dir

    device = torch.device("cuda:0")

    ens = EnsembleModels(RewardFCModel,{'input_dim':23,"hidden_dim":128,"num_hidden_layers":3},16,device)
    mpc =  Allegro_MPC(obj_name=args.obj,planner_num=1024,Horizon=5,rollout_length=150,sigma=1.0,timestep=0.005,frameskip=8)
    result = {}
    for i in range(0,args.range,args.freq):
        ens.stacked_params = torch.load(os.path.join(model_dir,'ensemble_{}.pt'.format(i)))
        print(ens.stacked_params['fc_0.weight'].shape)
        #input()
        r_sum = eval_ensemble(mpc,ens)
        print('ensemble_{}.pt'.format(i),r_sum)
        result[str(i)] = r_sum


    np.savez(os.path.join(model_dir,'eval_result_{}.npz'.format(args.id)), **result)