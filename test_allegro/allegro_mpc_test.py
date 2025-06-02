import numpy as np
import jax
import jax.numpy as jnx
import torch
from torch2jax import j2t, t2j
import sys,os
from brax import math

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

from Trajectory.Allegro_mpc import Allegro_MPC
os.environ['CUDA_VISIBLE_DEVICES'] ='2'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.40'
jax.config.update('jax_default_matmul_precision', 'high')

def torch_reward_gt(input,verbose = False): #input:(x,u)
    # define the goal object pose
    obj_target_quat = math.quat_rot_axis(
        axis=jnx.array([0., 1., 0.]), angle = 1.5 * jnx.pi/2)
    obj_target_pos = jnx.array([0.02, -0.00, 0.02])

    if verbose:
        print('p', obj_target_pos)
        print('q',obj_target_quat)
        print('quat norm', np.linalg.norm(obj_target_quat))

    #convert to torch
    obj_target_quat = j2t(obj_target_quat)
    obj_target_pos = j2t(obj_target_pos)
    obj_target_pos.requires_grad = False
    obj_target_quat.requires_grad = False

    # get the object pose
    qpos = input[...,0:23]
    qvel = input[...,23:22+23]
    obj_pos = qpos[...,-7:-4]
    obj_quat = qpos[...,-4:]

    obj_quat = obj_quat/(torch.norm(obj_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)
    obj_target_quat = obj_target_quat/(torch.norm(obj_target_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)

    # object orientation cost
    cost_quat = 1 - ((obj_quat*obj_target_quat).sum(dim=-1)) ** 2
    #cost_quat =2 * torch.arccos(torch.abs((obj_quat*obj_target_quat).sum(dim=-1)))/torch.pi
    #cost_quat = torch.sqrt(cost_quat)
    #cost_pos = torch.norm(obj_target_pos[...,2]-obj_pos[...,2],p=2,dim=-1)**2
    pos_diff = (obj_target_pos-obj_pos)**2
    pos_diff[2]*=10
    cost_pos = torch.sum(pos_diff,dim=-1)
    # cost_pos = (obj_target_pos[...,2]-obj_pos[...,2])**2
    #cost_pos[2]*=500

    fallen_cost = torch.clip(-obj_pos[...,2] + 0.030,0.0,1.0)

    action = input[...,-16:]

    # control cost
    cost_control = torch.norm(action,p=2,dim=-1)
    # contact cost
    pos_ff_tip = input[...,45:48]
    pos_mf_tip = input[...,48:51]
    pos_rf_tip = input[...,51:54]
    pos_th_tip = input[...,54:57]
    cost_contact = torch.norm(pos_ff_tip-obj_pos,p=2,dim=-1)**2+torch.norm(pos_mf_tip-obj_pos,p=2,dim=-1)**2+\
            torch.norm(pos_rf_tip-obj_pos,p=2,dim=-1)**2+  torch.norm(pos_th_tip-obj_pos,p=2,dim=-1)**2
    if verbose:
        print(cost_quat)
        print(cost_pos.shape)
        print(cost_control)
        print(cost_contact)

    cost = 100*cost_quat+100*cost_pos+0.0*cost_control + 500*cost_contact #+ 5000*fallen_cost
    #cost = 0*cost_quat+0*cost_pos+0.0*cost_control + 1000*cost_contact #+ 5000*fallen_cost
    reward = -cost.unsqueeze(-1)*0.01

    return reward

mpc = Allegro_MPC(obj_name='bunny',planner_num=512,Horizon=5,rollout_length=200,sigma=0.3, timestep = 0.005, frameskip = 8)
obs_seq, act_seq = mpc.generate_traj_allegro(torch_reward_gt,render=True,noiselevel=0.0,lam_mppi=5e-5) #0.01
test_input = jnx.concat((obs_seq,act_seq),axis = -1)
test_input = j2t(test_input)
print(torch_reward_gt(test_input,verbose=True).flatten())
#print(obs_seq[-1])
