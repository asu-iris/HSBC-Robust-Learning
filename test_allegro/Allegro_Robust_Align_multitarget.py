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


def torch_reward_gt(input,verbose = False): #input:(x,u)
    # define the goal object pose
    
    obj_target_pos = jnx.array([0.0, -0.00, 0.01]) * 10
    obj_target_pos = j2t(obj_target_pos)
    obj_target_pos.requires_grad = False

    obj_pos = input[...,16:19]
    # object orientation cost
    cost_quat = input[...,19]
    #cost_quat =2 * torch.arccos(torch.abs((obj_quat*obj_target_quat).sum(dim=-1)))/torch.pi
    #cost_quat = torch.sqrt(cost_quat)
    pos_diff = (obj_target_pos-obj_pos)**2
    #pos_diff[2]*=10
    cost_pos = torch.sum(pos_diff,dim=-1)
    #cost_pos = torch.norm(obj_target_pos-obj_pos,p=2,dim=-1)**2
    
    pos_ff_tip = input[...,20:23]
    pos_mf_tip = input[...,23:26]
    pos_rf_tip = input[...,26:29]
    pos_th_tip = input[...,29:32]

    cost_contact = torch.norm(pos_ff_tip-obj_pos,p=2,dim=-1)**2+torch.norm(pos_mf_tip-obj_pos,p=2,dim=-1)**2+\
            torch.norm(pos_rf_tip-obj_pos,p=2,dim=-1)**2+  torch.norm(pos_th_tip-obj_pos,p=2,dim=-1)**2
    
    fallen_cost = torch.sign(torch.clip(-obj_pos[...,2] - 0.30,0.0,1.0))
    
    cost = 100*cost_quat+40*cost_pos + 5*cost_contact #+ 100*fallen_cost
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
    
    return -0.5*cost_contact.unsqueeze(-1) #0.4
    
def input_wrapper(input,mask,target_quat = torch.tensor([1,0.0,0.0,0.0],dtype = torch.float32,device = torch.device("cuda:0"))):
    input_reduced = input[...,mask]
    input_reduced[...,16:19]*=10
    input_reduced[...,23:35]*=10

    joints = input_reduced[...,0:16]
    obj_pos = input_reduced[...,16:19]
    obj_pos[...,2] = torch.clip(obj_pos[...,2],-0.7,1e5)
    obj_quat = input_reduced[...,19:23]
    ftps = input_reduced[...,23:35]

    quat_dist = 1 - (obj_quat*target_quat).sum(dim=-1) ** 2
    quat_dist = quat_dist.unsqueeze(-1)

    final_output = torch.concat((joints,obj_pos,quat_dist,ftps),dim=-1)

    return final_output





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory to save results',type=str)
    parser.add_argument('--device', help='cuda device',type=str)
    parser.add_argument('--rounds', help='total rounds',type=int,default=100)
    parser.add_argument('--obj', help='object',type=str,default = 'cube')
    parser.add_argument('--err', help='error number of every batch',type=int)
    parser.add_argument('--opt', help='optimizer: \"Adam\" or \"MCMC\"',type=str)
    parser.add_argument('--dense', help='densify',type=bool,default=False)
    parser.add_argument('--model_num', help='number',type=int,default=32)
    parser.add_argument('--freq', help='frequency',type=int,default=5)


    args = parser.parse_args()
    print(args)
    #input()

    #os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    )

    #Targets:
    axis_list = [jnx.array([0., 0., 1.]),jnx.array([0., 1., 0.]),jnx.array([1., 0., 0.])]
    angles_list = [0.5 * jnx.pi/2, -0.5 * jnx.pi/2, 1.0 * jnx.pi/2, -1.0 * jnx.pi/2, 1.5 * jnx.pi/2, -1.5 * jnx.pi/2, 2.0 * jnx.pi/2]
    target_quat_list = []
    for axis in axis_list:
        for angle in angles_list:
            obj_target_quat = math.quat_rot_axis(axis=axis, angle = angle)
            obj_target_quat = j2t(obj_target_quat)
            obj_target_quat = obj_target_quat/(torch.norm(obj_target_quat,p=2,dim=-1).unsqueeze(-1)+1e-7)
            target_quat_list.append(obj_target_quat)


    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':32,"hidden_dim":64,"num_hidden_layers":3},args.model_num,device)

    timestep = 0.005
    frame_skip = 8

    # if args.obj != 'cube':
    #     timestep = 0.008
    #     frame_skip = 5

    mpc = Allegro_MPC(obj_name=args.obj,planner_num=1024,Horizon=5,rollout_length=100,sigma=1.0,timestep=timestep, frameskip=frame_skip)

    Filedir = args.dir
    NUM_ROUNDS = args.rounds
    NOISE_LEVEL = 0.1
    EPSILON=0.5 #0.4
    MEASURE_THRESH = 0.70
    GROUP_LENGTH=10
    round_cnt = 0

    current_pairs_list = []
    current_labels_list = []
    current_disagree_list=[]

    history_groups = []
    history_labels = []

    disagreement_log = []

    traj_buf = []
    BUF_LEN=20

    INPUT_MASK = np.full(73,True)
    #INPUT_MASK[0:16] = False
    INPUT_MASK[23:45] = False
    INPUT_MASK[-16:] = False

    measure = 0.0
    noise_idx = 0
    while round_cnt < NUM_ROUNDS:
        #post-filtering
        print('----------------------Logging Stage---------------------------')
        if round_cnt%args.freq==0:
            torch.save(ens.stacked_params, os.path.join(Filedir,'ensemble_{}.pt'.format(round_cnt)))

        measure = 0.0
        print('----------------------Collecting Stage---------------------------')
        while len(current_pairs_list)<GROUP_LENGTH:
            #planning with mean of rewards
            #mpc.sigma = 0.3
            if round_cnt<0:
                print('random policy')
                obs_seq, act_seq = mpc.generate_random_traj()
            else:
                noise_levels = (NOISE_LEVEL,0.0)
                #noise_once = np.random.choice([NOISE_LEVEL,0.0])
                noise_once = noise_levels[noise_idx]
                noise_idx = (noise_idx+1)%2
                #epsilon_once = np.random.choice([EPSILON,1.0])
                print('noise level selected', noise_once)

                target_id = np.random.randint(len(target_quat_list))
                target = target_quat_list[target_id]
                print('target selected', target)
                model_fn = lambda x:ens.prediction(input_wrapper(x,INPUT_MASK,target)).mean(dim=0) + seperate_contact_reward(input_wrapper(x,INPUT_MASK,target))

                success_flag = False
                while not success_flag:
                    try: #make sure the planning is successful
                        print('current mppi sigma,',mpc.sigma)
                        obs_seq, act_seq = mpc.generate_traj_allegro(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=noise_once,lam_mppi=0.01)

                        #check fallen condition
                        obj_z = obs_seq[...,18]
                        if jnx.any(obj_z<-0.01):
                            print('object fallen')
                            fall_idx = jnx.argwhere((obj_z<-0.01))[0,0]
                            print('fall idx',fall_idx)
                            if fall_idx<=40:
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
            traj_tensor_1 = j2t(traj).to(device)
            traj_tensor_1 = input_wrapper(traj_tensor_1,INPUT_MASK,target)

            # start_idx = np.random.randint(51)
            # traj_tensor_1 = traj_tensor_1[start_idx:start_idx+50]
            
            # scan the buffer for disagreement
            for t_2 in traj_buf[::-1]:
                len_1 = traj_tensor_1.shape[0]
                len_2 = t_2.shape[0]

                min_len = min(len_1,len_2)

                if min_len>=80:
                    num_pairs = 3

                else:
                    num_pairs = 2

                # start_idx_1 = 20 + np.random.choice(60,3) #+ np.array([0,20,35])
                # np.random.shuffle(start_idx_1)
                # start_idx_2 = 20 + np.random.choice(60,3) #+ np.array([0,20,35])
                # np.random.shuffle(start_idx_2)
                start_idx_1 = np.random.choice(len_1 - 20, num_pairs, replace=False)
                start_idx_2 = np.random.choice(len_2 - 20, num_pairs, replace=False)
                for i in range(num_pairs):
                    seg_1 = traj_tensor_1[start_idx_1[i]:start_idx_1[i]+20]
                    seg_2 = t_2[start_idx_2[i]:start_idx_2[i]+20]

                    pref,measure = calc_disagreement(ens,seg_1,seg_2)

                    if measure > MEASURE_THRESH:
                        trajs = [seg_1,seg_2]
                        print('found good pair, measure',measure)
                        disagreement_log.append(measure)
                        current_pairs_list.append(torch.stack(trajs,dim=0)) #dim: 2*traj_length*(x_dim+u_dim)
                        current_disagree_list.append(measure)

            traj_buf.append(traj_tensor_1)

            if len(traj_buf)>BUF_LEN:
                traj_buf = traj_buf[1:]
        
        print('---------------------Learning Stage-------------------------------')
        random.shuffle(current_pairs_list)
        # order = np.argsort(current_disagree_list)[::-1]
        # current_pairs_list = [current_pairs_list[i] for i in order]
        current_pairs_list = current_pairs_list[:GROUP_LENGTH]
        current_labels_list = []       
        for p in current_pairs_list:
            pref_label = torch_reward_gt(p[0]).sum() > torch_reward_gt(p[1]).sum()
            current_labels_list.append(pref_label)
            #reset current list

        assert len(current_pairs_list) == len(current_labels_list)
        
        trajs_data = torch.stack(current_pairs_list,dim=0)
        trajs_label = torch.stack(current_labels_list,dim=0)

        #set current_pairs_list to residual
        #current_pairs_list = residual
        current_pairs_list = []
        current_disagree_list = []
        #print('residual length', len(current_pairs_list))

        #randomly flip the label
        err_num = args.err
        correct_num = GROUP_LENGTH - err_num
        if err_num>0:
            print('There is {} wrong labels'.format(err_num))
            print('before flip',trajs_label)
            idx = np.random.choice(10,size=err_num,replace=False)
            trajs_label[idx] = ~trajs_label[idx]
            print('after flip',trajs_label)
        else:
            print('no error labels')
            print('labels', trajs_label)

        if round_cnt==0:
            history_data = trajs_data.unsqueeze(0)
            history_label_tensor = trajs_label.unsqueeze(0)

        else:
            history_data = torch.cat((history_data,trajs_data.unsqueeze(0)))
            history_label_tensor = torch.cat((history_label_tensor,trajs_label.unsqueeze(0)))
        

        #MCMC
        #opt = NaiveLangevinOptim(ens.stacked_params.values(),lr=0.005,noise_scale_factor=0.02,weight_decay=0.001)
        if args.opt == "Adam":
            opt = torch.optim.Adam(ens.stacked_params.values(),lr=0.002,weight_decay=0.001)
        elif args.opt == "MCMC":
            opt = NaiveLangevinOptim(ens.stacked_params.values(),lr=0.005,noise_scale_factor=0.02,weight_decay=0.001) 
        else:
            raise RuntimeError("Invalid Optimizer")
            
        print('run num', round_cnt)
        for i in range(500):
            opt.zero_grad()
            reward_pred = ens.prediction(history_data).squeeze(dim=-1).mean(dim=-1)
            reward_contact = seperate_contact_reward(history_data).squeeze(dim=-1).mean(dim=-1)
            # print('reward_pred',reward_pred.shape)
            # print('reward contact',reward_contact.shape)
            # input()
            reward_diff = reward_pred[...,0] + reward_contact[...,0] - reward_pred[...,1] - reward_contact[...,1]

            S_batch = torch.sigmoid(5*(2*history_label_tensor -1)*reward_diff).sum(dim=-1) #5

            sigmoid_S_batch = torch.sigmoid(3*(S_batch - correct_num*0.75)) #3

            l=-torch.log(sigmoid_S_batch + 1e-7).sum()   
            # if i==0:
            #     input('breakpoint')         

            l.backward()
            opt.step()
            if i%100==99:
                with torch.no_grad():
                    #p = sigmoid_S_batch.prod(dim=1)
                    p = sigmoid_S_batch.mean(dim=1)
                #print('potentials',potentials)
                print('mean probs',p.flatten())

        #post-filtering
        #pref_pred = reward_pred[...,0] >= reward_pred[...,1]
        with torch.no_grad():
            reward_pred = ens.prediction(history_data).squeeze(dim=-1).mean(dim=-1)
            reward_contact = seperate_contact_reward(history_data).squeeze(dim=-1).mean(dim=-1)
            # print('reward_pred',reward_pred.shape)
            # print('reward contact',reward_contact.shape)
            # input()
            reward_diff = reward_pred[...,0] + reward_contact[...,0] - reward_pred[...,1] - reward_contact[...,1]
            pref_pred = reward_diff >= 0

        print((pref_pred==history_label_tensor).sum(dim=-1))
        #in_flag = torch.all((pref_pred==history_label_tensor).sum(dim=-1) >= correct_num - 1,dim=1) # relax the densification
        in_flag = torch.all((pref_pred==history_label_tensor).sum(dim=-1) >= correct_num -1, dim=1)

        if args.dense:
            print('densify')
            ens.fliter_and_densify(in_flag)
        else:
            print('no densify')

        round_cnt+=1

        #update control params
        NOISE_LEVEL = max(0.02 ,NOISE_LEVEL-0.002)
        #mpc.sigma = max(mpc.sigma-0.001, 0.35)


        del trajs_data,trajs_label

        torch.cuda.empty_cache()
        #print('potential sum',l)

    mpc.rollout_length = 200
    target_id = np.random.randint(len(target_quat_list))
    target = target_quat_list[target_id]
    print('target selected', target)
    model_fn = lambda x:ens.prediction(input_wrapper(x,INPUT_MASK,target)).mean(dim=0) + seperate_contact_reward(input_wrapper(x,INPUT_MASK,target))
    obs_seq, act_seq = mpc.generate_traj_allegro(model_fn,render=True,
                                                init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.005,filename=os.path.join(Filedir,'simulation_output.html'))

