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
# from Trajectory.Humanoid_mpc import Humanoid_MPC
from Trajectory.unitree_mpc import Unitree_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim
from itertools import combinations
import argparse
from test_go2.Go2_Robust_Align_free import torch_reward_gt_2, input_wrapper
def BT_prob(x_0,x_1,alpha=1.0):
    return torch.exp(x_0*alpha) / (torch.exp(x_0*alpha) + torch.exp(x_1 * alpha))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory to save results',type=str)
    parser.add_argument('--device', help='cuda device',type=str)
    parser.add_argument('--rounds', help='total rounds',type=int,default=100)
    parser.add_argument('--err', help='error number of every batch',type=int)
    parser.add_argument('--opt', help='optimizer: \"Adam\" or \"MCMC\"',type=str)
    parser.add_argument('--model_num', help='number',type=int,default=32)
    parser.add_argument('--freq', help='frequency',type=int,default=5)


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
    os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    )

    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':39,"hidden_dim":64,"num_hidden_layers":3},args.model_num,device)

    mpc = Unitree_MPC(planner_num=1024,Horizon=25,rollout_length=150,sigma=1.0)

    Filedir = args.dir
    NUM_ROUNDS = args.rounds
    NOISE_LEVEL = 0.12
    EPSILON=0.85 #0.4
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

    INPUT_MASK = np.full(53,True)
    INPUT_MASK[0:2] = False
    INPUT_MASK[-12:] = False
    model_fn = lambda x:ens.prediction(input_wrapper(x,INPUT_MASK)).mean(dim=0)

    measure = 0.0
    noise_idx = 0
    while round_cnt < NUM_ROUNDS:
        print('----------------------Logging Stage---------------------------')
        if round_cnt%args.freq==0:
            torch.save(ens.stacked_params, os.path.join(Filedir,'ensemble_{}.pt'.format(round_cnt)))
            
        measure = 0.0
        print('----------------------Collecting Stage---------------------------')
        while len(current_pairs_list)<GROUP_LENGTH:
            #planning with mean of rewards
            #mpc.sigma = 0.3
            if round_cnt<10:
                print('random policy')
                success_flag = False
                while not success_flag:
                    try:
                        obs_seq, act_seq = mpc.generate_random_traj()
                        assert not jnx.isnan(obs_seq).any()
                        success_flag = True

                    except AssertionError:
                        print('failure: re-planning')
                        success_flag = False
                
            else:
                noise_levels = (NOISE_LEVEL,0.0)
                #noise_once = np.random.choice([NOISE_LEVEL,0.0])
                noise_once = noise_levels[noise_idx]
                noise_idx = (noise_idx+1)%2
                print('noise level selected', noise_once)
                success_flag = False
                while not success_flag:
                    try: #make sure the planning is successful
                        print('current mppi sigma,',mpc.sigma)
                        obs_seq, act_seq = mpc.generate_traj_unitree(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=noise_once,lam_mppi=5e-3)
                        success_flag = True
                    except AssertionError:
                        print('failure: re-planning')
                        success_flag = False

            traj = jnx.concat((obs_seq,act_seq),axis=1)
            # traj = traj[...,INPUT_MASK]
            traj_tensor_1 = j2t(traj).to(device)
            traj_tensor_1 = input_wrapper(traj_tensor_1,INPUT_MASK)

            # start_idx = np.random.randint(51)
            # traj_tensor_1 = traj_tensor_1[start_idx:start_idx+50]
            
            # scan the buffer for disagreement
            for t_2 in traj_buf[::-1]:
                len_1 = traj_tensor_1.shape[0]
                len_2 = t_2.shape[0]

                start_idx_1 = np.random.choice(len_1-25,3,replace=False)
                start_idx_2 = np.random.choice(len_2-25,3,replace=False)

                for i in range(3):
                    seg_1 = traj_tensor_1[start_idx_1[i]:start_idx_1[i]+25]
                    seg_2 = t_2[start_idx_2[i]:start_idx_2[i]+25]

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
        current_pairs_list = current_pairs_list[:GROUP_LENGTH]
        current_labels_list = []    
        for p in current_pairs_list:
            pref_label = torch_reward_gt_2(p[0]).sum() > torch_reward_gt_2(p[1]).sum()
            current_labels_list.append(pref_label)
            #reset current list

        assert len(current_pairs_list) == len(current_labels_list)
        
        trajs_data = torch.stack(current_pairs_list,dim=0)
        trajs_label = torch.stack(current_labels_list,dim=0)

        #set current_pairs_list to residual
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
            opt = torch.optim.Adam(ens.stacked_params.values(),lr=0.005,weight_decay=0.001)
        elif args.opt == "MCMC":
            opt = NaiveLangevinOptim(ens.stacked_params.values(),lr=0.005,noise_scale_factor=0.02,weight_decay=0.001) 
        else:
            raise RuntimeError("Invalid Optimizer")
            
        print('run num', round_cnt)
        for i in range(500):
            opt.zero_grad()
            data_flattened = history_data.flatten(start_dim=0,end_dim=1)
            segs_0 = data_flattened[:,0,...]
            segs_1 = data_flattened[:,1,...]
            
            label_flattened = history_label_tensor.flatten(start_dim=0,end_dim=1)
            reward_0 = ens.prediction(segs_0).squeeze(dim=-1).mean(dim=-1)
            reward_1 = ens.prediction(segs_1).squeeze(dim=-1).mean(dim=-1)

            P_pred_0 = BT_prob(reward_0,reward_1,alpha=3.0)
            P_pred_1 = 1 - P_pred_0

            P_pred = torch.stack((P_pred_0,P_pred_1),dim=-1)

            tmp_label = torch.zeros_like(P_pred)
            tmp_label[...,0] = label_flattened * 1
            tmp_label[...,1] = 1 - label_flattened * 1

            l = torch.abs(P_pred - tmp_label).sum(dim=-1).mean(dim=-1).sum()
            l.backward()
            if i%100==99:
                print('loss',l.item())
            
            opt.step()

        round_cnt+=1
        #update control params
        NOISE_LEVEL = max(0.004 ,NOISE_LEVEL-0.002)
        #mpc.sigma = max(mpc.sigma-0.001, 0.35)

        del trajs_data,trajs_label

        torch.cuda.empty_cache()
        #print('potential sum',l)

    mpc.rollout_length = 150
    #mpc.sigma = 0.6
    obs_seq, act_seq = mpc.generate_traj_unitree(model_fn,render=True,
                                                init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.005,filename=os.path.join(Filedir,'simulation_output.html'))

