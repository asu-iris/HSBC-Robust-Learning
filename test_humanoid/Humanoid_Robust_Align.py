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
from Trajectory.Humanoid_mpc import Humanoid_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim
from itertools import combinations
import argparse

def torch_reward_gt(input): #input:(x,u)
    #standing = torch.exp(-(1.1-input[...,0])**2)
    standing = torch.clip(input[...,0]/1.2,0.0,1.0)
    #small_ctrl = torch.exp(-0.02 * torch.norm(input[...,-23:]))
    torso_vel = torch.norm(input[...,22:24],p=2,dim=-1)
    torso_w = torch.abs(input[...,27])
    small_vel = torch.exp(-0.1 * torso_vel - 0.3 * torso_w)
    small_ctrl = 1.0

    return (standing * small_ctrl * small_vel).unsqueeze(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory to save results',type=str)
    parser.add_argument('--device', help='cuda device',type=str)
    parser.add_argument('--rounds', help='total rounds',type=int,default=100)
    parser.add_argument('--err', help='error number of every batch',type=int)
    parser.add_argument('--opt', help='optimizer: \"Adam\" or \"MCMC\"',type=str)
    parser.add_argument('--dense', help='densify',type=bool,default=False)
    parser.add_argument('--model_num', help='number',type=int,default=32)
    parser.add_argument('--freq', help='frequency',type=int,default=5)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
    os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True '
    )

    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':62,"hidden_dim":64,"num_hidden_layers":3},args.model_num,device)

    mpc = Humanoid_MPC(planner_num=1024,Horizon=25,rollout_length=200,sigma=1.2)

    Filedir = args.dir
    NUM_ROUNDS = args.rounds
    NOISE_LEVEL = 0.2
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

    INPUT_MASK = np.full(270,True)
    INPUT_MASK[45:-17] = False
    model_fn = lambda x:ens.prediction(x[...,INPUT_MASK]).mean(dim=0)

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
            if round_cnt<20:
                print('random policy')
                obs_seq, act_seq = mpc.generate_random_traj()
            else:
                noise_levels = (NOISE_LEVEL,0.0)
                #noise_once = np.random.choice([NOISE_LEVEL,0.0])
                noise_once = noise_levels[noise_idx]
                noise_idx = (noise_idx+1)%2
                #epsilon_once = np.random.choice([EPSILON,1.0])
                print('noise level selected', noise_once)
                success_flag = False
                while not success_flag:
                    try: #make sure the planning is successful
                        print('current mppi sigma,',mpc.sigma)
                        obs_seq, act_seq = mpc.generate_traj_humanoid(model_fn,render=False,init_pos=None, init_vel=None,noiselevel=noise_once,lam_mppi=0.01)
                        success_flag = True
                    except AssertionError:
                        print('failure: re-planning')
                        success_flag = False

            traj = jnx.concat((obs_seq,act_seq),axis=1)
            traj = traj[...,INPUT_MASK]
            traj_tensor_1 = j2t(traj).to(device)

            # start_idx = np.random.randint(51)
            # traj_tensor_1 = traj_tensor_1[start_idx:start_idx+50]
            
            # scan the buffer for disagreement
            for t_2 in traj_buf[::-1]:
                start_idx_1 = np.random.choice(51,3) + np.array([20,60,100])
                np.random.shuffle(start_idx_1)
                start_idx_2 = np.random.choice(51,3) + np.array([20,60,100])
                np.random.shuffle(start_idx_2)
                for i in range(3):
                    seg_1 = traj_tensor_1[start_idx_1[i]:start_idx_1[i]+50]
                    seg_2 = t_2[start_idx_2[i]:start_idx_2[i]+50]

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
            opt = torch.optim.Adam(ens.stacked_params.values(),lr=0.005,weight_decay=0.001)
        elif args.opt == "MCMC":
            opt = NaiveLangevinOptim(ens.stacked_params.values(),lr=0.005,noise_scale_factor=0.02,weight_decay=0.001) 
        else:
            raise RuntimeError("Invalid Optimizer")
            
        print('run num', round_cnt)
        for i in range(500):
            opt.zero_grad()
            reward_pred = ens.prediction(history_data).squeeze(dim=-1).mean(dim=-1)
            #print('reward_pred',reward_pred.shape)
            reward_diff = reward_pred[...,0] - reward_pred[...,1]

            S_batch = torch.sigmoid(5*(2*history_label_tensor -1)*reward_diff).sum(dim=-1)

            sigmoid_S_batch = torch.sigmoid(3*(S_batch - correct_num*0.75)) #2 1

            l=-torch.log(sigmoid_S_batch).sum()   
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
        pref_pred = reward_pred[...,0] >= reward_pred[...,1]
        print((pref_pred==history_label_tensor).sum(dim=-1))
        #in_flag = torch.all((pref_pred==history_label_tensor).sum(dim=-1) >= correct_num - 1,dim=1) # relax the densification
        in_flag = torch.all((pref_pred==history_label_tensor).sum(dim=-1) >= correct_num -1, dim=1)

        if args.dense:
            print('densify')
            ens.fliter_and_densify(in_flag)
        


        round_cnt+=1

        #update control params
        NOISE_LEVEL = max(0.02 ,NOISE_LEVEL-0.002)
        #mpc.sigma = max(mpc.sigma-0.001, 0.35)


        del trajs_data,trajs_label

        torch.cuda.empty_cache()
        #print('potential sum',l)

    mpc.rollout_length = 300
    #mpc.sigma = 0.5
    obs_seq, act_seq = mpc.generate_traj_humanoid(model_fn,render=True,
                                                init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.005,filename=os.path.join(Filedir,'simulation_output.html'))

