import torch
import numpy as np
from torch2jax import j2t,t2j
import jax
import sys,os

os.environ['CUDA_VISIBLE_DEVICES'] ='2'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'
#jax.config.update("jax_default_device", jax.devices()[2])

import jax.numpy as jnx

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
sys.path.append(os.path.abspath(os.getcwd()))

#jax.config.update("jax_default_device", jax.devices()[2])

from ensemble.ensemble import EnsembleModels
from models.reward_model import RewardFCModel
from Trajectory.cartpole_mpc import Cartpole_MPC
from algorithm.disagreement import calc_disagreement,calc_disagreement_batch
from MCMC.opt_mcmc import NaiveLangevinOptim
import argparse
import random



def torch_reward_gt(input): #input:(x,u)
    upright = (input[...,2] + 1)/2
    middle = torch.exp(-input[...,0]**2)
    small_ctrl = torch.exp(-4*input[...,-1]**2)
    small_ctrl = (4+small_ctrl)/5

    small_vel = torch.exp(-0.5*input[...,4]**2)
    small_vel = (1 + small_vel) / 2

    return (upright*middle*small_ctrl*small_vel).unsqueeze(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='directory to save results',type=str)
    
    parser.add_argument('--device', help='cuda device',type=str)
    parser.add_argument('--rounds', help='total rounds',type=int,default=100)
    parser.add_argument('--err', help='error number of every batch',type=int)

    parser.add_argument('--dense', help='densify',type=bool,default=False)
    parser.add_argument('--model_num', help='number',type=int,default=32)
    parser.add_argument('--freq', help='frequency',type=int,default=5)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] =args.device
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.25'

    device = torch.device("cuda:0")
    ens = EnsembleModels(RewardFCModel,{'input_dim':6,"hidden_dim":64,"num_hidden_layers":3},args.model_num,device)

    mpc = Cartpole_MPC(rollout_length=100,planner_num=256,sigma=1.0)

    Filedir = args.dir
    NUM_ROUNDS = args.rounds
    NOISE_LEVEL=0.2
    MEASURE_THRESH = 0.75
    GROUP_LENGTH = 10
    round_cnt = 0
    current_pairs_list = []
    current_labels_list = []
    history_groups = []
    history_labels = []

    disagreement_log = []

    traj_buf = []
    BUF_LEN=20
    model_fn = lambda x:ens.prediction(x).mean(dim=0)
    noise_idx = 0

    data = torch.load(os.path.join(args.dir,'history_data.pt')).to(device)
    label = torch.load(os.path.join(args.dir,'history_label_tensor.pt')).to(device)

    while round_cnt < NUM_ROUNDS:
        if round_cnt%args.freq==0:
            torch.save(ens.stacked_params, os.path.join(Filedir,'ensemble_{}.pt'.format(round_cnt)))
        err_num = args.err
        correct_num = GROUP_LENGTH - err_num

        opt = torch.optim.Adam(ens.stacked_params.values(),lr=0.005,weight_decay=0.001)
        history_data = data[:round_cnt+1]
        history_label_tensor = label[:round_cnt+1]
        print('run num', round_cnt)
        for i in range(500):
            opt.zero_grad()
            reward_pred = ens.prediction(history_data).squeeze(dim=-1).mean(dim=-1)
            #print('reward_pred',reward_pred.shape)
            reward_diff = reward_pred[...,0] - reward_pred[...,1]

            S_batch = torch.sigmoid(10*(2*history_label_tensor -1)*reward_diff).sum(dim=-1)
            
            #sigmoid_S_batch = (torch.relu(S_batch)-torch.relu(S_batch- alpha*GROUP_LENGTH + 1))/(alpha*GROUP_LENGTH - 1)
            #sigmoid_S_batch = torch.sigmoid(3*(S_batch - correct_num + 2)) #2 1
            sigmoid_S_batch = torch.sigmoid(3*(S_batch - correct_num*0.75)) #2 1
            #potentials = -torch.log(sigmoid_S_batch).sum(dim=-1)
            l=-torch.log(sigmoid_S_batch).sum()   
            # if i==0:
            #     input('breakpoint')         

            l.backward()
            opt.step()
            if i%100==99:
                with torch.no_grad():
                    p = sigmoid_S_batch.mean(dim=1)
                #print('potentials',potentials)
                print('mean probs',p.flatten())
                

        #clear group data
        current_pairs_list=[]
        current_labels_list=[]

        #post-filtering
        pref_pred = reward_pred[...,0] >= reward_pred[...,1]
        print((pref_pred==history_label_tensor).sum(dim=-1))
        in_flag = torch.all((pref_pred==history_label_tensor).sum(dim=-1) >= correct_num - 1,dim=1)

        if args.dense:
            print('densify')
            ens.fliter_and_densify(in_flag)
        
        round_cnt+=1
        NOISE_LEVEL = max(0.02 ,NOISE_LEVEL-0.002)

        torch.cuda.empty_cache()
        #print('potential sum',l)

    mpc.rollout_length = 200
    #mpc.sigma = 1.0
    obs_seq, act_seq = mpc.generate_traj_cartpole(model_fn,render=True,
                                                init_pos=None, init_vel=None,noiselevel=0.0,lam_mppi=0.01,filename=os.path.join(Filedir,'simulation_output.html'))

