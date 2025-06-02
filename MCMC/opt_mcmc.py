import torch
import numpy as np
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy

class NaiveLangevinOptim(Optimizer):
    def __init__(self, params, lr=required, noise_scale_factor = 1e-2, weight_decay = 0.001) -> None:
        assert weight_decay>0
        self.noise_scale_factor = noise_scale_factor
        defaults = dict(lr=lr,weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            #print(group['params'])
            for p in group['params']:
                if p.grad is None:
                        continue
                #here d_p is already the derivative of potential, which is -log(p)
                d_p = p.grad.data
                d_p.add_(weight_decay*p)

                #do gradient descend
                p.data.add_(d_p,alpha = -lr)

                #noise
                p.data.add_(self.noise_scale_factor * np.sqrt(2*lr) * torch.randn_like(p))

