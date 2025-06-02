import torch
from torch.func import stack_module_state,functional_call,vmap, grad
import copy
import numpy as np

class EnsembleModels(object):
    def __init__(self,modelclass,model_args,n_models,device) -> None:
        self.modelclass = modelclass
        self.device = device
        self.n_models = n_models
        self.model_args = model_args
        if model_args is not None:
            models = [modelclass(**model_args).to(self.device) for i in range(n_models)]
        else:
            models = [modelclass().to(self.device) for i in range(n_models)]

        self.stacked_params, self.stacked_buffer = stack_module_state(models)
        #print(self.stacked_buffer)
        #print(self.stacked_params['fc1.weight'].shape)

        base_model = copy.deepcopy(models[0])
        base_model = base_model.to('meta')

        def fmodel(params, buffers, x):
            return functional_call(base_model, (params, buffers), (x,))
        
        self.fmodel = fmodel
        
        #self.batched_pred = vmap(fmodel, in_dims=(0, 0, None), chunk_size=self.n_models//2)
        self.batched_pred = vmap(fmodel, in_dims=(0, 0, None))



    def prediction(self,data):
        return self.batched_pred(self.stacked_params, self.stacked_buffer,data)
    
    def get_one_model(self,index):
        #manually make a state dict
        state_dict = {}
        for key,values in self.stacked_params.items():
            state_dict[key] = values[index].detach().clone()

        if self.model_args is not None:
            model = self.modelclass(**self.model_args)
        else:
            model = self.modelclass()
        
        model.load_state_dict(state_dict)

        return model.to(self.device)
    
    # filter the candidate models by preference
    def fliter_and_densify(self, mask, sigma=1e-4):
        num_kept = mask.sum()
        if num_kept == self.n_models or num_kept<=1:
            print("skipping densify because num_kept is {}".format(num_kept))
            return
        
        for p in self.stacked_params.values():
            p_kept = p[mask]
            duplicate_num = self.n_models//num_kept
            residual = self.n_models%num_kept
            shape_len = len(p_kept.shape)
            tile_shape = [1 for i in range(shape_len)]
            tile_shape[0] = duplicate_num
            p_new = torch.tile(p_kept,tile_shape)
            p_new = torch.concat((p_new,p_kept[0:residual]),dim=0)
            p_new[num_kept:] += sigma*torch.randn_like(p_new[num_kept:])
            assert p_new.shape == p.shape

            p.data.copy_(p_new)

    def add_noise(self,sigma):
        for p in self.stacked_params.values():
            p.data.add_(sigma * torch.randn_like(p))

    
def flatten_stacked_params(stacked_params):
    """
    return a new tensor of shape (num_models,total number of params in one model)
    """
    param_list = []
    for p in stacked_params.values():
        p_flatten = p.detach().flatten(start_dim=1)
        param_list.append(p_flatten)

    return torch.concat(param_list,dim=1)

def zero_grad_stacked_params(stacked_params):
    """
    clear the grad of the stacked params
    """
    for p in stacked_params.values():
        p.grad.zero_()



