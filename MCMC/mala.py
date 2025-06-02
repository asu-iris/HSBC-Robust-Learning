import torch
from ensemble.ensemble import flatten_stacked_params

def log_Q_gpu(potential, params_prime, params, step):
    """
    density = exp(-potential)
    """
    potential(params).sum().backward()
    p_step_list = []
    for p in params.values():
        p_step = p - step * p.grad
        p_step_list.append(p_step.detach().flatten(start_dim=1))
    p_step_flatten = torch.concat(p_step_list,dim=1)

    return -(torch.norm(flatten_stacked_params(params_prime) - p_step_flatten, p=2, dim=1) ** 2) / (4 * step)

def update_with_acceptance(new_params,old_params,mask):
    for p,p_old in zip(new_params.values(),old_params.values()):
        shape_len = len(p.shape)
        target_shape = [1 for i in range(shape_len)]
        target_shape[0] = mask.shape[0]

        mask_reshaped = torch.reshape(mask,target_shape)
        p.data.mul_(mask_reshaped)
        p.data.add_((1-mask_reshaped)*p_old)

def mala(potential_fn,new_param,old_param):
    log_q_nom = log_Q_gpu(potential_fn,old_param,new_param,0.001)
    log_q_dem = log_Q_gpu(potential_fn,new_param,old_param,0.001)
    #print(log_q_nom.shape)
    log_ratio = -potential_fn(new_param).flatten() + potential_fn(old_param).flatten() + log_q_nom - log_q_dem
    alpha = torch.minimum(torch.ones_like(log_ratio),torch.exp(log_ratio))
    mask= (torch.rand_like(alpha) < alpha).float()

    update_with_acceptance(new_param,old_param,mask)
    return alpha