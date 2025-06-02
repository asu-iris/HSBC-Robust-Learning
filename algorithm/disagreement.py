import torch
from ensemble.ensemble import EnsembleModels

def calc_disagreement(ensemble: EnsembleModels, traj_1, traj_2):
    with torch.no_grad():
        sum_reward_1 = ensemble.prediction(traj_1).sum(dim=1).flatten()
        sum_reward_2 = ensemble.prediction(traj_2).sum(dim=1).flatten()
    
    preference_pred = (sum_reward_1>sum_reward_2)
    n_positive = preference_pred.sum()
    n_negative = ensemble.n_models - n_positive

    measure = 4*n_positive*n_negative/ensemble.n_models**2
    return preference_pred,measure.item()

def calc_disagreement_batch(ensemble: EnsembleModels, traj_1_batch, traj_2_batch):
    assert traj_1_batch.shape == traj_2_batch.shape
    with torch.no_grad():
        sum_reward_1 = ensemble.prediction(traj_1_batch).sum(dim=-2).squeeze(-1).T #batch_size * num_models
        sum_reward_2 = ensemble.prediction(traj_2_batch).sum(dim=-2).squeeze(-1).T
    
    preference_pred = (sum_reward_1>sum_reward_2)
    n_positive = preference_pred.sum(dim=-1)
    n_negative = ensemble.n_models - n_positive

    measure = 4*n_positive*n_negative/ensemble.n_models**2
    return preference_pred,measure

def calc_disagreement_jax(jax_func, traj_1, traj_2, n_models):

    sum_reward_1 = jax_func(traj_1).sum(axis=1).flatten()
    sum_reward_2 = jax_func(traj_2).sum(axis=1).flatten()
    
    preference_pred = (sum_reward_1>sum_reward_2)
    n_positive = preference_pred.sum()
    n_negative = n_models - n_positive

    measure = 4*n_positive*n_negative/n_models**2
    return preference_pred,measure



