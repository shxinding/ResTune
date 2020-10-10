from botorch.sampling.samplers import SobolQMCNormalSampler
from .WGPE import WGPE, get_target_model_loocv_sample_preds, get_fitted_model
import os
import numpy as np
import torch
from .knobs import logger
from botorch.models import ModelListGP
from .utils.meta_weight_calculation import compute_rank_weights_meta
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_POSTERIOR_SAMPLES=256
N_RESTART_CANDIDATES = 512
N_RESTARTS = 10
Q_BATCH_SIZE = 1
N_BATCH = 10
RANDOM_INITIALIZATION_SIZE = 3
N_TRIALS = 10
MC_SAMPLES = 512
noise_std = 0.05


def load_source_model(load_dir_tps, load_dir_cpu):
    workload_list = os.listdir(load_dir_tps)
    for w in workload_list:
        if w.split(".")[-1] != 'pkl':
            workload_list.remove(w)
            continue
        fn = os.path.join(load_dir_tps, w)
        if os.path.isdir(fn):
            workload_list.remove(w)
            continue
    model_dir = {}
    for w in workload_list:
        model_path_tps = os.path.join(load_dir_tps, w)
        model_tps = torch.load(model_path_tps)
        model_path_cpu = os.path.join(load_dir_cpu, w)
        model_cpu = torch.load(model_path_cpu)
        model_dir[w] = (model_tps, model_cpu)
    return model_dir


def roll_col(X, shift):
    """
    Rotate columns to right by shift.
    """
    return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)


def compute_ranking_loss(f_samps, target_y):
    """
    Compute ranking loss for each sample from the posterior over target points.
    """
    n = target_y.shape[0]
    if f_samps.ndim == 4:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y_tps = torch.cartesian_prod(
            target_y[:,0].squeeze(-1),
            target_y[:,0].squeeze(-1),
        ).view(n, n, 2)
        cartesian_y_cpu = torch.cartesian_prod(
            target_y[:,1].squeeze(-1),
            target_y[:,1].squeeze(-1),
        ).view(n, n, 2)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = ((f_samps[:,:,:,0].diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps[:,:,:,0]) ^ (
                    cartesian_y_tps[..., 0] < cartesian_y_tps[..., 1])  & \
                (f_samps[:,:,:,1].diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps[:,:,:,1]) ^ (
                cartesian_y_cpu[..., 0] < cartesian_y_cpu[..., 1])
                     ).sum(dim=-1).sum(dim=-1)
    else:
        rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, target_y.shape[0]):
            rank_loss +=   (((roll_col(f_samps[:,:,0], i) < f_samps[:,:,0]) ^ (roll_col(y_stack[:,:,0], i) < y_stack[:,:,0])) \
               | ((roll_col(f_samps[:, :, 1], i) < f_samps[:, :, 1]) ^ (roll_col(y_stack[:, :, 1], i) < y_stack[:, :, 1]))).sum(dim=-1)
    return rank_loss



def compute_rank_weights(train_x, train_y, base_models, target_model, num_samples):
    """
    Compute ranking weights for each base model and the target model (using
        LOOCV for the target model). Note: Weight dilution addressed
    """
    ranking_losses = []

    # compute ranking loss for each base model
    for task in list(base_models.keys()):
        model = base_models[task]
        # compute posterior over training points for target task
        posterior_tps = model[0].posterior(train_x)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps_tps = sampler(posterior_tps).squeeze(-1).squeeze(-1)

        posterior_cpu = model[1].posterior(train_x)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps_cpu = sampler(posterior_cpu).squeeze(-1).squeeze(-1)
        base_f_samps = torch.stack((base_f_samps_tps, base_f_samps_cpu), -1)
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y).numpy())
    # compute ranking loss for target model using LOOCV
    # f_samps
    #train_yvar = torch.full_like(train_y, noise_std ** 2)
    target_f_samps_tps = get_target_model_loocv_sample_preds(train_x, train_y[:,0].reshape(-1,1), target_model[0], num_samples)
    target_f_samps_cpu = get_target_model_loocv_sample_preds(train_x, train_y[:,1].reshape(-1,1), target_model[1], num_samples)
    target_f_samps = torch.stack((target_f_samps_tps, target_f_samps_cpu), -1)
    ranking_losses.append(compute_ranking_loss(target_f_samps, train_y).numpy())
    # prevent weight dilution by discarding models that are
    # substantially worse than the target model. Model i is discarded
    # from the ensemble if the median of its loss samples
    # is greater than the 95th percentile of the target loss samples
    ranking_losses = np.vstack(ranking_losses)
    target_benchmark = np.percentile(ranking_losses[-1],95)
    discard_model_index = (np.median(ranking_losses, axis=1) > target_benchmark).nonzero()

    ranking_loss_tensor = torch.tensor(ranking_losses)
    # compute best model (minimum ranking loss) for each sample
    best_models = torch.argmin(ranking_loss_tensor, dim=0)
    # compute proportion of samples for which each model is best
    rank_weights = best_models.bincount(minlength=len(ranking_losses)).type_as(train_x) / num_samples
    rank_weights[discard_model_index] = 0
    #restrict the number of source model small than 10
    rank_weights_sort = rank_weights.tolist().copy()
    rank_weights_sort.sort(reverse=True)
    restrict_num = 9 if len(base_models) > 10 else -1
    rank_weights[(rank_weights <= rank_weights_sort[restrict_num]).nonzero()] = 0
    sum_weight = float(rank_weights.sum().detach().numpy())
    rank_weights = rank_weights / sum_weight
    if rank_weights[-1] < 0.5:
        rank_weights[:-1] = rank_weights[:-1] / float(rank_weights[:-1].sum().detach().numpy()) * 0.5
        rank_weights[-1] = 0.5
    rank_weights_dir ={}
    count = 0
    for task in list(base_models.keys()):
        rank_weights_dir[task] = float(rank_weights[count].detach().numpy())
        count = count + 1
    rank_weights_dir['target'] = float(rank_weights[-1].detach().numpy())


    return rank_weights_dir, rank_weights



def initialize_WGPE_model_multiOutput(train_x, train_y, load_dir_tps, load_dir_cpu, tps_constrained):
    target_model_tps = get_fitted_model(train_x, train_y[:,0].reshape(-1,1))
    target_model_cpu = get_fitted_model(train_x, train_y[:,1].reshape(-1,1))
    target_model = (target_model_tps, target_model_cpu)
    source_model_dir = load_source_model(load_dir_tps, load_dir_cpu)
    rank_weight_dir, rank_weights = compute_rank_weights(train_x, train_y, source_model_dir, target_model,
                                                         NUM_POSTERIOR_SAMPLES)
    model_list_tps = []
    for task in list(source_model_dir.keys()):
        model_list_tps.append(source_model_dir[task][0])
    model_list_tps.append(target_model[0])

    model_list_cpu = []
    for task in list(source_model_dir.keys()):
        model_list_cpu.append(source_model_dir[task][1])
    model_list_cpu.append(target_model[1])


    # log weight
    d_order = sorted(rank_weight_dir.items(), key=lambda x: x[1], reverse=True)
    map_num = sum([w[1] != 0 for w in d_order])
    # present_num = map_num if map_num < 5 else 5
    logger.info("[Wokrload Mapping]: map {} workloads, top weights: {}".format(map_num, d_order[:map_num]))
    if map_num == 1:
        non_zero_weight_indice = (rank_weights ** 2 > 0).nonzero()[0]
        model_tps = model_list_tps[non_zero_weight_indice]
        model_cpu = model_list_cpu[non_zero_weight_indice]
        model = ModelListGP(model_tps, model_cpu)
        tps_constrained = (tps_constrained - model_tps.Y_mean) / model_tps.Y_std
        return model, True, tps_constrained.detach().numpy().flatten()[0]
    else:
        wgpe_model_tps = WGPE(model_list_tps, rank_weights)
        wgpe_model_cpu = WGPE(model_list_cpu, rank_weights)
        model = ModelListGP(wgpe_model_tps, wgpe_model_cpu)
        return model, False, None


def initialize_WGPE_model_meta(train_x, train_y, workload_target, load_dir_tps, load_dir_cpu):
    try:
        target_model_tps = get_fitted_model(train_x, train_y[:, 0].reshape(-1, 1))
        target_model_cpu = get_fitted_model(train_x, train_y[:, 1].reshape(-1, 1))
        target_model = (target_model_tps, target_model_cpu)
    except:
        target_model = None

    source_model_dir = load_source_model(load_dir_tps, load_dir_cpu)
    rank_weight_dir, rank_weights = compute_rank_weights_meta(source_model_dir, workload_target, target_model)
    model_list_tps = []
    for task in list(source_model_dir.keys()):
        model_list_tps.append(source_model_dir[task][0])

    model_list_cpu = []
    for task in list(source_model_dir.keys()):
        model_list_cpu.append(source_model_dir[task][1])

    if target_model:
        model_list_cpu.append(target_model_cpu)
        model_list_tps.append(target_model_tps)

    d_order = sorted(rank_weight_dir.items(), key=lambda x: x[1], reverse=True)
    map_num = sum([w[1] != 0 for w in d_order])
    logger.info("[Wokrload Mapping]: map {} workloads, top weights: {}".format(map_num, d_order[:map_num]))

    wgpe_model_tps = WGPE(model_list_tps, rank_weights)
    wgpe_model_cpu = WGPE(model_list_cpu, rank_weights)

    model = ModelListGP(wgpe_model_tps, wgpe_model_cpu)
    return model, False, None

