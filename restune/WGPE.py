from botorch.models.gpytorch import GPyTorchModel
from gpytorch.models import GP
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from torch.nn import ModuleList
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
import os
import numpy as np
import torch
from .knobs import logger
import pdb

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

class WGPE(GP, GPyTorchModel):
    """
    Rank-weighted GP ensemble. Note: this class inherits from GPyTorchModel which provides an
        interface for GPyTorch models in botorch.
    """
    _num_outputs = 1  # metadata for botorch
    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "WGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        self.to(weights)

    def forward(self, x):
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights ** 2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            #posterior_mean = posterior.mean.squeeze(-1) * model.Y_std + model.Y_mean
            #posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
            posterior_mean = posterior.mean.squeeze(-1)
            posterior_cov = posterior.mvn.lazy_covariance_matrix
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            if not self.weights[-1] ** 2 > 0:
                weighted_covars.append(posterior_cov * weight ** 2)

        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        posterior_target = self.models[-1].posterior(x)
        posterior_cov_target = posterior_target.mvn.lazy_covariance_matrix
        if self.weights[-1] ** 2 > 0:
            weighted_covars.append(posterior_cov)
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)


def get_fitted_model(train_X, train_Y, state_dict=None, Y_mean=-1, Y_std=-1):
    """
    Get a single task GP. The model will be fit unless a state_dict with model
        hyperparameters is provided.
    """
    if Y_std == -1 :
        Y_mean = train_Y.mean(dim=-2, keepdim=True)
        Y_std = train_Y.std(dim=-2, keepdim=True)
    #if torch.isnan(Y_std):
    #   return 0
    model = SingleTaskGP(train_X,   (train_Y - Y_mean) / Y_std)
    model.Y_mean = Y_mean
    model.Y_std = Y_std
    if state_dict is None:
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)
    return model


def load_source_model(load_dir):
    workload_list = os.listdir(load_dir)
    for w in workload_list:
        if w.split(".")[-1] != 'pkl':
            workload_list.remove(w)
            continue
        fn = os.path.join(load_dir, w)
        if os.path.isdir(fn):
            workload_list.remove(w)
            continue
    model_dir = {}
    for w in workload_list:
        model_path = os.path.join(load_dir, w)
        model = torch.load(model_path)
        model_dir[w] = model
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
    if f_samps.ndim == 3:
        # Compute ranking loss for target model
        # take cartesian product of target_y
        cartesian_y = torch.cartesian_prod(
            target_y.squeeze(-1),
            target_y.squeeze(-1),
        ).view(n, n, 2)
        # the diagonal of f_samps are the out-of-sample predictions
        # for each LOO model, compare the out of sample predictions to each in-sample prediction
        rank_loss = ((f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^ (
                    cartesian_y[..., 0] < cartesian_y[..., 1])).sum(dim=-1).sum(dim=-1)
    else:
        rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
        y_stack = target_y.squeeze(-1).expand(f_samps.shape)
        for i in range(1, target_y.shape[0]):
            rank_loss += ((roll_col(f_samps, i) < f_samps) ^ (roll_col(y_stack, i) < y_stack)).sum(dim=-1)
    return rank_loss


def get_target_model_loocv_sample_preds(train_x, train_y,  target_model, num_samples):
    """
    Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.
    """
    batch_size = len(train_x)
    masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
    train_x_cv = torch.stack([train_x[~m] for m in masks])
    train_y_cv = torch.stack([train_y[~m] for m in masks])
    #train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
    state_dict = target_model.state_dict()
    # expand to batch size of batch_mode LOOCV model
    state_dict_expanded = {name: t.expand(batch_size, *[-1 for _ in range(t.ndim)]) for name, t in state_dict.items()}
    model = get_fitted_model(train_x_cv, train_y_cv, state_dict=state_dict_expanded)
    with torch.no_grad():
        posterior = model.posterior(train_x)
        # Since we have a batch mode gp and model.posterior always returns an output dimension,
        # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
        # the last dimension.
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        return sampler(posterior).squeeze(-1)


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
        posterior = model.posterior(train_x)
        sampler = SobolQMCNormalSampler(num_samples=num_samples)
        base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
        # compute and save ranking loss
        ranking_losses.append(compute_ranking_loss(base_f_samps, train_y).numpy())
    # compute ranking loss for target model using LOOCV
    # f_samps
    #train_yvar = torch.full_like(train_y, noise_std ** 2)
    target_f_samps = get_target_model_loocv_sample_preds(train_x, train_y, target_model, num_samples)
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
    if train_x.shape[0] > 20 and rank_weights[-1] < 0.5:
        rank_weights[:-1] = rank_weights[:-1] / float(rank_weights[:-1].sum().detach().numpy()) * 0.5
        rank_weights[-1] = 0.5
    rank_weights_dir ={}
    count = 0
    for task in list(base_models.keys()):
        rank_weights_dir[task] = float(rank_weights[count].detach().numpy())
        count = count + 1
    rank_weights_dir['target'] = float(rank_weights[-1].detach().numpy())


    return rank_weights_dir, rank_weights


def initialize_WGPE_model(train_x, train_y, load_dir):
    #train_yvar = torch.full_like(train_y, noise_std ** 2)
    target_model = get_fitted_model(train_x, train_y)
    source_model_dir = load_source_model(load_dir)
    rank_weight_dir, rank_weights = compute_rank_weights(train_x, train_y, source_model_dir, target_model, NUM_POSTERIOR_SAMPLES)
    model_list = []
    for task in list(source_model_dir.keys()):
        model_list.append(source_model_dir[task])

    model_list.append(target_model)
    wgpe_model = WGPE(model_list, rank_weights)
    #log weight
    d_order = sorted(rank_weight_dir.items(), key=lambda x: x[1], reverse=True)
    map_num = sum([w[1]!=0 for w in d_order])
    #present_num = map_num if map_num < 5 else 5
    logger.info("[Wokrload Mapping]: map {} workloads, top weights: {}".format(map_num, d_order[:map_num]))

    return wgpe_model

def get_acqf_wgpe(func_name, model, train_x, train_obj, **kwargs):
    if func_name == 'EI':
        Y_mean = train_obj.mean(dim=-2, keepdim=True)
        Y_std = train_obj.std(dim=-2, keepdim=True)
        train_obj = (train_obj - Y_mean) / Y_std
        EI = ExpectedImprovement(
            model=model,
            best_f=train_obj.max(),
            #X_baseline=train_x,
            maximize=True
        )
        return EI
    elif func_name == 'UCB':
        beta = kwargs['beta'] if 'beta' in kwargs else 0.2
        UCB = UpperConfidenceBound(
            model=model,
            beta=beta
        )
        return UCB
    elif func_name == 'qNEI':
        sampler_qnei = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
        qNEI = qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler_qnei,
        )
        return qNEI
    else:
        return None


def optimize_acqf_and_get_observation_wgpe(acq_func, train_x):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidate, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor([[0.0] * train_x.shape[1], [1.0] * train_x.shape[1]], device=device, dtype=torch.double),
        q=Q_BATCH_SIZE,
        num_restarts=N_RESTARTS,
        raw_samples=N_RESTART_CANDIDATES,
    )

    # fetch the new values
    new_x = candidate.detach()
    return new_x

