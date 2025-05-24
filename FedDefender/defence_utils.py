import numpy as np
import torch
import torch.nn as nn
import copy
from functools import reduce
import math


def get_krum(inputs):
    '''
    compute krum or multi-krum of input. O(dn^2)
    '''
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    n = inputs.shape[-1]
    f = n // 10  # 10% malicious points
    k = n - f - 2
    x = inputs.permute(0, 2, 1)
    cdist = torch.cdist(x, x, p=2)
    nbhDist, nbh = torch.topk(cdist, k + 1, largest=False)
    i_star = torch.argmin(nbhDist.sum(2))
    mkrum = inputs[:, :, nbh[:, i_star, :].view(-1)].mean(2, keepdims=True)
    return mkrum, nbh[:, i_star, :].view(-1)


def get_norm(inputs):
    '''
    compute krum or multi-krum of input. O(dn^2)
    '''
    number_to_consider = 8
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)
    x = inputs.permute(0, 2, 1)
    norm = x.norm(2, dim=-1, keepdim=True).view(-1)
    sorted_norm, sorted_idx = torch.sort(norm)
    used_idx = sorted_idx[:number_to_consider]
    global_weight = torch.mean(x[:, used_idx, :], dim=1).view(-1)
    return global_weight, used_idx


def median_opt(input):
    shape = input.shape
    input = input.sort()[0]
    if shape[-1] % 2 != 0:
        return input[..., (shape[-1] - 1) // 2]
    else:
        mid = shape[-1] // 2
        return (input[..., mid-1] + input[..., mid]) / 2.0


def repeated_median(y):
    eps = np.finfo(float).eps
    num_models = y.shape[1]
    total_num = y.shape[0]
    y_sorted = y.sort()[0]
    yyj = y_sorted.repeat(1, 1, num_models).reshape(total_num, num_models, num_models)
    yyi = yyj.transpose(-1, -2)
    xx = torch.arange(num_models, dtype=torch.float, device=y.device)
    xxj = xx.repeat(total_num, num_models, 1)
    xxi = xxj.transpose(-1, -2) + eps
    diag_inf = torch.diag(torch.tensor([float('Inf')] * num_models, device=y.device)).repeat(total_num, 1, 1)
    dividor = xxi - xxj + diag_inf
    slopes = (yyi - yyj) / dividor + diag_inf
    slopes, _ = slopes.sort()
    slopes = median_opt(slopes[:, :, :-1])
    slopes = median_opt(slopes)
    yy_median = median_opt(y_sorted)
    xx_median = torch.full((total_num,), (num_models - 1) / 2.0, device=y.device)
    intercepts = yy_median - slopes * xx_median
    return slopes, intercepts


def reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    slopes, intercepts = repeated_median(y)
    X_pure = y.sort()[1].sort()[1].type(torch.float).unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1, device=y.device), X_pure), dim=-1)
    X_X = X.transpose(1, 2) @ X
    X_X = X @ torch.inverse(X_X)
    H = X_X @ X.transpose(1, 2)
    diag = torch.eye(num_models, device=y.device).repeat(total_num, 1, 1)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.tensor(LAMBDA * np.sqrt(2. / num_models), device=y.device)
    # construis d’abord deux matrices [N×M] à partir de tes vecteurs, puis empile
    ints = intercepts.unsqueeze(1).repeat(1, num_models)  # (N×M)
    slps =    slopes.unsqueeze(1).repeat(1, num_models)    # (N×M)
    beta = torch.stack([ints, slps], dim=2)                # (N×M×2)

    X_expanded = X
    line_y = (beta * X_expanded).sum(dim=-1)
    residual = y - line_y
    M = median_opt(residual.abs().sort()[0][..., 1:])
    tau = 1.4826 * (1 + 5 / (num_models - 1)) * M + 1e-7
    e = residual / tau.repeat(num_models, 1).transpose(0,1)
    reweight = processed_H / e * torch.clamp(e / processed_H, -K, K)
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)
    reweight_regulized = reweight * reweight_std.repeat(num_models,1).transpose(0,1)
    restricted_y = y * (reweight >= thresh) + line_y * (reweight < thresh)
    return reweight_regulized, restricted_y


def median_reweight_algorithm_restricted(y, LAMBDA, thresh):
    num_models = y.shape[1]
    total_num = y.shape[0]
    y_sorted = y.sort()[0]
    X_pure = y_sorted.sort()[1].sort()[1].type(torch.float).unsqueeze(2)
    X = torch.cat((torch.ones(total_num, num_models, 1, device=y.device), X_pure), dim=-1)
    X_X = X.transpose(1, 2) @ X
    X_X = X @ torch.inverse(X_X)
    H = X_X @ X.transpose(1, 2)
    diag = torch.eye(num_models, device=y.device).repeat(total_num,1,1)
    processed_H = (torch.sqrt(1 - H) * diag).sort()[0][..., -1]
    K = torch.tensor(LAMBDA * np.sqrt(2. / num_models), device=y.device)
    y_med = median_opt(y_sorted).unsqueeze(1).repeat(1,num_models)
    residual = y - y_med
    M = median_opt(residual.abs().sort()[0][...,1:])
    tau = 1.4826 * (1 + 5/(num_models-1)) * M + 1e-7
    e = residual / tau.repeat(num_models,1).transpose(0,1)
    reweight = processed_H / e * torch.clamp(e/processed_H, -K, K)
    reweight[reweight != reweight] = 1
    reweight_std = reweight.std(dim=1)
    reweight_regulized = reweight * reweight_std.repeat(num_models,1).transpose(0,1)
    restricted_y = y * (reweight >= thresh) + y_med * (reweight < thresh)
    return reweight_regulized, restricted_y


def IRLS_median_split_restricted(w_locals, LAMBDA=2, thresh=0.1, mode='median'):
    SHARD_SIZE = 2000
    w = []
    for net in w_locals.values():
        w.append(net.state_dict())
    # Clone to detach from any graph
    w_med = {k: v.clone().detach() for k,v in w[0].items()}
    device = next(iter(w_med.values())).device
    reweight_sum = torch.zeros(len(w), device=device)
    for k in w_med.keys():
        shape = w_med[k].shape
        if not shape:
            continue
        total_num = reduce(lambda x,y: x*y, shape)
        y_list = torch.stack([w[i][k].view(-1) for i in range(len(w))])
        transposed_y = y_list.transpose(0,1)
        y_result = torch.zeros_like(transposed_y)
        if total_num < SHARD_SIZE:
            reweight, restricted_y = median_reweight_algorithm_restricted(transposed_y, LAMBDA, thresh)
            reweight_sum += reweight.sum(dim=0)
            y_result = restricted_y
        else:
            num_shards = math.ceil(total_num/SHARD_SIZE)
            for i in range(num_shards):
                shard = transposed_y[i*SHARD_SIZE:(i+1)*SHARD_SIZE]
                r, ry = median_reweight_algorithm_restricted(shard, LAMBDA, thresh)
                reweight_sum += r.sum(dim=0)
                y_result[i*SHARD_SIZE:(i+1)*SHARD_SIZE] = ry
        y_result = y_result.transpose(0,1)
        for i in range(len(w)):
            w[i][k] = y_result[i].view(shape).to(device)
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)
    return w_med, reweight


def weighted_average(w_list, weights):
    # Clone first element to detach
    w_avg = {k: v.clone().detach() for k,v in w_list[0].items()}
    weights = weights / weights.sum()
    assert len(weights) == len(w_list)
    for k in w_avg.keys():
        w_avg[k] = sum(w_list[i][k] * weights[i] for i in range(len(w_list)))
    return w_avg, weights


import torch
import math
from functools import reduce

def IRLS_aggregation_split_restricted(w_locals, LAMBDA=2, thresh=0.1):
    SHARD_SIZE = 2000
    # Récupère directement les state_dicts
    w = list(w_locals.values())
    # initialise w_med par clonage pour détacher du graphe
    w_med = {k: v.clone().detach() for k, v in w[0].items()}
    device = next(iter(w_med.values())).device
    reweight_sum = torch.zeros(len(w), device=device)

    for k in w_med.keys():
        shape = w_med[k].shape
        # saute les scalaires
        if shape == () or 0 in shape:
            continue

        # mets en forme [num_clients × total_num_params]
        total_num = reduce(lambda a, b: a * b, shape)
        y_list = torch.stack([w[i][k].view(-1) for i in range(len(w))], dim=0)
        transposed_y = y_list.transpose(0, 1)  # [total_num × num_clients]
        y_result = torch.zeros_like(transposed_y, device=transposed_y.device)

        if total_num < SHARD_SIZE:
            # tout le shard sur GPU → si ça plante, basculer en CPU
            y_shard = transposed_y
            # bascule sur CPU pour IRLS
            y_cpu = y_shard.cpu()
            r_cpu, ry_cpu = reweight_algorithm_restricted(y_cpu, LAMBDA, thresh)
            # renvoie sur GPU
            r  = r_cpu.to(device)
            ry = ry_cpu.to(device)
            reweight_sum += r.sum(dim=0)
            y_result = ry

        else:
            # traitement shard par shard
            num_shards = math.ceil(total_num / SHARD_SIZE)
            for i in range(num_shards):
                start = i * SHARD_SIZE
                end   = min(start + SHARD_SIZE, total_num)
                y_shard = transposed_y[start:end]     # [shard_size × num_clients]
                # bascule sur CPU
                y_cpu = y_shard.cpu()
                r_cpu, ry_cpu = reweight_algorithm_restricted(y_cpu, LAMBDA, thresh)
                # renvoie sur GPU
                r  = r_cpu.to(device)
                ry = ry_cpu.to(device)
                reweight_sum += r.sum(dim=0)
                y_result[start:end] = ry

        # remet dans la forme originelle et met à jour chaque w[i][k]
        y_result = y_result.transpose(0, 1)  # [num_clients × total_num]
        for i in range(len(w)):
            w[i][k] = y_result[i].view(shape).to(device)

    # normalise les poids et calcule la moyenne pondérée
    reweight_sum = reweight_sum / reweight_sum.max()
    reweight_sum = reweight_sum * reweight_sum
    w_med, reweight = weighted_average(w, reweight_sum)

    return w_med, reweight


def get_weight(model_weight):
    weight_list = [v.view(-1).float() for v in model_weight.values()]
    return torch.cat(weight_list)
