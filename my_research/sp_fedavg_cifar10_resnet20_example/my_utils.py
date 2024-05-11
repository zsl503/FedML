from collections import OrderedDict
import copy
from datetime import datetime
import logging
import math
import os
from typing import List
import numpy as np

TIME = datetime.now().strftime('%Y%m%d%H%M')


class LayerFilter:

    def __init__(self,
                 unselect_keys: List[str] = None,
                 all_select_keys: List[str] = None,
                 any_select_keys: List[str] = None) -> None:
        self.update_filter(unselect_keys, all_select_keys, any_select_keys)

    def update_filter(self,
                      unselect_keys: List[str] = None,
                      all_select_keys: List[str] = None,
                      any_select_keys: List[str] = None):
        self.unselect_keys = unselect_keys if unselect_keys is not None else []
        self.all_select_keys = all_select_keys if all_select_keys is not None else []
        self.any_select_keys = any_select_keys if any_select_keys is not None else []

    def __call__(self, param_dict):
        if len(self.unselect_keys + self.all_select_keys +
               self.any_select_keys) == 0:
            return param_dict
        return {
            layer_key: param
            for layer_key, param in param_dict.items()
            if (len(self.unselect_keys) == 0 or all(
                key not in layer_key for key in self.unselect_keys)) and (
                    len(self.all_select_keys) == 0 or all(
                        key in layer_key
                        for key in self.all_select_keys)) and (
                            len(self.any_select_keys) == 0 or any(
                                key in layer_key
                                for key in self.any_select_keys))
        }

    def __str__(self) -> str:
        return f"unselect_keys:{self.unselect_keys}  all_select_keys:{self.all_select_keys}  any_select_keys:{self.any_select_keys}"


def aggregate_layer(w_locals, layer_name):
    training_num = 0
    for local_sample_number, local_model_params in w_locals:
        training_num += local_sample_number

    (sample_num, averaged_params) = w_locals[0]
    averaged_layer = averaged_params[layer_name] * sample_num / training_num
    for local_sample_number, local_model_params in w_locals:
        w = local_sample_number / training_num
        averaged_layer += local_model_params[layer_name] * w

    return averaged_layer


def get_cka_matrix(w_list, layer_name):
    cka_matrix = np.eye(len(w_list))
    for i, (_, w_i) in enumerate(w_list):
        dim = len(w_i[layer_name].shape)
        if dim == 0:
            continue
        for j, (_, w_j) in enumerate(w_list):
            if dim == 4:
                cka_matrix[i, j] = CKA(w_i[layer_name].mean(dim=[-1,-2]), w_j[layer_name].mean(dim=[-1,-2]))
            else:
                cka_matrix[i, j] = CKA(w_i[layer_name], w_j[layer_name])
            
    return cka_matrix


def topk_indices(row, k):
    return np.argpartition(row, -k)[-k:]


def save_model_param(model_params,
                     round_idx,
                     path_tag,
                     pre_desc=None,
                     post_desc=None,
                     is_grad=True):
    # 保存全局权重，作为实验
    if pre_desc is None:
        pre_desc = path_tag

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'grad_lists' if is_grad else 'weight_lists', TIME,
                            path_tag)
    os.makedirs(save_dir, exist_ok=True)
    if post_desc is None:
        path = os.path.join(save_dir, f'{pre_desc}_round_{round_idx}.pt')
    else:
        path = os.path.join(save_dir,
                            f'{pre_desc}_round_{round_idx}_{post_desc}.pt')

    logging.info(f"Save {path_tag} {round_idx} model params to '{path}'.")
    torch.save(model_params, path)


def weight_sub(weight_x, weight_y):
    """
    Calculate the difference between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
    Returns:
        dict: weight_x - weight_y
    """
    device = next(iter(weight_x.values())).device
    # Create a new dictionary to store the weight differences
    weight_diff = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        # Compute the difference between corresponding weight tensors
        diff = weight_x[key].to(device) - weight_y[key].to(device)
        # Store the difference in the weight_diff dictionary
        weight_diff[key] = diff
    return weight_diff


def weight_add(weight_x, weight_y):
    """
    Calculate the addition result between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
    Returns:
        dict: weight_x + weight_y
    """
    # Create a new dictionary to store the weight differences
    device = next(iter(weight_x.values())).device
    weight_add = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        # Compute the difference between corresponding weight tensors
        weight_add[key] = weight_x[key].to(device) + weight_y[key].to(device)
    return weight_add


def get_model_gradient(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict
    
    Args:
        - model: (torch.nn.Module), torch model
    
    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = grad
    return grads


def linear_kernel(X, Y):
    return np.matmul(X, Y.transpose(0, 1))


def rbf(X, Y, sigma=None):
    """
    Radial-Basis Function kernel for X and Y with bandwith chosen
    from median if not specified.
    """
    GX = np.dot(X, Y.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def HSIC(K, L):
    """
    Calculate Hilbert-Schmidt Independence Criterion on K and L.
    """
    n = K.shape[0]
    H = np.identity(n) - (1. / n) * np.ones((n, n))

    KH = np.matmul(K, H)
    LH = np.matmul(L, H)
    return 1. / ((n - 1)**2) * np.trace(np.matmul(KH, LH))


def CKA(X, Y, kernel=None):
    """
    Calculate Centered Kernel Alingment for X and Y. If no kernel
    is specified, the linear kernel will be used.
    """
    kernel = linear_kernel if kernel is None else kernel
    if len(X.shape) == 1:
        X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
    K = kernel(X, X)
    L = kernel(Y, Y)

    hsic = HSIC(K, L)
    varK = np.sqrt(HSIC(K, K))
    varL = np.sqrt(HSIC(L, L))
    return hsic / (varK * varL)


# -------------------第二种cka计算方法------------
from torch import Tensor
import torch


def centering(k: Tensor, inplace: bool = True) -> Tensor:
    if not inplace:
        k = torch.clone(k)
    means = k.mean(dim=0)
    means -= means.mean() / 2
    k -= means.view(-1, 1)
    k -= means.view(1, -1)
    return k


def linear_hsic(k: Tensor, l: Tensor, unbiased: bool = True) -> Tensor:
    assert k.shape[0] == l.shape[0], 'Input must have the same size'
    m = k.shape[0]
    if unbiased:
        k.fill_diagonal_(0)
        l.fill_diagonal_(0)
        kl = torch.matmul(k, l)
        score = torch.trace(kl) + k.sum() * l.sum() / (
            (m - 1) * (m - 2)) - 2 * kl.sum() / (m - 2)
        return score / (m * (m - 3))
    else:
        k, l = centering(k), centering(l)
        return (k * l).sum() / ((m - 1)**2)


def cka_score(x1: Tensor, x2: Tensor, gram: bool = False) -> Tensor:
    assert x1.shape[0] == x2.shape[0], 'Input must have the same batch size'
    if len(x1.shape) == 1:
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
    if not gram:
        x1 = torch.matmul(x1, x1.transpose(0, 1))
        x2 = torch.matmul(x2, x2.transpose(0, 1))
    cross_score = linear_hsic(x1, x2)
    self_score1 = linear_hsic(x1, x1)
    self_score2 = linear_hsic(x2, x2)
    return cross_score / torch.sqrt(self_score1 * self_score2)


# -----------------------余弦相似度计算--------------------
def cos_similar(x1: Tensor, x2: Tensor):
    x1 = x1.flatten()
    x2 = x2.flatten()
    return torch.sum(x1 * x2) / (torch.norm(x1) * torch.norm(x2))
