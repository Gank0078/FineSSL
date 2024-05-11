import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from matplotlib import rcParams
from sklearn.manifold import TSNE
import os

config = {
    "font.family":'Arial',
    "font.size": 18,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)


class _ECELoss(nn.Module):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    # def forward(self, logits, labels, t=1.0):

    def forward(self, preds, targets, confs, t=1.0):
        # softmaxes = F.softmax(logits/t, dim=1)
        # confidences, predictions = torch.max(softmaxes, 1)
        confidences, predictions, labels = confs, preds, targets
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=preds.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def entropy(x, mean=False):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """
    EPS = 1e-8
    x_ = torch.clamp(x, min = EPS)
    b = x_ * torch.log(x_)

    if len(b.size()) == 2:  # Sample-wise entropy
        if mean is True:
            return - b.sum(dim=1).mean()
        else:
            return - b.sum(dim=1)
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()

    def forward(self, x, target, weight=None):
        if weight is not None:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1) * weight
        else:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shoot_infs(inp_tensor):
    """Replaces inf by maximum of tensor"""
    mask_inf = torch.isinf(inp_tensor)
    ind_inf = torch.nonzero(mask_inf)
    if len(ind_inf) > 0:
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = 0
            elif len(ind) == 1:
                inp_tensor[ind[0]] = 0
        m = torch.max(inp_tensor)
        for ind in ind_inf:
            if len(ind) == 2:
                inp_tensor[ind[0], ind[1]] = m
            elif len(ind) == 1:
                inp_tensor[ind[0]] = m
    return inp_tensor
