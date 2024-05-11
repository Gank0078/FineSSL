import os
import numpy as np
import torch
from math import inf
from scipy import stats
import torch.nn.functional as F
import torch.nn as nn


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size=3*224*224, norm_std=0.1, seed=0):
    # n -> noise_rate
    # dataset -> mnist, cifar10 # not train_loader
    # labels -> labels (targets)
    # label_num -> class number
    # feature_size -> the size of input images (e.g. 28*28)
    # norm_std -> default 0.1
    # seed -> random_seed
    print("building dataset...")
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm((0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W).cuda()
    for i, (x, y, _) in enumerate(dataset):
        # 1*m *  m*10 = 1*10
        x = x.cuda()
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).cpu().numpy()
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    # record = [[0 for _ in range(label_num)] for i in range(label_num)]
    #
    # for a, b in zip(labels, new_label):
    #     a, b = int(a), int(b)
    #     record[a][b] += 1
    #
    # pidx = np.random.choice(range(P.shape[0]), 1000)
    # cnt = 0
    # for i in range(1000):
    #     if labels[pidx[i]] == 0:
    #         a = P[pidx[i], :]
    #         cnt += 1
    #     if cnt >= 10:
    #         break
    return np.array(new_label)
