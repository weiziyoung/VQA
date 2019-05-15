# -*- coding: utf-8 -*-
# @Time    : 2019/5/11 12:26 AM
# @Author  : weiziyang
# @FileName: counting.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch.autograd import Variable


class PiecewiseLin(nn.Module):
    """
    学习线性分段函数的参数，用于去除intra-object edges,是论文中公式12的实现
    """
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.weight = nn.Parameter(torch.ones(n + 1))
        # the first weight here is always 0 with a 0 gradient
        self.weight.data[0] = 0

    def forward(self, x):
        # all weights are positive -> function is monotonically increasing
        w = self.weight.abs()
        # make weights sum to one -> f(1) = 1
        w = w / w.sum()
        w = w.view([self.n + 1] + [1] * x.dim())
        # keep cumulative sum for O(1) time complexity
        csum = w.cumsum(dim=0)
        csum = csum.expand((self.n + 1,) + tuple(x.size()))
        w = w.expand_as(csum)

        # figure out which part of the function the input lies on
        y = self.n * x.unsqueeze(0)
        idx = Variable(y.long().data)
        f = y.frac()

        # contribution of the linear parts left of the input
        x = csum.gather(0, idx.clamp(max=self.n))
        # contribution within the linear segment the input falls into
        x = x + f * w.gather(0, (idx + 1).clamp(max=self.n))
        return x.squeeze(0)


class Count(nn.Module):
    def __init__(self, object_num):
        super().__init__()
        self.object_num = object_num
        self.linear_function = nn.ModuleList([PiecewiseLin(16) for _ in range(16)])

    def forward(self, boxes, attention):
        # a * a.T
        A = attention


