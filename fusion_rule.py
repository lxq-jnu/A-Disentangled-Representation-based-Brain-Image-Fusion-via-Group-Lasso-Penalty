import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from math import exp
from collections import OrderedDict



class ChannelPool(nn.Module):
    def forward(self, x):  # 计算最大池化和平均池化，并连接
        return (torch.max(x, 1)[0].unsqueeze(1) + torch.max(x, 1)[0].unsqueeze(1))


class SD(nn.Module):
    def __init__(self, lam=1, size=5, stride=1, pad=2, C=9e-4):
        super().__init__()
        self.lam = lam
        self.size = size
        self.stride = stride
        self.C = C
        self.pad = pad

    def forward(self, IA):
        # IA = nn.ReflectionPad2d(5)(IA)

        window = torch.ones((1, 1, self.size, self.size)) / (self.size * self.size)
        window = window.to(IA.device)
        #print(IA.shape)

        mean_IA = F.conv2d(IA, window, padding=self.pad, stride=self.stride)

        mean_IA_2 = F.conv2d(torch.pow(IA, 2), window, padding=self.pad, stride=self.stride)

        var_IA = mean_IA_2 - torch.pow(mean_IA, 2)

        # print(var_IA)
        # print(var_IB)
        # print(mean_IA)
        # print(mean_IB)

        return var_IA, mean_IA


class sigmaAB(nn.Module):
    def __init__(self, lam=1, size=5, stride=1, C=9e-4):
        super().__init__()
        self.lam = lam
        self.size = size
        self.stride = stride
        self.C = C
        self.pad = 0

    def forward(self, IA, IB):
        IA = nn.ReflectionPad2d(2)(IA)
        IB = nn.ReflectionPad2d(2)(IB)

        window = torch.ones((1, 1, self.size, self.size)) / (self.size * self.size)
        window = window.to(IA.device)

        mean_IA = F.conv2d(IA, window, stride=self.stride, padding=self.pad)
        mean_IB = F.conv2d(IB, window, stride=self.stride, padding=self.pad)

        mean_IA_2 = F.conv2d(torch.pow(IA, 2), window, stride=self.stride, padding=self.pad)
        mean_IB_2 = F.conv2d(torch.pow(IB, 2), window, stride=self.stride, padding=self.pad)

        var_IA = mean_IA_2 - torch.pow(mean_IA, 2)
        var_IB = mean_IB_2 - torch.pow(mean_IB, 2)

        # print(var_IA)
        # print(var_IB)
        # print(mean_IA)
        # print(mean_IB)

        mean_IAIB = F.conv2d(IA * IB, window, stride=self.stride, padding=self.pad)

        sigma_IAIB = (mean_IAIB - mean_IA * mean_IB) / (torch.sqrt(var_IA + self.C) * torch.sqrt(var_IB + self.C))

        return sigma_IAIB


class fusion(nn.Module):
    def __init__(self, lam=1, size=3, stride=1, pad=1, C=9e-4):
        super().__init__()
        self.lam = lam
        self.size = size
        self.stride = stride
        self.C = C
        self.pad = pad
        self.sd = SD()
        self.ch_pool = ChannelPool()
        self.cor = sigmaAB()

    def forward(self, f_A, f_B):


        cp_A = torch.sum(torch.abs(f_A), 1).unsqueeze(1)
        cp_B = torch.sum(torch.abs(f_B), 1).unsqueeze(1)


        sd_A, m_A = self.sd(cp_A)
        sd_B, m_B = self.sd(cp_B)



        w2 = m_A / (m_A + m_B)


        final_fuse = w2*f_A + (1-w2) * f_B


        return final_fuse




class LaplacianConv(nn.Module):
    # 仅有一个参数，通道，用于自定义算子模板的通道
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展到3个维度
        kernel = np.repeat(kernel, self.channels, axis=0)  # 3个通道都是同一个算子
        #self.pad = nn.ReflectionPad2d(1)
        self.pad = nn.ReplicationPad2d(1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 不允许求导更新参数，保持常量

    def __call__(self, x):
        # 第一个参数为输入，由于这里只是测试，随便打开的一张图只有3个维度，pytorch需要处理4个维度的样本，因此需要添加一个样本数量的维度
        # padding2是为了能够照顾到边角的元素
        self.weight.data = self.weight.data.to(x.device)
        x = self.pad(x)

        x = F.conv2d(x, self.weight, padding=0, groups=self.channels)
        return x


