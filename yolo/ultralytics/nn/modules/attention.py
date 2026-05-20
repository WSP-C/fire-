import torch
import torch.nn as nn


class SimAM(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        b, c, h, w = x.size()
        n = w * h - 1

        # 计算每个像素与其均值的差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)

        # 计算能量函数 e_t
        # 公式: e_t = ((t - mu)^2) / (4 * (sigma^2 + lambda)) + 0.5
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        # 通过 Sigmoid 限制权重范围，并作用于原特征图
        return x * self.activaton(y)