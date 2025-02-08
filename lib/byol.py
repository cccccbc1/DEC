import torch
import torch.nn as nn
import copy

class BYOL(nn.Module):
    def __init__(self, base_encoder, projection_dim=256):
        super(BYOL, self).__init__()
        self.online_network = base_encoder
        self.target_network = copy.deepcopy(base_encoder)
        # 禁用梯度计算以保持目标网络参数不变
        for param in self.target_network.parameters():
            param.requires_grad = False

        # 添加投影头
        self.projection = nn.Sequential(
            nn.Linear(base_encoder.layers[-1], projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        self.prediction = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def update_moving_average(self, beta=0.99):
        for online_params, target_params in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_params.data = beta * target_params.data + (1 - beta) * online_params.data

    def forward(self, x1, x2):
        z1 = self.projection(self.online_network(x1))
        q1 = self.prediction(z1)
        with torch.no_grad():
            z2 = self.projection(self.target_network(x2))
        return q1, z2