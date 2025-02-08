import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
import math


class MSELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return 0.5 * torch.mean((input-target)**2)

class BCELoss(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, input, target):
        return -torch.mean(torch.sum(target*torch.log(torch.clamp(input, min=1e-10))+
            (1-target)*torch.log(torch.clamp(1-input, min=1e-10)), 1))

class BYOLLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(BYOLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, pred_online, proj_target):
        """
        计算 BYOL 对比损失。
        :param pred_online: 在线网络的预测特征 (batch_size, feature_dim)
        :param proj_target: 目标网络的投影特征 (batch_size, feature_dim)
        :return: 对比损失
        """
        # 归一化特征
        pred_norm = F.normalize(pred_online, dim=1)
        target_norm = F.normalize(proj_target, dim=1)

        # 计算余弦相似度
        similarity = (pred_norm * target_norm).sum(dim=1)

        # 计算对比损失
        contrastive_loss = 2 - 2 * similarity.mean()
        return contrastive_loss
