import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from denoisingAutoencoder import DenoisingAutoencoder
from dec_pytorch.lib.utils import Dataset
from dec_pytorch.lib.datasets import MNIST
from sdae_byol import StackedDAE

batch_size = 256

model = StackedDAE(input_dim=784, z_dim=10, binary=False,
                   encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", byol_dim=128,
                   dropout=0)

train_loader = torch.utils.data.DataLoader(
    MNIST('../dataset/mnist', train=True, download=False),
    batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNIST('../dataset/mnist', train=False),
    batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的权重
pretrain_save_path = "stacked_dae_pretrained.pth"
model.load_model(torch.load(pretrain_save_path, map_location=device))
model.fit(train_loader, test_loader, lr=0.01, num_epochs=1000, alpha=0.1, save_path="model")