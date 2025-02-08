import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from sdae_byol import StackedDAE
from dec_pytorch.lib.utils import Dataset
from dec_pytorch.lib.datasets import MNIST
from dec_pytorch.lib.ops import MSELoss, BCELoss

batch_size = 256

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
])
# train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
train_loader = torch.utils.data.DataLoader(
    MNIST('../dataset/mnist', train=True, download=False),
    batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(
    MNIST('../dataset/mnist', train=False),
    batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Using device: {torch.cuda.current_device()}" if torch.cuda.is_available() else "Using CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型
model = StackedDAE(input_dim=784, z_dim=10, binary=False,
                   encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", byol_dim=128,
                   dropout=0)
model = model.to(device)
# 训练
model.pretrain(train_loader, test_loader, lr=0.1, batch_size=batch_size,
              num_epochs=150, corrupt=0.2, loss_type="mse")
pretrain_save_path = "./stacked_dae_pretrained.pt"
torch.save(model.state_dict(), pretrain_save_path)

# model = StackedDAE(input_dim=784, z_dim=10, binary=False,
#                    encodeLayer=[500, 500, 2000], decodeLayer=[2000, 500, 500], activation="relu", byol_dim=128,
#                    dropout=0)
# # 加载预训练的权重
# pretrain_load_path = "stacked_dae_pretrained.pth"
# model.load_state_dict(torch.load(pretrain_save_path, map_location=device))
model.fit(train_loader, test_loader, lr=0.05, num_epochs=10000, corrupt=0.2, alpha=0.05, save_path="./train_model/model")

