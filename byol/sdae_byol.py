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
from dec_pytorch.lib.ops import MSELoss, BCELoss
from sklearn.cluster import KMeans
from dec_pytorch.lib import metrics
import scipy.io as sio
from torchvision import transforms
from torch.utils.data import DataLoader



batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 构建全连接网络
def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)


# 生成掩码噪声
def masking_noise(data, corrupt_prob):
    mask = torch.rand_like(data) > corrupt_prob  # 生成掩码
    noisy_data = data * mask.float()  # 应用掩码
    return noisy_data

def masking_noise2(data, shift_range=5, shape=(1, 28, 28)):
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # 随机旋转-10到+10度
        transforms.Pad(padding=2),  # 填充2个像素宽的边框，使得可以向任意方向平移最多2个像素
        transforms.RandomCrop(size=28),  # 从填充后的图像中随机裁剪28x28大小的图像
        transforms.ToTensor(),  # 转换成Tensor，并且数值归一化到[0.0, 1.0]
    ])
    return datasets.ImageFolder(data, transform=transform)


class StackedDAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
                 encodeLayer=None, decodeLayer=None, activation="relu",
                 dropout=0, tied=False, byol_dim=128, momentum=0.996):
        super().__init__()
        if decodeLayer is None:
            decodeLayer = [2000, 500, 500]
        if encodeLayer is None:
            encodeLayer = [500, 500, 2000]
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout

        # 原始 SDAE 编码器和解码器
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = nn.Sigmoid() if binary else None

        # BYOL 组件
        self.online_projector = nn.Sequential(
            nn.Linear(z_dim, 256),  # 输入维度为 z_dim
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, byol_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(byol_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, byol_dim)
        )

        # 目标网络（动量更新）
        self.target_encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.target_projector = nn.Sequential(
            nn.Linear(z_dim, 256),  # 输入维度为 z_dim，与 online_projector 一致
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, byol_dim)
        )
        self._init_target_network()
        self.momentum = momentum

    def _init_target_network(self):
        # 初始同步目标网络参数
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        # 冻结目标网络参数
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 编码器
        h = self.encoder(x)
        z = self._enc_mu(h)
        # 解码器
        x_recon = self.decode(z)

        # BYOL 在线网络
        proj_online = self.online_projector(z)
        pred_online = self.predictor(proj_online)

        # BYOL 目标网络（不计算梯度）
        with torch.no_grad():
            h_target = self.target_encoder(x)
            z_target = self._enc_mu(h_target)
            proj_target = self.target_projector(z_target)

        return x_recon, pred_online, proj_target

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    @torch.no_grad()
    def update_target_network(self):
        # 动量更新目标网络参数
        for online_param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = self.momentum * target_param.data + (1 - self.momentum) * online_param.data

    def loss_function(self, recon_x, x, pred_online, proj_target, alpha=0.1):
        # 重构损失（交叉熵或 MSE）
        if self._dec_act is not None:  # 二值数据用交叉熵
        #     recon_loss = -torch.mean(torch.sum(x * torch.log(recon_x + 1e-10) +
        #                                        (1 - x) * torch.log(1 - recon_x + 1e-10), dim=1))
            recon_loss = 0.5 * torch.mean((recon_x - x) ** 2)
        else:  # 连续数据用 MSE
            recon_loss = F.mse_loss(recon_x, x)

        # BYOL 对比损失（余弦相似度）
        pred_norm = F.normalize(pred_online, dim=1)
        target_norm = F.normalize(proj_target, dim=1)
        contrastive_loss = 2 - 2 * (pred_norm * target_norm).sum(dim=1).mean()

        # 联合损失
        total_loss = recon_loss + alpha * contrastive_loss
        return total_loss, recon_loss, contrastive_loss

    def pretrain(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.2,
                 loss_type="cross-entropy"):
        trloader = trainloader
        valoader = validloader
        daeLayers = []
        for l in range(1, len(self.layers)):
            infeatures = self.layers[l - 1]
            outfeatures = self.layers[l]
            if l != len(self.layers) - 1:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation=self.activation, dropout=corrupt)
            else:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation="none", dropout=0)
            print(dae)
            if l == 1:
                dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                        loss_type=loss_type)
            else:
                if self.activation == "sigmoid":
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                            loss_type="cross-entropy")
                else:
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt,
                            loss_type="mse")
            data_x = dae.encodeBatch(trloader)
            valid_x = dae.encodeBatch(valoader)
            trainset = Dataset(data_x, data_x)
            trloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=0)
            validset = Dataset(valid_x, valid_x)
            valoader = torch.utils.data.DataLoader(
                validset, batch_size=1000, shuffle=False, num_workers=0)
            daeLayers.append(dae)

        self.copyParam(daeLayers)

    def copyParam(self, daeLayers):
        if self.dropout == 0:
            every = 2
        else:
            every = 3
        # input layer
        # copy encoder weight
        self.encoder[0].weight.data.copy_(daeLayers[0].weight.data)
        self.encoder[0].bias.data.copy_(daeLayers[0].bias.data)
        self._dec.weight.data.copy_(daeLayers[0].deweight.data)
        self._dec.bias.data.copy_(daeLayers[0].vbias.data)

        for l in range(1, len(self.layers) - 2):
            # copy encoder weight
            self.encoder[l * every].weight.data.copy_(daeLayers[l].weight.data)
            self.encoder[l * every].bias.data.copy_(daeLayers[l].bias.data)

            # copy decoder weight
            self.decoder[-(l - 1) * every - 2].weight.data.copy_(daeLayers[l].deweight.data)
            self.decoder[-(l - 1) * every - 2].bias.data.copy_(daeLayers[l].vbias.data)

        # z layer
        self._enc_mu.weight.data.copy_(daeLayers[-1].weight.data)
        self._enc_mu.bias.data.copy_(daeLayers[-1].bias.data)
        self.decoder[0].weight.data.copy_(daeLayers[-1].deweight.data)
        self.decoder[0].bias.data.copy_(daeLayers[-1].vbias.data)

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10, corrupt=0.2,
            loss_type="mse", alpha=0.1, save_path=None):
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        self.to(device)  # 将模型移动到选定的设备
        if loss_type == "mse":
            criterion = MSELoss()
        elif loss_type == "cross-entropy":
            criterion = BCELoss()

        # 初始验证
        self.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for inputs, _ in validloader:
                inputs = inputs.view(inputs.size(0), -1).float().to(device)
                recon_x, _, _ = self.forward(inputs)
                valid_loss += criterion(recon_x, inputs).item() * inputs.size(0)
        print(f"#Epoch 0: Valid Loss: {valid_loss / len(validloader.dataset):.4f}")

        ACC_all=[]
        NMI_all=[]
        LOSS_all = []

        for epoch in range(num_epochs):
            self.train()
            total_loss, total_recon, total_contrast = 0.0, 0.0, 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float().to(device)
                # 生成两个增强视图（使用 masking_noise）
                # x1 = masking_noise(inputs, corrupt).to(device)  # 增强视图1：加噪声
                # x2 = masking_noise(inputs, corrupt).to(device)  # 增强视图2：加噪声
                x2 = inputs.to(device)  # 增强视图1：加噪声
                x1 = masking_noise(inputs, corrupt).to(device)  # 增强视图2：加噪声

                if torch.cuda.is_available():
                    x1, x2 = x1.cuda(), x2.cuda()
                x1, x2 = Variable(x1), Variable(x2)

                optimizer.zero_grad()

                # 前向传播
                recon_x1, pred_online1, proj_target2 = self.forward(x1)
                recon_x2, pred_online2, proj_target1 = self.forward(x2)

                # 计算损失
                loss1, recon_loss1, contrast_loss1 = self.loss_function(recon_x1, x1, pred_online1, proj_target2, alpha)
                loss2, recon_loss2, contrast_loss2 = self.loss_function(recon_x2, x2, pred_online2, proj_target1, alpha)
                total_batch_loss = (loss1 + loss2) / 2

                # 反向传播
                total_batch_loss.backward()
                optimizer.step()

                # 动量更新目标网络
                self.update_target_network()

                # 记录损失
                total_loss += total_batch_loss.item() * inputs.size(0)
                total_recon += (recon_loss1 + recon_loss2).item() * inputs.size(0)
                total_contrast += (contrast_loss1 + contrast_loss2).item() * inputs.size(0)

            # 验证阶段
            self.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for inputs, _ in validloader:
                    inputs = inputs.view(inputs.size(0), -1).float().to(device)
                    recon_x, _, _ = self.forward(inputs)
                    valid_loss += criterion(recon_x, inputs).item() * inputs.size(0)

            # 在子空间做聚类
            # z_all = []
            # label_true = []
            # for batch_idx, (inputs, target) in enumerate(trainloader):
            #     inputs = inputs.view(inputs.size(0), -1).float()
            #     inputs = inputs.cuda()
            #     inputs = Variable(inputs)
            #     z, outputs, _ = self.forward(inputs)
            #     z1 = z.data.cpu().numpy()  # 直接转换为NumPy数组
            #     z_all.append(z1)
            #     label_true.append(target.cpu().numpy())  # 保持为列表，稍后处理
            #
            # # 将列表转换为大数组之前，先展平或调整形状
            # z_all = np.vstack(z_all)  # 假设z的形状适合垂直堆叠
            # label_true = np.concatenate(label_true)  # 连接所有批次的标签
            #
            # # 确保z_all和label_true的形状符合预期
            # # print("Shape of z_all:", z_all.shape)
            # # print("Shape of label_true:", label_true.shape)
            #
            # ##kmeans result
            # estimator = KMeans(10)
            # estimator.fit(z_all)
            # centroids = estimator.cluster_centers_
            # label_pred = estimator.labels_
            # acc = metrics.acc(label_true, label_pred)
            # nmi = metrics.nmi(label_true, label_pred)
            # print("acc:", acc)
            # print("nmi", nmi)
            # ACC_all.append(acc)
            # NMI_all.append(nmi)
            # if acc > 0.85:
            #     sdae_savepath = ("./model/csdae-%d.pt" % epoch)
            #     self.save_model(sdae_savepath)
            #     path = './Mat/our_sdae_subspace' + str(epoch)
            #     sio.savemat(path + '.mat', {'feature': z_all, 'true': label_true, 'pred': label_pred})

            # 打印训练信息
            avg_loss = total_loss / len(trainloader.dataset)
            avg_recon = total_recon / len(trainloader.dataset)
            avg_contrast = total_contrast / len(trainloader.dataset)
            print(f"Epoch {epoch + 1}/{num_epochs}: "
                  f"Total Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, Contrast={avg_contrast:.4f}")

            # 保存模型
            if save_path and (epoch + 1) % 100 == 0:  # 每 100 个 epoch 保存一次
                self.save_model(f"{save_path}_epoch{epoch + 1}.pt")
                print(f"Model saved to {save_path}_epoch{epoch + 1}.pt")

    def save_model(self, path):
        """保存模型参数"""
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        """加载模型参数"""
        self.load_state_dict(torch.load(path, map_location=torch.device('gpu')))
        print(f"Model loaded from {path}")




