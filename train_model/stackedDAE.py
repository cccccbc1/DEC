import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.cluster import KMeans
from dec_pytorch.lib import metrics
import numpy as np
import math
from dec_pytorch.lib.utils import Dataset, masking_noise
from dec_pytorch.lib.ops import MSELoss, BCELoss
from dec_pytorch.lib.denoisingAutoencoder import DenoisingAutoencoder

def buildNetwork(layers, activation="relu", dropout=0):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if activation=="relu":
            net.append(nn.ReLU())
        elif activation=="sigmoid":
            net.append(nn.Sigmoid())
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch//100))
    toprint = True
    for param_group in optimizer.param_groups:
        if param_group["lr"]!=lr:
            param_group["lr"] = lr
            if toprint:
                print("Switching to learning rate %f" % lr)
                toprint = False

class StackedDAE(nn.Module):
    def __init__(self, input_dim=784, z_dim=10, binary=True,
        encodeLayer=[400], decodeLayer=[400], activation="relu", 
        dropout=0, tied=False):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self.decoder = buildNetwork([z_dim] + decodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        
        self._dec = nn.Linear(decodeLayer[-1], input_dim)
        self._dec_act = None
        if binary:
            self._dec_act = nn.Sigmoid()

    def decode(self, z):
        h = self.decoder(z)
        x = self._dec(h)
        if self._dec_act is not None:
            x = self._dec_act(x)
        return x

    def loss_function(self, recon_x, x):
        loss = -torch.mean(torch.sum(x*torch.log(torch.clamp(recon_x, min=1e-10))+
            (1-x)*torch.log(torch.clamp(1-recon_x, min=1e-10)), 1))

        return loss

    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)

        return z, self.decode(z)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)

    def pretrain(self, trainloader, validloader, lr=0.001, batch_size=128, num_epochs=10, corrupt=0.2, loss_type="cross-entropy"):
        trloader = trainloader
        valoader = validloader
        daeLayers = []
        for l in range(1, len(self.layers)):
            infeatures = self.layers[l-1]
            outfeatures = self.layers[l]
            if l!= len(self.layers)-1:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation=self.activation, dropout=corrupt)
            else:
                dae = DenoisingAutoencoder(infeatures, outfeatures, activation="none", dropout=0)
            print(dae)
            if l==1:
                dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type=loss_type)
            else:
                if self.activation=="sigmoid":
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type="cross-entropy")
                else:
                    dae.fit(trloader, valoader, lr=lr, batch_size=batch_size, num_epochs=num_epochs, corrupt=corrupt, loss_type="mse")
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
        if self.dropout==0:
            every = 2
        else:
            every = 3
        # input layer
        # copy encoder weight
        self.encoder[0].weight.data.copy_(daeLayers[0].weight.data)
        self.encoder[0].bias.data.copy_(daeLayers[0].bias.data)
        self._dec.weight.data.copy_(daeLayers[0].deweight.data)
        self._dec.bias.data.copy_(daeLayers[0].vbias.data)

        for l in range(1, len(self.layers)-2):
            # copy encoder weight
            self.encoder[l*every].weight.data.copy_(daeLayers[l].weight.data)
            self.encoder[l*every].bias.data.copy_(daeLayers[l].bias.data)

            # copy decoder weight
            self.decoder[-(l-1)*every-2].weight.data.copy_(daeLayers[l].deweight.data)
            self.decoder[-(l-1)*every-2].bias.data.copy_(daeLayers[l].vbias.data)

        # z layer
        self._enc_mu.weight.data.copy_(daeLayers[-1].weight.data)
        self._enc_mu.bias.data.copy_(daeLayers[-1].bias.data)
        self.decoder[0].weight.data.copy_(daeLayers[-1].deweight.data)
        self.decoder[0].bias.data.copy_(daeLayers[-1].vbias.data)

    def fit(self, trainloader, validloader, lr=0.001, num_epochs=10, corrupt=0.3,
        loss_type="mse"):
        """
        data_x: FloatTensor
        valid_x: FloatTensor
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Stacked Denoising Autoencoding Layer=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        if loss_type=="mse":
            criterion = MSELoss()
        elif loss_type=="cross-entropy":
            criterion = BCELoss()

        # validate
        total_loss = 0.0
        total_num = 0
        for batch_idx, (inputs, _) in enumerate(validloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            if use_cuda:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs = self.forward(inputs)

            valid_recon_loss = criterion(outputs, inputs)
            total_loss += valid_recon_loss.data * len(inputs)
            total_num += inputs.size()[0]

        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))
        self.train()
        for epoch in range(num_epochs):
            # train 1 epoch
            adjust_learning_rate(lr, optimizer, epoch)
            train_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()
                    inputs_corr = inputs_corr.cuda()
                optimizer.zero_grad()
                inputs = Variable(inputs)
                inputs_corr = Variable(inputs_corr)

                z, outputs = self.forward(inputs_corr)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.data*len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            valid_loss = 0.0
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.cuda()
                inputs = Variable(inputs)
                z, outputs = self.forward(inputs)

                valid_recon_loss = criterion(outputs, inputs)
                valid_loss += valid_recon_loss.data * len(inputs)

            print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f" % (
                epoch+1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

        ACC_all=[]
        NMI_all=[]
        LOSS_all = []

        # 在子空间做聚类
        z_all = []
        label_true = []
        for batch_idx, (inputs, target) in enumerate(trainloader):
            inputs = inputs.view(inputs.size(0), -1).float()
            inputs = inputs.cuda()
            inputs = Variable(inputs)
            z, outputs = self.forward(inputs)
            z1 = z.data.cpu().numpy()  # 直接转换为NumPy数组
            z_all.append(z1)
            label_true.append(target.cpu().numpy())  # 保持为列表，稍后处理

        # 将列表转换为大数组之前，先展平或调整形状
        z_all = np.vstack(z_all)  # 假设z的形状适合垂直堆叠
        label_true = np.concatenate(label_true)  # 连接所有批次的标签

        # 确保z_all和label_true的形状符合预期
        # print("Shape of z_all:", z_all.shape)
        # print("Shape of label_true:", label_true.shape)

        ##kmeans result
        estimator = KMeans(10)
        estimator.fit(z_all)
        centroids = estimator.cluster_centers_
        label_pred = estimator.labels_
        acc = metrics.acc(label_true, label_pred)
        nmi = metrics.nmi(label_true, label_pred)
        print("acc:", acc)
        print("nmi", nmi)
        ACC_all.append(acc)
        NMI_all.append(nmi)
        if acc > 0.85:
            sdae_savepath = ("./model/csdae-%d.pt" % epoch)
            self.save_model(sdae_savepath)

        # 打印训练信息
        avg_loss = total_loss / len(trainloader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Total Loss={avg_loss:.4f}")

