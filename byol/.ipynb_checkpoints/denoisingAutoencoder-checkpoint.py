import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import math
from dec_pytorch.lib.utils import Dataset, masking_noise
from dec_pytorch.lib.ops import MSELoss, BCELoss


torch.backends.cudnn.benchmark = True

class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_features, out_features, activation="relu",
                 dropout=0.2, tied=False):
        super(self.__class__, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if tied:
            self.deweight = self.weight.t()
        else:
            self.deweight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        self.vbias = Parameter(torch.Tensor(in_features))

        if activation == "relu":
            self.enc_act_func = nn.ReLU()
        elif activation == "sigmoid":
            self.enc_act_func = nn.Sigmoid()
        elif activation == "none":
            self.enc_act_func = None
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.01
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        stdv = 0.01
        self.deweight.data.uniform_(-stdv, stdv)
        self.vbias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        if self.enc_act_func is not None:
            return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
        else:
            return self.dropout(F.linear(x, self.weight, self.bias))

    def encode(self, x, train=True):
        if train:
            self.dropout.train()
        else:
            self.dropout.eval()
        if self.enc_act_func is not None:
            return self.dropout(self.enc_act_func(F.linear(x, self.weight, self.bias)))
        else:
            return self.dropout(F.linear(x, self.weight, self.bias))

    def encodeBatch(self, dataloader):
        encoded = []
        self.eval()  # 确保模型在评估模式
        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, (inputs, _) in enumerate(dataloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                # if torch.cuda.is_available():
                inputs = inputs.to('cuda', non_blocking=True)
                hidden = self.encode(inputs, train=False)
                encoded.append(hidden)  # 将结果移回CPU以进行后续处理

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def decode(self, x, binary=False):
        if not binary:
            return F.linear(x, self.deweight, self.vbias)
        else:
            return torch.sigmoid(F.linear(x, self.deweight, self.vbias))

    def fit(self, trainloader, validloader, lr=0.001, batch_size=256, num_epochs=10, corrupt=0.3,
            loss_type="mse"):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()  # 将模型移动到GPU
        print("=====Denoising Autoencoding layer=======")
        scaler = torch.cuda.amp.GradScaler()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)
        if loss_type == "mse":
            criterion = MSELoss()
        elif loss_type == "cross-entropy":
            criterion = BCELoss()

        # validate
        total_loss = 0.0
        total_num = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(validloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                if use_cuda:
                    inputs = inputs.to('cuda', non_blocking=True)  # ==== 使用non_blocking ====
                hidden = self.encode(inputs)
                if loss_type == "cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)

                valid_recon_loss = criterion(outputs, inputs)
                total_loss += valid_recon_loss.item() * len(inputs)
                total_num += inputs.size()[0]

        valid_loss = total_loss / total_num
        print("#Epoch 0: Valid Reconstruct Loss: %.4f" % (valid_loss))

        self.train()  # 确保模型在训练模式
        for epoch in range(num_epochs):
            # train 1 epoch
            train_loss = 0.0
            adjust_learning_rate(lr, optimizer, epoch)
            for batch_idx, (inputs, _) in enumerate(trainloader):
                inputs = inputs.view(inputs.size(0), -1).float()
                inputs_corr = masking_noise(inputs, corrupt)
                if use_cuda:
                    inputs = inputs.cuda()  # 将数据移动到GPU
                    inputs_corr = inputs_corr.cuda()  # 将数据移动到GPU
                optimizer.zero_grad()
                hidden = self.encode(inputs_corr)
                if loss_type == "cross-entropy":
                    outputs = self.decode(hidden, binary=True)
                else:
                    outputs = self.decode(hidden)
                recon_loss = criterion(outputs, inputs)
                train_loss += recon_loss.item() * len(inputs)
                recon_loss.backward()
                optimizer.step()

            # validate
            valid_loss = 0.0
            with torch.no_grad():  # 禁用梯度计算
                for batch_idx, (inputs, _) in enumerate(validloader):
                    inputs = inputs.view(inputs.size(0), -1).float()
                    if use_cuda:
                        inputs = inputs.cuda()  # 将数据移动到GPU
                    hidden = self.encode(inputs, train=False)
                    if loss_type == "cross-entropy":
                        outputs = self.decode(hidden, binary=True)
                    else:
                        outputs = self.decode(hidden)

                    valid_recon_loss = criterion(outputs, inputs)
                    valid_loss += valid_recon_loss.item() * len(inputs)

            print("#Epoch %3d: Reconstruct Loss: %.4f, Valid Reconstruct Loss: %.4f" % (
                epoch + 1, train_loss / len(trainloader.dataset), valid_loss / len(validloader.dataset)))

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 100))
    toprint = True
    for param_group in optimizer.param_groups:
        if param_group["lr"] != lr:
            param_group["lr"] = lr
            if toprint:
                print("Switching to learning rate %f" % lr)
                toprint = False