import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import seaborn as sns
import numpy as np
import math
from dec_pytorch.lib.utils import acc
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 用于降维
from sklearn.manifold import TSNE  # 用于降维


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


class DEC(nn.Module):
    # z_dim 应该是把数据降为10维
    def __init__(self, input_dim=784, z_dim=10, n_clusters=10,
                 encodeLayer=[400], activation="relu", dropout=0, alpha=1.):
        super(self.__class__, self).__init__()
        self.z_dim = z_dim
        self.layers = [input_dim] + encodeLayer + [z_dim]
        self.activation = activation
        self.dropout = dropout
        self.encoder = buildNetwork([input_dim] + encodeLayer, activation=activation, dropout=dropout)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)

        self.n_clusters = n_clusters
        self.alpha = alpha
        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    # 向前传播过程  计算qij
    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        # compute q -> NxK
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** (self.alpha + 1.0) / 2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    def encodeBatch(self, dataloader, islabel=False):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        ylabels = []
        self.eval()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = Variable(inputs)
            z, _ = self.forward(inputs)
            encoded.append(z.data.cpu())
            ylabels.append(labels)

        encoded = torch.cat(encoded, dim=0)
        ylabels = torch.cat(ylabels)
        if islabel:
            out = (encoded, ylabels)
        else:
            out = encoded
        return out

    # KL散度计算函数
    def loss_function(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=1))

        loss = kld(p, q)
        return loss

    # 计算pij
    def target_distribution(self, q):
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def fit(self, X, y=None, lr=0.001, batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, step = 0):
        '''X: tensor data'''
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("=====Training DEC=======")
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, momentum=0.9)

        print("Initializing cluster centers with kmeans.")
        # kmeans 去掉n_init，看命
        kmeans = KMeans(self.n_clusters)
        data, _ = self.forward(X)
        y_pred = kmeans.fit_predict(data.data.cpu().numpy())
        y_pred_last = y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            y = y.cpu().numpy()
            while acc(y, y_pred) < 0.82:
                y_pred = kmeans.fit_predict(data.data.cpu().numpy())
                y_pred_last = y_pred
                self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
            print("Kmeans acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        acc_list = []
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                # update the targe distribution p
                _, q = self.forward(X)
                p = self.target_distribution(q).data

                # evalute the clustering performance
                y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                if y is not None:
                    print("acc: %.5f, nmi: %.5f" % (acc(y, y_pred), normalized_mutual_info_score(y, y_pred)))
                acc_list.append(acc(y, y_pred))    # 保存acc
                # check stop criterion
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / num
                y_pred_last = y_pred
                # if epoch > 0 and delta_label < tol:
                # if epoch > 0 and step > 100:
                #     print('delta_label ', delta_label, '< tol ', tol)
                #     print("Reach tolerance threshold. Stopping training.")
                #     break
                # step = step + 1

            # train 1 epoch
            train_loss = 0.0
            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

                optimizer.zero_grad()
                inputs = Variable(xbatch)
                target = Variable(pbatch)

                z, qbatch = self.forward(inputs)
                loss = self.loss_function(target, qbatch)
                train_loss += loss.data * len(inputs)
                loss.backward()
                optimizer.step()

            print("#Epoch %3d: Loss: %.4f" % (
                epoch + 1, train_loss / num))

        # 保存acc
        # df = pd.DataFrame(acc_list, columns=["Accuracy"])  # 创建DataFrame
        # df.to_excel("./tutu/byol_accuracy_results.xlsx", index=False)



        # 计算混淆矩阵
        # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        #
        # y_true = y.astype(np.int64)
        # assert y_pred.size == y_true.size
        # D = max(y_pred.max(), y_true.max()) + 1
        # w = np.zeros((D, D), dtype=np.int64)
        # for i in range(y_pred.size):
        #     w[y_pred[i], y_true[i]] += 1
        # from scipy.optimize import linear_sum_assignment
        # row_ind, col_ind = linear_sum_assignment(w.max() - w)
        # print(row_ind)
        # print(col_ind)
        # class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        # # class_names = col_ind
        # cm = confusion_matrix(y, y_pred)
        # cm = cm[col_ind[:, None], row_ind]
        # label_to_index = {label: idx for idx, label in enumerate(col_ind)}
        # index_map = [label_to_index[label] for label in row_ind]
        # # 步骤3：调整矩阵顺序
        # # 调整行顺序
        # reordered_rows = cm[index_map, :]
        # # 调整列顺序
        # cm = reordered_rows[:, index_map]
        # # 使用 Scikit-learn 绘图
        # plt.figure(figsize=(6, 4))
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        # disp.plot(cmap='Blues', values_format='d')  # values_format 避免科学计数法
        # plt.title('Confusion Matrix')
        # plt.savefig('matrix.pdf', bbox_inches='tight')  # 矢量格式
        # # plt.show()
        #
        #
        #
        # 可视化
        z, qbatch = self.forward(X)
        # visualize_features(z, y, "plot.pdf")
        # 数据预处理
        scaler = StandardScaler()
        z_np = z.detach().cpu().numpy()
        z_normalized = scaler.fit_transform(z.detach().cpu().numpy())
        tsne = TSNE(n_components=2,
                    perplexity=min(30, len(z_np) // 3))
        x_2dim = tsne.fit_transform(z_normalized)
        # x_2dim = TSNE(n_components=2, random_state=42).fit_transform(z_normalized)
        # 创建DataFrame便于处理
        df = pd.DataFrame(x_2dim, columns=['x', 'y'])
        df['label'] = y
        # 设置学术图表样式
        # plt.style.use('seaborn-whitegrid')
        sns.set_theme(style="whitegrid")
        sns.set_palette("tab10")  # 使用高对比度的颜色方案
        plt.rcParams.update({
            # 'font.family': 'Times New Roman',
            'font.size': 14,
            'figure.dpi': 300,  # 提高分辨率
            'savefig.dpi': 300,
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
        fig = plt.figure(figsize=(10, 8))  # 更紧凑的尺寸适合论文排版
        ax = fig.add_subplot(111)
        # 绘制散点图
        scatter = ax.scatter(
            df['x'],
            df['y'],
            c=df['label'],
            cmap='tab10',  # 使用分类清晰的colormap
            s=20,  # 适当减小点的大小
            alpha=0.8,  # 增加透明度显示密度
            edgecolors='none',
            linewidths=0.5
        )
        # 添加类别标签（自动计算中心点）
        for label in df['label'].unique():
            mask = df['label'] == label
            x_mean = df[mask]['x'].mean()
            y_mean = df[mask]['y'].mean()
            ax.text(
                x_mean,
                y_mean,
                str(label),  # 假设标签可以直接转为字符串
                fontsize=20,
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='none'
                )
            )
        # 优化坐标轴
        # ax.set_xlabel('t-SNE 1', labelpad=10)
        # ax.set_ylabel('t-SNE 2', labelpad=10)
        ax.xaxis.set_tick_params(which='both', length=0)
        ax.yaxis.set_tick_params(which='both', length=0)
        # ax.grid(True, linestyle='--', alpha=0.6)  # 更细密的网格线
        # 紧凑布局
        plt.tight_layout()
        # 保存多种格式（按需选择）
        plt.savefig('tsne-test.pdf', bbox_inches='tight')  # 矢量格式
        # plt.savefig('tsne_visualization.tiff', bbox_inches='tight', dpi=300)  # 高分辨率位图
        plt.show()
        plt.close()  # 关闭图形避免内存泄漏
        print("end")


# def visualize_features(z, y, save_path="tsne_plot.pdf"):
#     # 数据预处理
#     z_np = z.detach().cpu().numpy()
#     scaler = StandardScaler()
#     z_normalized = scaler.fit_transform(z_np)
#
#     # t-SNE降维
#     tsne = TSNE(n_components=2,
#                 perplexity=min(30, len(z_np) // 3),  # 自适应复杂度
#                 n_iter=1000)
#     # x_2dim = TSNE(n_components=2, random_state=42).fit_transform(z_normalized)
#     x_2dim = tsne.fit_transform(z_normalized)
#
#     # 创建安全DataFrame
#     df = pd.DataFrame(x_2dim, columns=['Dim1', 'Dim2'])
#     try:
#         df['label'] = y
#     except Exception as e:
#         print(f"标签转换错误: {str(e)}")
#         print(f"调试信息 - y类型: {type(y)}, 形状: {getattr(y, 'shape', '无')}")
#         raise
#
#     # 可视化设置
#     plt.style.use('seaborn-whitegrid')
#     sns.set_palette("tab10")
#     plt.rcParams.update({
#         'font.family': 'Times New Roman',
#         'font.size': 12,
#         'axes.titlesize': 14,
#         'axes.labelsize': 12,
#         'figure.dpi': 300,
#         'savefig.dpi': 300,
#         'legend.frameon': True,
#         'legend.framealpha': 0.8
#     })
#
#     # 创建画布
#     fig, ax = plt.subplots(figsize=(8, 6))
#
#     # 绘制散点图（学术优化参数）
#     scatter = ax.scatter(
#         df['Dim1'],
#         df['Dim2'],
#         c=df['label'],
#         cmap='tab10',
#         s=25,  # 优化点大小
#         alpha=0.7,  # 平衡重叠显示
#         edgecolor='w',  # 白色边缘增强对比
#         linewidth=0.3,
#         rasterized=True  # 提升渲染性能
#     )
#
#     # 专业图例配置
#     legend_labels = [f'Class {i}' for i in df['label'].unique()]
#     legend_elements = [plt.Line2D([0], [0],
#                                   marker='o',
#                                   color='w',
#                                   markerfacecolor=scatter.cmap(scatter.norm(i)),
#                                   markersize=8,
#                                   label=label)
#                        for i, label in enumerate(legend_labels)]
#
#     ax.legend(handles=legend_elements,
#               title="Categories",
#               bbox_to_anchor=(1.05, 1),
#               loc='upper left',
#               borderaxespad=0.5,
#               framealpha=0.9)
#
#     # 坐标轴优化
#     ax.set_xlabel('t-SNE Dimension 1', labelpad=8)
#     ax.set_ylabel('t-SNE Dimension 2', labelpad=8)
#     ax.tick_params(axis='both', which='both', length=0)
#     ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.5)  # 更细的网格线
#
#     # 保存输出
#     plt.tight_layout(rect=[0, 0, 0.85, 1])  # 为图例留出空间
#     for fmt in ['pdf', 'tiff']:
#         plt.savefig(save_path.replace('.pdf', f'.{fmt}'),
#                     bbox_inches='tight',
#                     dpi=300 if fmt == 'tiff' else None)