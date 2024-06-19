import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans


def density_peak_clustering(data, y, num_centers, dc):
    # 计算距离矩阵
    distances = pairwise_distances(data)

    n = distances.shape[0]

    # 步骤1：使用高斯核计算局部密度
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(np.exp(-(distances[i] / dc) ** 2))

    # 步骤2：计算 delta
    delta = np.zeros(n)
    nneigh = np.zeros(n, dtype=int)
    sorted_rho_idx = np.argsort(-rho)  # 按照密度降序排序
    delta[sorted_rho_idx[0]] = np.max(distances[sorted_rho_idx[0]])
    for i in range(1, n):
        delta[sorted_rho_idx[i]] = np.min(distances[sorted_rho_idx[i], sorted_rho_idx[:i]])
        nneigh[sorted_rho_idx[i]] = sorted_rho_idx[np.argmin(distances[sorted_rho_idx[i], sorted_rho_idx[:i]])]

    # delta大但是rho小的点定义为噪声 去除噪声点

    # 步骤3：选择聚类中心
    delta_rho = delta * rho
    cluster_centers_idx = np.argsort(-delta_rho)[:num_centers]
    cluster_centers = np.zeros(n, dtype=bool)
    cluster_centers[cluster_centers_idx] = True
    # 获取聚类中心的位置
    cluster_centers_pos = data[cluster_centers_idx]

    # 判定离群点，假设rho < 阈值 或 delta > 阈值为离群点
    rho_threshold = np.percentile(rho, 5)  # 局部密度阈值（取前5%的点作为离群点）
    delta_threshold = np.percentile(delta, 95)  # 相对密度阈值（取后5%的点作为离群点）
    outliers = (rho < rho_threshold) & (delta > delta_threshold)
    filtered_data = data[~outliers]
    filtered_labels = y[~outliers]

    # 步骤4：分配簇
    # clusters = -np.ones(n, dtype=int)
    # clusters[cluster_centers_idx] = np.arange(num_centers)
    # for i in sorted_rho_idx:
    #     if not cluster_centers[i]:
    #         clusters[i] = clusters[nneigh[i]]
    # 使用 KMeans 分配剩余点
    kmeans = KMeans(n_clusters=num_centers, init=cluster_centers_pos, n_init=1)
    kmeans.fit(data)
    clusters = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    clusters = clusters[~outliers]

    return filtered_data, filtered_labels, clusters, cluster_centers_idx, cluster_centers, rho, delta, outliers


# 示例使用
# 示例数据
'''
# X = np.loadtxt('../dataset/Sprial.txt')
X, y = make_blobs(n_samples=300, centers=5, random_state=42)
# 定义截断距离dc和聚类中心数量
dc = np.percentile(pairwise_distances(X), 2)  # 截断距离
num_centers = 4  # 聚类中心数量

# 执行密度峰值聚类
x, y, clusters, cluster_centers_idx, cluster_centers_pos, rho, delta = density_peak_clustering(X, y, num_centers, dc)

# 绘制决策图
plt.figure()
plt.scatter(rho, delta, s=20, edgecolor='k')
plt.scatter(rho[cluster_centers_idx], delta[cluster_centers_idx], color='red', s=50, edgecolor='k')
plt.xlabel('rho')
plt.ylabel('Delta')
plt.title('DPC-Decision Graph')
plt.show()

# 绘制聚类结果图
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', s=20, edgecolor='k')
plt.scatter(cluster_centers_pos[:, 0], cluster_centers_pos[:, 1], color='red', s=50, edgecolor='k')
plt.title('DPC-result')
plt.legend()
plt.show()
'''
