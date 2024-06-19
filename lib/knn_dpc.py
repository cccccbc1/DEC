import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def knn_dpc(data, y, k, num_centers):
    # 计算距离矩阵
    distances = pairwise_distances(data)
    n = distances.shape[0]

    # 计算K近邻距离矩阵
    sorted_distances = np.sort(distances, axis=1)
    knn_distances = sorted_distances[:, 1:k + 1]

    # 使用高斯核计算局部密度
    dc = np.median(knn_distances)
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(np.exp(-(distances[i] / dc) ** 2))

    # 计算相对密度（delta）
    delta = np.zeros(n)
    for i in range(n):
        higher_density_points = np.where(rho > rho[i])[0]
        if higher_density_points.size == 0:
            delta[i] = np.max(distances[i])
        else:
            delta[i] = np.min(distances[i, higher_density_points])

    # 判定离群点，假设rho < 阈值 或 delta > 阈值为离群点
    rho_threshold = np.percentile(rho, 5)  # 局部密度阈值（取前5%的点作为离群点）
    delta_threshold = np.percentile(delta, 95)  # 相对密度阈值（取后5%的点作为离群点）
    outliers = (rho < rho_threshold) & (delta > delta_threshold)

    data, y = remove_outliers(data, y, outliers)

    # 去除后重新聚类
    # 计算距离矩阵
    distances = pairwise_distances(data)
    n = distances.shape[0]
    # 计算K近邻距离矩阵
    sorted_distances = np.sort(distances, axis=1)
    knn_distances = sorted_distances[:, 1:k + 1]
    dc = np.median(knn_distances)
    rho = np.zeros(n)
    for i in range(n):
        rho[i] = np.sum(np.exp(-(distances[i] / dc) ** 2))

    # 计算相对密度（delta）
    delta = np.zeros(n)
    for i in range(n):
        higher_density_points = np.where(rho > rho[i])[0]
        if higher_density_points.size == 0:
            delta[i] = np.max(distances[i])
        else:
            delta[i] = np.min(distances[i, higher_density_points])

    # 选择聚类中心，根据 rho * delta 的值排序
    cluster_centers_idx = np.argsort(-rho * delta)[:num_centers]
    cluster_centers = np.zeros(n, dtype=bool)
    cluster_centers[cluster_centers_idx] = True

    # 使用KMeans分配非中心点到聚类中心
    kmeans = KMeans(n_clusters=num_centers, init=data[cluster_centers_idx], n_init=10)
    y_pred = kmeans.fit_predict(data)

    # 返回结果
    cluster_centers_pos = data[cluster_centers_idx]
    return data, y, y_pred, cluster_centers_idx, cluster_centers_pos, rho, delta, outliers


def remove_outliers(data, labels, outliers):
    # 去除离群点
    filtered_data = data[~outliers]
    filtered_labels = labels[~outliers]
    return filtered_data, filtered_labels

'''
# 示例数据
X, y = make_blobs(n_samples=300, centers=5, random_state=42)

# 运行KNN-DPC算法
k = 10
num_centers = 5
X, y, y_pred, cluster_centers_idx, cluster_centers_pos, rho, delta, outliers = knn_dpc(X, y, k, num_centers)

# 输出结果
print("聚类标签：", y_pred)
print("聚类中心索引：", cluster_centers_idx)
print("聚类中心位置：", cluster_centers_pos)
print("局部密度：", rho)
print("相对密度：", delta)
print("离群点索引：", np.where(outliers)[0])

# 可视化结果
plt.figure(figsize=(12, 6))

# 绘制数据点和聚类结果
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', marker='o')
plt.scatter(cluster_centers_pos[:, 0], cluster_centers_pos[:, 1], color='red', marker='x')
# plt.scatter(X[outliers, 0], X[outliers, 1], color='black', marker='o', edgecolors='w', s=100, label='Outliers')
plt.title('KNN-DPC Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.legend()

# 绘制决策图（rho-delta图）
plt.subplot(1, 2, 2)
plt.scatter(rho, delta)
plt.scatter(rho[cluster_centers_idx], delta[cluster_centers_idx], color='red')
# plt.scatter(rho[outliers], delta[outliers], color='black', edgecolors='w', s=100, label='Outliers')
plt.title('Decision Graph (Rho-Delta)')
plt.xlabel('Rho (Local Density)')
plt.ylabel('Delta (Distance to Higher Density)')
# plt.legend()

plt.tight_layout()
plt.show()
'''