import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import normalized_mutual_info_score  # NMI
from sklearn.metrics import rand_score  # RI
from sklearn.metrics import accuracy_score  # ACC
from sklearn.metrics import f1_score  # F-measure
from sklearn.metrics import adjusted_rand_score  # ARI

from sklearn.cluster import KMeans


# 生成 Spiral 数据集
df= np.loadtxt('./dataset/Sprial.txt')


# 绘制散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(df[0], df[1], s=50, alpha=0.8)
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.title('Scatter Plot')
# plt.show()

kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(df)
centers = kmeans.cluster_centers_

# 绘制数据点和聚类中心
plt.figure(figsize=(8, 6))
plt.scatter(df[:, 0], df[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.8)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('K-means Clustering with Centers')
plt.show()

