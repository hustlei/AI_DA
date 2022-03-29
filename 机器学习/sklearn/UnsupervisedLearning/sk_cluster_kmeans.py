# 聚类算法（无监督学习）

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

## 数据准备
data, target = make_blobs(n_samples=400, centers=4, cluster_std=[1.5,1,1,1])
"""
n_samples(int/序列):如果参数为int，代表总样本数；如果参数为array-like，数组中的每个数代表每一簇的样本数。默认值为100。
n_features(int):样本点的维度。默认为2。
centers(int):样本中心数。如果样本数为int且centers=None，生成三个样本中心；如果样本数（n_samples）为数组，则centers要么为None，要么为数组的长度。
cluster_std(float/float序列):样本中，簇的标准差，默认为1。可以设置为[1.0,3.0]，为不同的簇设置不同的值。
center_box(float元组(min, max)):每个簇的上下限。默认为(-10,10)
"""


## 建立模型
model = KMeans(n_clusters=4)

## 训练模型
model.fit(data)
centers = model.cluster_centers_

## 测试
out = model.predict(data)


## 可视化

#score = model.score(data, target)#不知道是什么分数#accuracy_score不适用，因为聚类的label顺序不同
from sklearn import metrics
score = metrics.silhouette_score(data, target)#样本平均轮廓系数，是聚类效果好坏的一种评价方式。最佳值为1，最差值为-1。接近0的值表示重叠的群集。负值通常表示样本已分配给错误的聚类，因为不同的聚类更为相​​似
print("input silhouette:{:.3f}".format(score))

score = metrics.silhouette_score(data, out) #out或者model.labels_
print("out silhouette:{:.3f}".format(score))

plt.scatter(data[:,0], data[:,1], c=out)
plt.scatter(centers[:,0], centers[:,1], c='red', s=100)
plt.show()