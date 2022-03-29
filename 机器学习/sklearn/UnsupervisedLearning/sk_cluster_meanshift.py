# 聚类算法：均值漂移算法/层次聚类分析/均值移位聚类分析（无监督学习）
"""
均值漂移算法没有做出任何假设，是一个非参数算法。
也称为层次聚类或均值移位聚类分析。

不要求预先知道聚类的类别个数，对聚类的形状也没有限制。

基本步骤：

+ 首先，我们需要从分配给自己的集群的数据点开始。
+ 现在，它计算质心并更新新质心的位置。
+ 通过重复这个过程，我们移近群集的峰值，即朝向更高密度的区域。
+ 该算法在质心不再移动的阶段停止。
"""

from sklearn.cluster import MeanShift
from sklearn import datasets, metrics
from matplotlib import pyplot as plt

## 加载数据
data, target = datasets.make_blobs(n_samples=400, centers=4)

## 构建模型
model = MeanShift()

## 训练模型
model.fit(data)
centers = model.cluster_centers_

## 测试数据
target_pred = model.labels_ #target_pred = model.predict(data)

## 可视化
score = metrics.silhouette_score(data, target)
test_score = metrics.silhouette_score(data, target_pred)
print("data silhouette:{:.2f}\npredict silhouette:{:.2f}".format(score, test_score))

plt.scatter(data[:,0], data[:,1], c=target_pred)
plt.scatter(centers[:,0], centers[:,1], c='r', s=100)
from matplotlib import style
style.use("ggplot")
plt.show()