# 最近邻居算法
"""从给定数据集中找到输入点的最近点,
实际上就是计算每个点与目标点的距离，找出距离最近的点"""
"""
KNN是一种非参的，惰性的算法模型。

非参的意思并不是说这个算法不需要参数，而是意味着这个模型不会对数据做出任何的假设，
与之相对的是线性回归（我们总会假设线性回归是一条直线）。
也就是说KNN建立的模型结构是根据数据来决定的，这也比较符合现实的情况，
毕竟在现实中的情况往往与理论上的假设是不相符的。

惰性又是什么意思呢？想想看，同样是分类算法，逻辑回归需要先对数据进行大量训练（tranning），
最后才会得到一个算法模型。而KNN算法却不需要，它没有明确的训练数据的过程，或者说这个过程很快。
"""

from sklearn.neighbors import NearestNeighbors
from sklearn import datasets, metrics
from matplotlib import pyplot as plt
import numpy as np

## 加载数据

points = np.array([[3.1, 2.3], [2.3, 4.2], [3.9, 3.5], [3.7, 6.4], [4.8, 1.9],
             [8.3, 3.1], [5.2, 7.5], [4.8, 4.7], [3.5, 5.1], [4.4, 2.9],])

x = [3.3, 2.9]
k = 4 #搜索最近点个数

## 构建模型
model = NearestNeighbors(n_neighbors=k)

## 计算
model.fit(points)

## 测试
distances, indices = model.kneighbors([x]) #结果均为列向量

## 可视化
print("K Nearest Neighbors to x:")
for i in range(k):
    print("point {}:{}".format(i+1, points[indices[0,i]]))

#fig, ax = plt.subplots()
#ax = plt.figure().add_subplot()
plt.scatter(points[:,0],points[:,1]) #所有点
plt.scatter(x[0], x[1], marker="x") #x点
npts = points[indices[0]]
plt.scatter(npts[:,0], npts[:,1]) #计算得到的x临近的点

from matplotlib.patches import Circle
circle = Circle(x, radius=max(distances[0]), facecolor="none", edgecolor="k")
ax = plt.gca()
ax.set_aspect(1)
ax.add_patch(circle)

plt.show()