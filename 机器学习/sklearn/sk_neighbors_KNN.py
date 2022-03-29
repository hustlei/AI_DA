# KNN分类器
"""K-Nearest Neighbors（KNN）分类器是一种分类模型，它使用最近邻算法对给定数据点进行分类。
它在大量分类和回归问题中非常成功，例如字符识别或图像分析。"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets, metrics
from matplotlib import pyplot as plt
import numpy as np

## 加载数据

digits = datasets.load_digits() #1797组数据，每个数据包含64个数（8x8）点阵
data = digits.data #1797x64
images = digits.images #1797x8x8   data[0].reshape(8,8)==images[0]
target = digits.target #0~9

"""显示数据"""
#plt.imshow(images[100])
#plt.show()


## 构建模型
model = KNeighborsClassifier(10)

## 训练
model.fit(data, target)

## 测试
target_test = model.predict(data[1700].reshape(1,-1))
print("target:{}\npredict:{}".format(target[1700], target_test))
plt.imshow(images[1700])
plt.show()
