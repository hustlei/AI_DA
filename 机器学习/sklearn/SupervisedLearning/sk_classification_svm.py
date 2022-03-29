# 支持向量机
"""支持向量机（SVM）是一种监督机器学习算法，可用于回归和分类。
SVC支持向量分类器"""
import numpy as np
from sklearn import svm, datasets
from sklearn import metrics
import matplotlib.pyplot as plt

## 加载数据
datas = datasets.load_iris()  # 鸢尾花数据，150组，每组4个数值
data = datas.data[:, :2]  # 只分析前两列
dataname = datas.feature_names
target = datas.target
targetname = datas.target_names

## 建立模型
model = svm.SVC()

## 训练
model.fit(data, target)

## 测试
### 准备数据
x_plot, y_plot = np.meshgrid(np.linspace(4, 8, 100), np.linspace(2, 5, 100))
xy_plot = np.c_[x_plot.ravel(), y_plot.ravel()]

### 测试
target_pred = model.predict(xy_plot)
target_pred = target_pred.reshape(x_plot.shape)

## 可视化
success_rate = metrics.accuracy_score(target, model.predict(data))
print("{:.2%}".format(success_rate))

plt.contourf(x_plot, y_plot, target_pred, alpha=0.3, cmap=plt.cm.tab10)
plt.scatter(data[:, 0], data[:, 1], c=target)#, cmap=plt.cm.Set1)
plt.show()
