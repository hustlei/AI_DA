# 逻辑回归模型(监督分类算法）
"""逻辑回归模型是监督分类算法族的成员之一。
通过使用逻辑函数估计概率来测量因变量和自变量之间的关系。"""
import numpy as np
from sklearn import linear_model, datasets
from sklearn import metrics
import matplotlib.pyplot as plt

## 加载数据
datas = datasets.load_iris()  # 鸢尾花数据，150组，每组4个数值
data = datas.data[:, :2]  # 只分析前两列
dataname = datas.feature_names
target = datas.target
targetname = datas.target_names

## 建立模型
model = linear_model.LogisticRegression(solver = 'liblinear', C = 75)

## 训练
model.fit(data, target)

## 测试
### 准备数据
x_plot, y_plot = np.meshgrid(np.linspace(4, 8, 100), np.linspace(2, 5, 100))
xy_plot = np.c_[x_plot.ravel(), y_plot.ravel()]

### 测试
test_pred = model.predict(xy_plot)
test_pred = test_pred.reshape(x_plot.shape)

target_pred = model.predict(data)
## 可视化
success_rate = metrics.accuracy_score(target, target_pred)
print("{:.2%}".format(success_rate))

plt.contourf(x_plot, y_plot, test_pred, alpha=0.3, cmap=plt.cm.tab10)
#plt.pcolormesh(x_plot, y_plot, test_pred, cmap=plt.cm.Set2) 
plt.scatter(data[:, 0], data[:, 1], c=target, cmap=plt.cm.Set1)
plt.show()
