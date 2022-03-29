# 多变量线性回归(sklearn)
"""y=a0+a1x1+a2x2+...+anxn+b"""
from sklearn import linear_model, metrics  # 线性模型，评价指标
import numpy as np
from matplotlib import pyplot as plt

## 输入数据处理
datapath = "linear3to1.data"
data = np.loadtxt(datapath, delimiter=",")
x = data[:, :-1]  # 列向量
y = data[:, -1]  # 行列向量都可以

num4train = int(0.6 * len(x))
x_train = x[:num4train]
y_train = y[:num4train]
x_test = x[num4train:]
y_test = y[num4train:]

## 模型定义
reg_linear = linear_model.LinearRegression()
#reg_linaer = linear_model.LinearRegression(normalize=True)
    #fit_intercept：选择是否需要计算截距，默认为True，如果中心化了的数据可以选择false
    #normalize:选择是否需要标准化（中心化），默认为false，参数fit_intercept为False才起作用

## 训练模型(线性拟合）
reg_linear.fit(x_train, y_train)

weight = reg_linear.coef_
bias = reg_linear.intercept_

## 测试
y_test_predict = reg_linear.predict(x_test)

###性能评估
mae = metrics.mean_absolute_error(y_test, y_test_predict)  # 平均绝对误差
mad = metrics.median_absolute_error(y_test, y_test_predict)# 中位数绝对偏差/绝对中位差
mse = metrics.mean_squared_error(y_test, y_test_predict) # 均方误差
ev_score = metrics.explained_variance_score(y_test, y_test_predict) # 解释方差得分（接近于1说明自变量解释因变量的方差变化效果好，值越小则说明效果越差）
r2_score = metrics.r2_score(y_test, y_test_predict) #解释回归模型的方差得分，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。

print("Performance of Linear regressor:")
print("Mean absolute error =", round(mae, 2))
print("Median absolute error =", round(mad, 2))
print("Mean squared error =", round(mse, 2))
print("Explain variance score =", round(ev_score, 2))
print("R2 score =", round(r2_score, 2))

## 可视化
fig = plt.figure()
ax = fig.add_subplot(321,projection="3d")
ax.scatter(x[:,0], x[:,1], y)
ax1 = fig.add_subplot(322,projection="3d")
ax1.scatter(x[:,1], x[:,2], y)

err = reg_linear.predict(x)-y
ax2 = fig.add_subplot(312)
ax2.plot(err)
err =err/y*100
ax3 = fig.add_subplot(313)
ax3.plot(err)


text="y="
for i in range(x.shape[-1]):
    text+="{:+.3f}x{}".format(weight[i],i+1)
text+="{:+.3f}".format(bias)
ax.text(1,5, 0,text)
plt.show()
