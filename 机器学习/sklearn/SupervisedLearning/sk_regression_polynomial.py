# 多项式(sklearn)
# [1, a, b, a^2, ab, b^2]
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from matplotlib import pyplot as plt

## 输入数据处理
x1 = np.random.rand(10, 1) * 10
x2 = np.random.rand(10, 1) * 10
x = np.concatenate((x1, x2), axis=-1)
y = (0.1 * x1 + 0.2 * x2) ** 2 + np.random.randn(10, 1)

## 创建多项式
polynomial = PolynomialFeatures(degree=2)
x_transfromed = polynomial.fit_transform(x)

## 定义线性模型
reg_multi_linear = linear_model.LinearRegression()

## 训练
reg_multi_linear.fit(x_transfromed, y)

weight = reg_multi_linear.coef_.reshape(-1)
bias = reg_multi_linear.intercept_

## 测试
y_predict = reg_multi_linear.predict(x_transfromed)

mae = metrics.mean_absolute_error(y, y_predict)
mse = metrics.mean_squared_error(y, y_predict)
r2score = metrics.r2_score(y, y_predict)
print("err:", mae, mse, r2score)
print("weight:", weight)
print("bias:", bias)

## 可视化
fig = plt.figure()
ax = fig.add_subplot(211, projection="3d")
ax.scatter(x1, x2, y)

xgrid, ygrid = np.meshgrid(np.arange(10), np.arange(10))
zgrid = weight[0] * 1 + weight[1] * xgrid + weight[2] * ygrid + weight[3] * xgrid ** 2 + weight[4] * xgrid * ygrid + \
        weight[5] * ygrid ** 2 + bias
ax.plot_wireframe(xgrid, ygrid, zgrid, color="r")

err = y_predict - y
ax1 = fig.add_subplot(212)
ax1.plot(err)

plt.show()
