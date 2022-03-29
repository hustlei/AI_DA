# 非线性回归
"""y=2^x+x^3+e^x"""

"""
+ 模型定义
+ 数据加载
+ 训练模型
+ 测试
+ 可视化
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn import metrics


## 模型定义
class NonLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Parameter(torch.randn(1))
        self.a2 = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.a1 ** x + x ** self.a2 + torch.exp(x)
        return x


model = NonLinear()
opt = torch.optim.SGD(params=model.parameters(), lr=0.0000001)
mseloss = nn.MSELoss()

## 数据加载
num = 100  # 数据规模
x = torch.rand(num, 1) * 10 + 0.001
y = 2 ** x + x ** 3 + torch.exp(x)
y += torch.randn(num, 1)

## 训练
for i in range(200):
    out = model(x)
    loss = mseloss(out, y)
    loss.backward()
    opt.step()
    opt.zero_grad()

w = [model.a1.item(), model.a2.item()]

## 测试
xtest = torch.arange(0.1, 10.1, 0.25)
with torch.no_grad():
    ytest = model(xtest)
    r2 = metrics.r2_score(y, out)

txt = "y="
txt += "{:+.3f}^x+x^{:.3f}+e^x".format(w[0], w[1])
txt += "\nR^2={:.3f}".format(r2)

plt.text(0, 100, txt)
plt.scatter(x, y)
plt.plot(xtest, ytest, color='r')
plt.show()
