# 多元线性回归
"""y=a1x1+a2x2+...+anxn+b"""

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

## 模型定义
model = nn.Linear(2,1)
opt = torch.optim.SGD(params=model.parameters(),lr=0.01)
mseloss = nn.MSELoss()

## 数据加载
num = 100 #数据规模
x1 = torch.rand(num,1)*10
x2 = torch.rand(num,1)*10
x = torch.cat((x1,x2),axis=-1)
y = 2*x1+3*x2+4
y+= torch.randn(num,1)

## 训练
for i in range(100):
    out = model(x)
    loss=mseloss(out,y)
    loss.backward()
    opt.step()
    opt.zero_grad()

w = model.weight.detach().numpy().reshape(-1)
b = model.bias.item()

## 测试
x1test = torch.arange(10)
x2test = torch.arange(10)
x1test,x2test=torch.meshgrid(x1test,x2test,indexing='xy')
ytest = w[0]*x1test + w[1]*x2test+b

fig = plt.figure()
ax=fig.add_subplot(projection="3d")
ax.scatter(x1,x2,y)
ax.plot_wireframe(x1test,x2test,ytest,color='r')
plt.show()
