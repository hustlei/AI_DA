import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt

order = 3
# 1.模型定义
layer = nn.Linear(order, 1)  # 定义网络结构,n阶多项式
mseloss = nn.MSELoss()  # 定义损失函数
opt = torch.optim.SGD(params=layer.parameters(), lr=0.00000005)  # 定义优化函数

# 2.数据加载
"""y=x^3+2x^2+3x+4方程上随机取（0,10）内的num个点"""
num = 100
x = torch.rand(num, 1) * 10  # 使用nn.Linear，数据规模必须是第一维
y = x ** 3 + 2 * x ** 2 + 3 * x + 4
y += torch.randn(num, 1)  # 加上随机噪声

"""一元三阶方程修改为3元一阶线性方程"""
def data_transform(x):
    x_tmp = x
    x_transformed = x
    for i in range(order-1):
        x_tmp *= x
        x_transformed = torch.cat((x_transformed, x_tmp), axis=-1)
    return x_transformed

x_transformed=data_transform(x)

# 3.训练模型
for i in range(10000000):
    opt.zero_grad()  # lay.zero_grad()相同，清零梯度
    out = layer(x_transformed)  # forward
    loss = mseloss(out, y)  # 计算loss
    loss.backward()  # backward
    opt.step()  # 更新梯度

# 4.可视化及测试
txt = "y="
for ii in range(order,0,-1):
    txt+="{:+.3f}x^{}".format(layer.weight[0,ii-1].item(),ii)
txt += "{:+.3f}".format(layer.bias.item())
txt += "\nloop count:{}".format(i + 1)
txt += "\nStandard Deviation:{:.3f}".format(loss.item())  # 标准差，即均方差
plt.text(.25, 25, txt, fontsize=10, color="blue")

##测试网络的代码通常在 torch.no_grad() 下完成
with torch.no_grad():
    x1 = torch.linspace(0, 10, 11).reshape(-1, 1)
    x1t = data_transform(x1)
    y1 = layer(x1t)  # .detach().numpy() #layer.weight.item()*x1 + layer.bias.item()

plt.scatter(x,y)
#plt.scatter(x,out.detach().numpy(), color='g')
plt.plot(x1, y1, 'r')
plt.show()
