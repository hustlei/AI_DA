import torch
from torch import nn
from matplotlib import pyplot as plt


#1.模型定义
layer=nn.Linear(1,1)  #定义网络结构
mseloss=nn.MSELoss()  #定义损失函数
opt=torch.optim.SGD(params=layer.parameters(), lr=0.01) #定义优化函数


#2.数据加载
## y=2x+3线型方程上随机取（0,10）内的num个点
num=100
x = torch.rand(num,1)*10      #使用nn.Linear，数据规模必须是第一维
y = 2*x+3
y += torch.randn(num,1)*2  #y=2x+3直线上加上随机噪声

## 绘制x,y分布
plt.scatter(x,y)



#3.训练模型
for i in range(100):
    opt.zero_grad() #lay.zero_grad()相同，清零梯度
    out=layer(x) #forward
    loss=mseloss(out,y) #计算loss
    loss.backward() #backward
    opt.step() #更新梯度


#4.可视化及测试
txt="y={0:.3f}x+{1:.3f}".format(layer.weight.item(),layer.bias.item())
txt+="\nloop count:{}".format(i+1)
txt+="\nStandard Deviation:{:.3f}".format(loss.item()) #标准差，即均方差
plt.text(.25,25,txt,fontsize=10,color="lightblue") 

##测试网络的代码通常在 torch.no_grad() 下完成
with torch.no_grad():
    x1 = torch.linspace(0,10,11).reshape(-1,1)
    y1 = layer(x1)  #.detach().numpy() #layer.weight.item()*x1 + layer.bias.item()

plt.plot(x1,y1,'r')
plt.xlim(0,10)
plt.ylim(0,30)
plt.show()