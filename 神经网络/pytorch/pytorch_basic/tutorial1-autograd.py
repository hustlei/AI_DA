import numpy as np
import matplotlib.pyplot as plt
import torch

x=torch.tensor(2., requires_grad=True)  #浮点数才能求梯度
b=torch.tensor(3.,requires_grad=True)
y= 2*x + b
z=(y-2)**2


z.backward()   #非标量，要转换为标量求梯度

print(x.grad, b.grad)

