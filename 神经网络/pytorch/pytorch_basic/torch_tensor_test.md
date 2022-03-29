# pytorch tensor学习


```python
import numpy as np
import torch
from torch.autograd import variable
from matplotlib import pyplot as plt
```

## 创建tensor


```python
a=torch.empty(2,3)#torch.Tensor(2,3)
b=torch.zeros(2,3)
c=torch.ones(2,3,dtype=torch.long)
d=torch.eye(2,3)
e=torch.rand(3,2)
shape=(2,3,)
f=torch.rand(shape)
print("a:",a)
print("b:",b)
print("c:",c)
print("d:",d)
print("e:",e)
```

    a: tensor([[6.6474e+22, 1.0860e-05, 1.6914e+22],
            [3.4032e-06, 2.1375e+20, 1.0524e+21]])
    b: tensor([[0., 0., 0.],
            [0., 0., 0.]])
    c: tensor([[1, 1, 1],
            [1, 1, 1]])
    d: tensor([[1., 0., 0.],
            [0., 1., 0.]])
    e: tensor([[0.1638, 0.2589],
            [0.5169, 0.5960],
            [0.4992, 0.5439]])
    
