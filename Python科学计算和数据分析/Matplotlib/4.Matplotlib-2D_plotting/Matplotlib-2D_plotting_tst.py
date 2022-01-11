#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#plot折线图
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.linspace(0, 4*np.pi,50)
y1=np.sin(x)
y2=np.cos(x)
y3=np.cos(x+np.pi)

#绘制'line1'。线条为灰色，线型为点划线('-.')，线宽为1
ax.plot(x,y1,c='gray', linestyle='-.', linewidth=1, label='line1')
#绘制'line2'。线条为红色，线型为虚线('--')
ax.plot(x,y2,'r--',label='line2')
#绘制'line3'。同时显示点和线。
#点大小为10，填充颜色为绿色，边缘颜色为红色，边缘宽度为1
#线为蓝色实线，线条宽度为2
ax.plot(x,y3,'ob-', mfc='g', mec='r', ms=10, linewidth=2 ,label='line3')

ax.legend(loc="upper right")#设置图例
plt.show()
'''



################################
#plot折线图
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.linspace(0, 4*np.pi,50)
y1=np.sin(x)
y2=np.cos(x)
y3=np.cos(x+np.pi)

#绘制'line1'。线条为灰色，线型为点划线('-.')，线宽为1
ax.plot(x,y1,c='gray', linestyle='-.', linewidth=1, label='line1')
#绘制'line2'。线条为红色，线型为虚线('--')
ax.plot(x,y2,'r--',label='line2')
#绘制'line3'。同时显示点和线。
#点大小为10，填充颜色为绿色，边缘颜色为红色，边缘宽度为1
#线为蓝色实线，线条宽度为2
ax.plot(x,y3,'ob-', mfc='g', mec='r', ms=10, linewidth=2 ,label='line3')

ax.legend(loc="upper right")#设置图例
plt.show()
'''


################################
#plot散点图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.random.randn(50)*10
y=np.random.randn(50)*10
s=(np.random.randn(50)*10)**2
c=np.random.randn(50)*10

#绘制散点图。颜色为c，大小为s（单位points^2），透明度为0.5
ax.scatter(x,y,s=s,c=c,alpha=0.5)

plt.show()
'''












