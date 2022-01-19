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
#scatter散点图
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


################################
#bar,barh柱状图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表


#准备数据
x=np.arange(6)
y=np.random.uniform(0,10,6)

#ax.bar(x,y) #绘制简单柱状图。用x坐标文本标签

#制定文本作为标签，填充灰色，描边宽度为1，颜色为橙色
ax.bar(x,y,tick_label=list('abcdef'),color='gray',lw=1,ec='orange') 

plt.show()
'''

################################
#pie饼图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表


#准备数据
x=np.arange(6)

#ax.pie(x) #绘制简单饼状图。无标签，默认颜色设置
ax.pie(x,
       labels=list('abcdef'),             #设置标签
       explode=[0,0,0.1,0,0,0],           #第2个饼图炸开
       wedgeprops={'lw':2,'ec':'lightblue'})  #描边
#plt.axis('off')  #不显示坐标轴

plt.show()
'''

################################
#stem火柴图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表


#准备数据
x=np.arange(6)
y=np.random.uniform(1,10,6)

#ax.stem(x,y)
ax.stem(x,y,
        linefmt="--",
        markerfmt="C2o")

plt.show()
'''



################################
#fill_between,fill_betweenx填充图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表


#准备数据
x=np.linspace(0,2*np.pi,200)
y1=np.sin(x)
y2=np.cos(x)

ax.fill_between(x,y1,y2,color='C2')  #填充y1,y2之间的区域
#ax.fill_between(x,y1,y2,where=y2>y1) #只填充y2>y1的区域

plt.show()
'''



################################
#axhline,axvline,axline绘制直线
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.axhline(y=1)
ax.axvline(x=1)
ax.axhline(y=3,xmin=0.1,xmax=0.8,c='r')
ax.axvline(x=3,ymin=0.1,ymax=0.8,c='r')
ax.axline((0,0),(1,1))

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
'''

################################
#hlines,vlines直线组
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.hlines([1,2,3],0,4)
ax.vlines([3,2,1],0.5,4)

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
'''

################################
#axhspan,axvspan填充直线
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.axhspan(1,2)
ax.axvspan(2,3)

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
'''

