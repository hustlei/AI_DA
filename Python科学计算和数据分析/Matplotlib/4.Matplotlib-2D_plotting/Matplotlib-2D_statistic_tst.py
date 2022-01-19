#!/usr/bin/env python
# -*- coding:utf-8 -*-


#==============================#
#统计绘图
#==============================#

################################
#hist直方图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(1000)
#ax.hist(x)
ax.hist(x,bins=8,ec='w',lw=1,color='C1') #设置分组数，设置edgecolor和linewidth

plt.show()
'''

################################
#hist2d直方图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(1000)
y=np.random.randn(1000)
#ax.hist2d(x,y)
h=ax.hist2d(x,y,bins=20,cmap="Oranges") #设置分组数
#ec='w',lw=1,density=True)。设置edgecolor和linewidth

plt.colorbar(h[3],ax=ax)
plt.show()
'''

################################
#boxplot箱型图
################################


import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100,5)  #5列，所以绘制5个箱型图
#ax.boxplot(x)
ax.boxplot(x,sym="r+",whis=1.25) #设置分组数

plt.show()


################################
#errorbar误差图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x = np.arange(-5,5,0.5)
y = np.sin(x/2)
yerr = np.random.randn(20)/5
#ax.errorbar(x,y,yerr)
ax.errorbar(x,y,yerr,fmt='o-',lw=2,capsize=4,c="C2") #设置x,y样式及线宽，设置误差样式

plt.show()
'''

################################
#hexbin六边形填充图
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)
plt.style.use('_mpl-gallery-nogrid')

x = np.random.randn(1000)
y = x+np.random.randn(1000)/5
#ax.hexbin(x,y)
h=ax.hexbin(x,y,gridsize=20,edgecolors='w')

plt.colorbar(h)
plt.show()
'''