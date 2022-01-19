#!/usr/bin/env python
# -*- coding:utf-8 -*-


#==============================#
#非结构化数据绘图
#==============================#

################################
#tricontour三角网格等值线
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
z=x**2+y**2
ax.plot(x, y, 'o', markersize=2, color='lightgrey')
#ax.tricontour(x,y,z)
ax.tricontourf(x,y,z,levels=8,cmap="Oranges")#colors和cmap不能同时设置

plt.show()
'''

################################
#tripcolor三角网格伪彩色图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
z=x**2+y**2
ax.plot(x, y, 'o', markersize=2, color='lightgrey')
ax.tripcolor(x,y,z)

plt.show()
'''


################################
#tripcolor三角网格伪彩色图
################################

import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
#ax.triplot(x,y)
ax.triplot(x,y, 'o-', markersize=2, color="C0", mfc='lightgrey',mec='lightgray')

plt.show()