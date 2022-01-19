#!/usr/bin/env python
# -*- coding:utf-8 -*-


#==============================#
#非结构化数据绘图
#==============================#

################################
#acorr自相关
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x=np.sin(t)
#ax.acorr(x)
ax.acorr(x,maxlags=25)

plt.show()
'''

################################
#xcorr互相关
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)
x2=np.cos(t)

#ax.xcorr(x1,x2)
ax.xcorr(x1,x2,maxlags=25)

plt.show()
'''

################################
#psd功率谱密度
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)

#ax.xcorr(x1,x2)
ax.psd(x1)

#####
#fft变换对比
#####

# k=np.fft.rfftfreq(len(t),d=1/25)
# xk=np.fft.rfft(x1)
# ax.plot(k[:10],xk[:10])


plt.show()
'''





################################
#csd交叉谱密度
################################


import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)
x2=np.cos(t)

ax.csd(x1,x2)

plt.show()
