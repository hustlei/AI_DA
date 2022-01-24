#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#动画
################################


################################
#手动更新图形实现动画
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.axis([0,10,0,2])
ax.set_aspect(3)

while True:
    ax.plot(t,y)
    plt.pause(0.1)
    ax.cla()
    t+=np.pi/30
    y=np.sin(t)
'''

################################
#matplotlib.animation模块FuncAnimation实现动画
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.set_aspect(3)
ax.plot(t,y,'--',c='gray')
line=ax.plot(t,y,c='C2')

def update(i):  #帧更新函数
    global t    #直接引用全局变量，也可以通过函数的frames或fargs参数传递。
    t+=0.1
    y=np.sin(t)
    line[0].set_ydata(y)
    return line

ani=FuncAnimation(fig,update,interval=100) #绘制动画
#ani.save("animate_func_basic.gif")  #保存动画
plt.show() #显示动画
'''

################################
#matplotlib.animation模块ArtistAnimation实现动画
################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
fig = plt.figure()
ax = fig.subplots()

arts=[]
t=np.linspace(0,np.pi*2,20)
for i in range(20):
    t+=np.pi*2/20
    y=np.sin(t)
    lines=ax.plot(y,'--',c='gray')  #绘制一帧图形
    arts.append(lines)              #每帧图形都保存到列表中

ani=ArtistAnimation(fig,arts,interval=200) #绘制动画
ani.save("animate_artists_basic.gif")  #保存动画
plt.show() #显示动画
