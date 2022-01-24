#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#基本方法
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()

ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)   #创建3d坐标系的第二种方法

theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
z = np.linspace(-2, 2, 100)

ax3d.plot(x,y,z)  #绘制3d螺旋线

plt.show()
'''

################################
#plot三维折线
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
z = np.linspace(0.5, 1.5, 100)

ax3d.plot(x,y,z)           #绘制3d螺旋线
ax3d.plot(x,y,zdir='z')    #绘制x,y平面图形。plot(x,y)也ok，'z'是默认值
ax3d.plot(x,y,2,zdir='z')  #绘制x,y平面图形指定高度z为2
ax3d.plot(y,z,zdir='x')    #绘制y,z平面图
ax3d.plot(y,z,-2,zdir='x') #绘制y,z平面图,指定x坐标值为-2

plt.show()
'''

################################
#scatter三维散点
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x = np.random.randn(50)
y = np.random.randn(50)
z = np.random.randn(50)
s = np.random.randn(50)*100

#ax3d.scatter(x,y,z)  #绘制3d散点图
#ax3d.scatter(x,y,z,marker=['*','o',...]) #设置不同的点样式
ax3d.scatter(x,y,z,s=s,c=s)  #绘制3d散点图
ax3d.scatter(x,y,-3,zdir='z',c='r') #3d坐标系绘制平面散点

plt.show()
'''



################################
#bar3d三维柱形图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x = np.arange(5)
y = np.arange(5)
z = np.zeros(5)  #柱子底部坐标
dx=1    #柱子平面宽度
dy=1    #柱子平面深度
dz=np.random.randint(1,15,5)    #柱子高度

ax3d.bar3d(x,y,z,dx,dy,dz)  #绘制3d柱形图

plt.show()
'''


################################
#stem三维火柴图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

t=np.linspace(-np.pi,np.pi,50)
x = np.sin(t)
y = np.cos(t)
z = np.linspace(-2,2,50)

ax3d.stem(x,y,z)  #绘制3d火柴图
#ax3d.stem(x,y,z,orientation="x", bottom=-2) #火柴根在yz平面

plt.show()
'''

################################
#errorbar误差图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

t=np.linspace(-np.pi,np.pi,50)
x = np.sin(t)
y = np.cos(t)
z = np.linspace(-4,4,50)
zerr=np.random.randn(50)

#ax3d.errorbar(x,y,z,zerr,capsize=2)  #只有z方向误差
#ax3d.errorbar(x,y,z,zerr,0.2,0.1,capsize=2)  #同时显示zerr,yerr,xerr，注意是三个误差线
ax3d.errorbar(x,y,z,zerr,capsize=2,errorevery=2) #每两个数据点绘制一个误差线。

plt.show()
'''

################################
#plot_wireframe三维网格面
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-2:2:0.2,-2:2:0.2]
z = x*np.exp(-x**2-y**2)

#ax3d.plot_wireframe(x,y,z)
#ax3d.plot_wireframe(x,y,z,rstride=2,cstride=2)# 两条线合并为一条线
ax3d.plot_wireframe(x,y,z,rcount=10,ccount=12)#设置最大显示线条数

plt.show()
'''


################################
#plot_surface三维曲面
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-3:3:0.2,-3:3:0.2]
z = x*np.exp(-x**2-y**2)

#ax3d.plot_surface(x,y,z)
#ax3d.plot_surface(x,y,z,rstride=2,cstride=2)# 两条线合并为一条线
#ax3d.plot_surface(x,y,z,rcount=16,ccount=18)#设置最大显示线条数
#ax3d.plot_surface(x,y,z,cmap="YlOrRd")
ax3d.plot_surface(x,y,z,cmap="YlOrRd")

plt.show()
'''


################################
#plot_trisurf非结构化三角曲面
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x=np.random.randn(200)*2
y=np.random.randn(200)*2
z = x*np.exp(-x**2-y**2)

#ax3d.plot_trisurf(x,y,z)
ax3d.plot_trisurf(x,y,z,cmap="YlOrRd")

plt.show()
'''

################################
#tricontour非结构三角网格等值线
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x=np.random.randn(200)*2
y=np.random.randn(200)*2
z = x*np.exp(-x**2-y**2)

#ax3d.tricontour(x,y,z)
ax3d.tricontour(x,y,z,levels=10,cmap="coolwarm")
#ax3d.tricontour(x,y,z,zdir='x',levels=10,cmap="coolwarm") #绘制x方向等值线

plt.show()
'''

################################
#contour三维等值线
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-3:3:0.2,-3:3:0.2]
z=x*np.exp(-x**2-y**2)

#ax3d.contour(x,y,z)
ax3d.contour(x,y,z,levels=10,cmap="coolwarm")  #指定等高线数和颜色
#ax3d.contourf(x,y,z,levels=10,cmap="coolwarm") #填充等高线
#ax3d.contour(x,y,z,zdir='x',levels=10)  #x方向等高线

#投影
#ax3d.contour(x,y,z,levels=10,zdir='x',offset=-3)
#ax3d.contour(x,y,z,levels=10,zdir='y',offset=3)
#ax3d.contour(x,y,z,levels=10,zdir='z',offset=-0.4)

plt.show()
'''


################################
#3d矢量图
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

t=np.linspace(-np.pi,np.pi,20)
x=np.sin(t)
y=np.cos(t)
z=np.linspace(-1,1,20)
u=np.sin(t+0.1)-x
v=np.cos(t+0.1)-y
w=0.1
ax3d.quiver(x,y,z,u,v,w)  #在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头

plt.show()
'''


################################
#3d文本
################################
'''
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

ax3d.text(0.5,0.5,0.5,'3dtext',c='r')   #在指定坐标处绘制文本。文本永远朝向用户
ax3d.text(0.1,0.1,0.5,'3dtextz',c='r',zdir='z')  #文本沿z轴方向打印
#ax3d.text2D(0.1,0.1,'2dtext',c='b') #好像效果和官方文档不一致。

plt.show()
'''


################################
#3d图形视角旋转动画
################################
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# load some test data for demonstration and plot a wireframe
X, Y, Z = axes3d.get_test_data(0.1)
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)  #设置视角
    plt.draw()  #重绘图
    plt.pause(.1)   #暂停
'''

################################
#3d体元素
################################
'''
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')

#filled为bool类型数组，在True的元素下标位置绘制体元素
i,j,k=np.indices((3,3,3))
filled= (i==j) & (j==k)  #3行3列3层，对角线为True
c=plt.get_cmap('RdBu')(np.linspace(0,1,27)).reshape(3,3,3,4)

#ax3d.voxels(filled)             #filled为True的位置绘制六面体
ax3d.voxels(filled,facecolors=c) #filled为True的位置绘制六面体,并设置颜色

plt.show()
'''



'''
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax=fig.add_subplot(projection='3d')

filled = np.ones((3,3,3))
filled[0,0,0]=False
x, y, z = np.indices((4,4,4))**1.2  #x,y,z的三个维度都必须比filled大1.
ax.voxels(x, y, z, filled, edgecolors='C1')

plt.show()
'''
