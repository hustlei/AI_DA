#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#基本Path图形
################################


################################
#圆形
################################
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig = plt.figure()
ax = fig.subplots()

#circle1=Circle((2,2),1)
circle1=Circle((2,2),1,fc='C2',ec='0.5',lw=2,alpha=0.5)

ax.add_patch(circle1)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#矩形
################################
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig = plt.figure()
ax = fig.subplots()

#rect1=Rectangle((1,1),2,1)
rect1=Rectangle((1,1),2,1,fc='C3',ec='C1',lw=2,alpha=0.5)

ax.add_patch(rect1)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#椭圆形
################################
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig = plt.figure()
ax = fig.subplots()

#e1=Ellipse((2,1.5),2,1)
e1=Ellipse((1,2),1.5,1,45)
e2=Ellipse((2,1.5),2,1,fc='C8',ec='C0',lw=2,alpha=0.5)

ax.add_patch(e1)
ax.add_patch(e2)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''

################################
#圆弧
################################
'''
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
fig = plt.figure()
ax = fig.subplots()

arc1=Arc((1,2),1.5,1)
arc2=Arc((2,1.5),2,1,0,30,120,ec='r')
arc3=Arc((2,1.5),2,1,theta1=200,theta2=300,color='g') #color不能缩写为c

ax.add_patch(arc1)
ax.add_patch(arc2)
ax.add_patch(arc3)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''

################################
#多边形
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
fig = plt.figure()
ax = fig.subplots()

pts=np.array([[0,0],[1,1],[1,0]])
p1=Polygon(pts)
p2=Polygon(np.array([[0,2],[1,2],[2,3]]),color="g")
p3=Polygon(np.array([[2,2],[3,2],[3,1]]),fc="C4",ec='C1')

ax.add_patch(p1)
ax.add_patch(p2)
ax.add_patch(p3)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''

################################
#正多边形
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
fig = plt.figure()
ax = fig.subplots()

p1=RegularPolygon([1,1.5],4,0.8)
p2=RegularPolygon([3,1.5],5,0.8,np.pi/2,color="g",ec='r',alpha=0.5)

ax.add_patch(p1)
ax.add_patch(p2)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#环形
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus
fig = plt.figure()
ax = fig.subplots()

a1=Annulus([1,2],0.8,0.2)
a2=Annulus([3,2],1,0.2, fc="0.5",ec='r',alpha=0.5)

ax.add_patch(a1)
ax.add_patch(a2)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#楔形、扇形
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
fig = plt.figure()
ax = fig.subplots()

a1=Wedge([1,2],0.8,90,120)
a2=Wedge([3,2],1,45,180, 0.5, fc="0.5",ec='r',alpha=0.5)

ax.add_patch(a1)
ax.add_patch(a2)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#阴影
################################

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Shadow
fig = plt.figure()
ax = fig.subplots()

c1=Circle([1,2],0.8)
c2=Circle([3,2],0.8,color='g')
s1=Shadow(c2,0.05,-0.05,lw=0)

ax.add_patch(c1)
ax.add_patch(c2)
ax.add_patch(s1)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''

################################
#Path
################################
################################
#基本方法
################################

'''
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.add_subplot()

path = Path.circle([2,2],1,color='r')
patch = PathPatch(path, facecolor='orange', lw=2)

ax.add_patch(patch)
ax.axis([0,4,0,4])
ax.set_aspect(1)
plt.show()
'''


################################
#文本路径
################################

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.subplots()

t1=TextPath([1,40],"textpath")
t2=TextPath([1,25],"textpath2",8)
t3=TextPath([1,10],"$e^{i\pi}+1=0$",usetex=True)

ax.add_patch(PathPatch(t1,fc='r'))
ax.add_patch(PathPatch(t2,fc='g'))
ax.add_patch(PathPatch(t3,fc='b'))
ax.axis([0,50,0,50])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
'''


################################
#自定义路径
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.subplots()

verts = [(0., 0.),  # left, bottom
   (0., 1.),  # left, top
   (1., 1.),  # right, top
   (1., 0.),  # right, bottom
   (0., 0.),  # ignored
]
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]
path = Path(verts, codes)
patch = PathPatch(path,fc='C4')

ax.add_patch(patch)
ax.axis([-1,2,-1,2])
ax.set_aspect(1)
plt.show()
'''

################################
#文本特效
################################

'''
import matplotlib.pyplot as plt
from matplotlib import patheffects
fig = plt.figure()
ax = fig.subplots()

text = ax.text(0.1, 0.5, 'Hello world!', size=20)
text.set_path_effects([patheffects.Normal()])
plt.show()
'''

################################
#绘图特效
################################

'''
import matplotlib.pyplot as plt
from matplotlib import patheffects
fig = plt.figure()
ax = fig.subplots()

ax.plot([0,3,2,4], path_effects=[patheffects.Normal()])
plt.show()
'''


################################
#绘图特效
################################

'''
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patheffects import Normal,PathPatchEffect, SimpleLineShadow,Stroke,TickedStroke
fig = plt.figure()
ax = fig.subplots()

#线条阴影
ax.plot([0,10,20,30],[0,5,2,10],lw=3,path_effects=[SimpleLineShadow(),Normal()])

#文本轮廓
text1 = ax.text(1,15,'text stands out',size=30,c='0.8')
text1.set_path_effects([Stroke(linewidth=1, foreground='C0')]) #描边特效

#文本填充阴影
text2 = ax.text(1, 30, 'Hatch shadow',size=40,weight=800)
text2.set_path_effects([PathPatchEffect((4,-4),hatch='xxxx'),PathPatchEffect(fc='w',lw=1)])

ax.axis([0,50,0,50])
plt.show()
'''


################################
#蒙版
################################
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
fig = plt.figure()
ax = fig.subplots()

textpath=TextPath((1,20),'textpath',prop={'weight':1000})
img=ax.imshow(np.random.randn(50,50),interpolation="bilinear")
img.set_clip_path(textpath,img.get_transform())

ax.axis([0,50,0,50])
plt.show()
'''


################################
#集合
################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
fig=plt.figure()
ax = fig.subplots()

ws = np.full(3,15)
hs = np.full(3,10)
angles = np.arange(3)*30
offsets= np.array([[10,10],[20,20],[30,30]])
cs=['C0','C2','C4']

ec = EllipseCollection(ws, hs, angles,offsets=offsets,transOffset=ax.transData,facecolors=cs)
ax.add_collection(ec)

ax.autoscale_view()
plt.show()