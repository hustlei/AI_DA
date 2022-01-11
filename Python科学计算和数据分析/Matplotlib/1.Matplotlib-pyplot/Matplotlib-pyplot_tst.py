#!/usr/bin/env python
# -*- coding:utf-8 -*-

#准备数据
import numpy as np
x=np.linspace(-np.pi,np.pi,30)
y=np.sin(x)


#导入matplotlib库pyplot模块
import matplotlib.pyplot as plt


################################
#面向对象
################################
'''
#创建画布
fig=plt.figure()

#创建图表
ax=fig.subplots()

#绘制折线图，设置点线样式，设置线条名称
ax.plot(x,y,'+r--', label='line1',mec='b',ms=10)  #点为蓝色+(大小为10)，线为红色虚线


#设置坐标轴
ax.set_xlabel('X axis')               #坐标轴文本标签
ax.set_ylabel('Y axis')
ax.set_xticks([-4,-2,0,2,4])       #主刻度
ax.set_xticks(np.arange(-4,4,0.5),minor=True)  #次刻度
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_yticks(np.arange(-1.5,1.5,0.1),minor=True)
ax.set_xlim(-3.5,3.5)                #设置显示刻度范围
ax.set_ylim(-1.5,1.5)

ax.grid(True,c='grey',linestyle=':')    #显示主刻度网格

#设置图例
ax.legend()                            #注意需要绘图时，指定label参数

#设置标题
ax.set_title("sample")

#保存显示图形
fig.savefig("sample.png")
plt.show()
'''
################################
#过程式
################################
'''
#绘制折线图，设置点线样式，设置线条名称
plt.plot(x,y,'+r-.', label='line1')   #点为蓝色+，线为红色点划线


#设置坐标轴
plt.xlabel('X axis')                  #坐标轴文本标签
plt.ylabel('Y axis')
plt.xticks([-4,-2,0,2,4])             #主刻度
#plt.xticks(np.arange(-4,4,0.5),minor=True)  #plt不支持次刻度设置
plt.yticks([-1,-0.5,0,0.5,1], rotation=30)
plt.xlim(-3.5,3.5)                    #设置显示刻度范围
plt.ylim(-1.5,1.5)

plt.grid(True,c='gray',linestyle=':') #显示主刻度网格

#设置图例
plt.legend()                          #注意需要绘图时，指定label参数

#设置标题
plt.title("sample")

#保存显示图形
plt.savefig("sample.png")

plt.show()

'''


################################
#字典数据
################################

'''
import numpy as np
import matplotlib.pyplot as plt

data = {'x': np.arange(50),
        'y': np.random.randint(0, 50, 50),
        'color': np.random.randn(50)}

plt.scatter('x', 'y', c='color', data=data)

plt.show()
'''


################################
#多子图
################################


fig=plt.figure()
ax1=fig.add_subplot(2,2,1)  #2行2列，第1个子图
ax1.plot(x,y,'r-')
ax2=fig.add_subplot(223)    #2行2列，第3个子图
ax2.plot(x,y,'b:')
ax3=fig.add_subplot(1,2,2)  #跨行子图
ax3.plot(x,y,'Dg--')

ax1.set_title("axes1")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax2.set_title("axes2")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax3.set_title("axes3")
ax3.set_xlabel("x")
ax3.set_ylabel("y")

#fig.subplots_adjust(wspace=0.5,hspace=0.5)


fig.suptitle("figtitle", x=0.5, y=0.98)
fig.supxlabel("figxlabel", x=0.5, y=0.02)
fig.supylabel("figylabel", x=0.02, y=0.5)
fig.tight_layout(pad=1)
plt.show()

