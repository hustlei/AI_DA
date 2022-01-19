import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax1=fig.subplots()
x=np.linspace(-np.pi,np.pi,200) 
y=np.sin(x)
ax1.plot(x,y)

'''
ax1.annotate("max point", (np.pi/2,1), xytext=(2,1.2), 
        #arrowprops={})  #使用默认值绘制箭头
        #arrowprops={"width":3})  #设置箭头线条宽度，默认宽度为3
        #arrowprops={"headwidth":12,"headlength":15})  #设置箭头三角形长宽，默认都是12
        #arrowprops={"shrink":0.1})  #箭头线条长度缩短0.1
        arrowprops={"width":1, "headwidth":6,"headlength":10,"color":"gray"})
'''

ax1.text(1,1,"max point",rotation=30,bbox=dict(fc='green',alpha=0.3))

ax1.grid(True)
plt.show()