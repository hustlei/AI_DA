import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.subplots()

ax.arrow(0.3,0.3,0.2,0.2,         #箭头位置0.3,0.3到0.5,0.5
        width=0.1,                #箭杆宽度
        fill=True,fc='gray',      #填充色
        lw=5,ec='orangered',      #边缘线条
        hatch='+')                #图案填充，颜色为ec

plt.show()