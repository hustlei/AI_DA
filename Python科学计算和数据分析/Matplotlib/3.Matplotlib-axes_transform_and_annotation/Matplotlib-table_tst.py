import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.subplots()

#ax.table(np.array([['11','12'],['21','22']]),loc="upper right")
#ax.table(cellColours=np.array([['C1','C2'],['C3','C4']]))
ax.table([['11','12'],['21','22']],
            cellColours=[['C1','C2'],['C3','C4']],
            loc="upper right",
            colWidths=[0.2,0.2],
            cellLoc="center",
            rowLabels=['row1','row2'],
            colLabels=['col1','col2'],
            rowColours=['C0','C0'],
            colColours=['C5','C5'])

ax.set_xticks([])

plt.show()