#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
# catplot——绘制分类散点图
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.catplot(data=tips, x="day", y="total_bill", kind="strip")
sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")

plt.show()
"""

################################
# catplot——分类分布图
################################

"""
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", y="total_bill", kind="box")
sns.catplot(data=tips, x="day", y="total_bill", kind="boxen")
sns.catplot(data=tips, x="day", y="total_bill", kind="violin")

plt.show()
"""


################################
# catplot——分类估计图
################################

"""
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.catplot(data=tips, x="day", y="total_bill", kind="bar")
# sns.catplot(data=tips, x="day", y="total_bill", kind="point")
sns.catplot(data=tips, x="day", kind="count")
# sns.lineplot(data=tips, x="day", y="total_bill")


plt.show()
"""


################################
# catplot，stripplot
################################

"""
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.catplot(data=tips, x="day", y="total_bill", kind="strip", hue="sex")
# sns.catplot(data=tips, x="day", y="total_bill", kind="strip", hue="sex", dodge=True)
sns.catplot(data=tips, x="day", y="total_bill", kind="strip", jitter=1.2)

plt.show()
"""

################################
# catplot，swarmplot
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", hue="sex")
sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", hue="sex", dodge=True)
# sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", jitter=1.2)

plt.show()
'''
################################
# catplot，boxplot
################################
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

#sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="sex")
#sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="sex", dodge=True) #dodge参数没有效果
#sns.catplot(data=tips, y="day", x="total_bill", kind="box")# 省略orient="h"，因为y是分类数据
sns.catplot(data=tips, x="day", y="total_bill", kind="box", whis=np.inf)

plt.show()
'''

################################
# catplot，boxenplot
################################
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", y="total_bill", kind="boxen", hue="sex", dodge=True)
sns.catplot(data=tips, x="day", y="total_bill", kind="boxen", k_depth=4)
sns.catplot(data=tips, x="day", y="total_bill", kind="boxen", k_depth="proportion")

plt.show()
'''


################################
# catplot，boxenplot
################################
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

#sns.catplot(data=tips, x="day", y="total_bill", kind="violin", hue="sex", dodge=True)
sns.catplot(data=tips, x="day", y="total_bill", kind="violin", hue="sex",split=True)
#sns.catplot(data=tips, x="day", y="total_bill", kind="violin", bw=0.2)

fig=plt.figure()
ax=fig.subplots(1,5)
sns.violinplot(data=tips, x="sex", y="total_bill", inner=None,ax=ax[0])
sns.violinplot(data=tips, x="sex", y="total_bill", inner="box",ax=ax[1])
sns.violinplot(data=tips, x="sex", y="total_bill", inner="quartile",ax=ax[2])
sns.violinplot(data=tips, x="sex", y="total_bill", inner="point",ax=ax[3])
sns.violinplot(data=tips, x="sex", y="total_bill", inner="stick",ax=ax[4])
ax[0].set_title("inner=None")
ax[1].set_title("inner=box")
ax[2].set_title("inner=quartile")
ax[3].set_title("inner=point")
ax[4].set_title("inner=stick")
fig.tight_layout(pad=0.1)

plt.show()
'''


################################
# catplot，barplot
################################
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

#sns.catplot(data=tips, x="day", y="total_bill", kind="bar", hue="sex", dodge=True)
#sns.catplot(data=tips, x="day", y="total_bill", kind="bar", capsize=0.3, errwidth=5, ci=50)
sns.catplot(data=tips, x="day", y="total_bill", kind="bar", estimator=max)

plt.show()
'''

################################
# catplot，pointplot
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")
_,ax=plt.subplots(1,2)

sns.pointplot(data=tips, x="day", y="total_bill", ax=ax[0])
sns.pointplot(data=tips, x="day", y="total_bill", join=False,ax=ax[1])
ax[0].set_title("join=True")
ax[1].set_title("join=False")

plt.show()
'''


################################
# catplot，countplot
################################

import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", kind="count", hue="sex")

plt.show()