#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
# displot——绘制直方图、核密度图、累积分布图
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="bill_length_mm")
sns.displot(data=penguins, x="bill_length_mm", kind="kde")
sns.displot(data=penguins, x="bill_length_mm", kind="ecdf")

plt.show()
"""


################################
# displot—-多种分布图组合在一个子图
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="bill_length_mm", kde=True)  # 在直方图中同时绘制核密度图

plt.show()
"""

################################
# displot—-多种分布图组合在一个子图
################################

"""
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="bill_length_mm", rug=True)
sns.displot(data=penguins, x="bill_length_mm", kind="kde", rug=True)

plt.show()
"""

################################
# displot—-双变量分布图（直方图、核密度）
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns

penguins = sns.load_dataset("penguins")

# sns.displot(data=penguins, x="flipper_length_mm", y="bill_length_mm")
# sns.displot(data=penguins, x="flipper_length_mm", y="bill_length_mm", kind="kde")


sns.displot(
    data=penguins, x="flipper_length_mm", y="bill_length_mm", kind="kde", rug=True
)

plt.show()
"""


################################
# displot—-分组分布图
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="flipper_length_mm", hue="species")
sns.displot(data=penguins, x="flipper_length_mm", kind="kde", hue="species")
sns.displot(data=penguins, x="flipper_length_mm", kind="ecdf", hue="species")

plt.show()
'''


'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="flipper_length_mm", col="species")
sns.displot(data=penguins, x="flipper_length_mm", hue="species",col="species")

plt.show()
'''

################################
# hsitplot,displot—-直方图基本参数
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")

#sns.displot(data=penguins, x="flipper_length_mm")
#sns.histplot(data=penguins, y="flipper_length_mm")

#sns.displot(data=penguins, x="flipper_length_mm",bins=10)
#sns.displot(data=penguins, x="flipper_length_mm",binwidth=20)
#sns.displot(data=penguins, x="flipper_length_mm", discrete=True)

sns.displot(data=penguins, x="flipper_length_mm",fill=False,element="step")
#sns.displot(data=penguins, x="flipper_length_mm",fill=False,element="poly")

plt.show()
'''

################################
# hsitplot,displot—-统计量
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")

sns.displot(data=penguins, x="flipper_length_mm",stat="probability")
sns.displot(data=penguins, x="flipper_length_mm",stat="percent")

plt.show()
'''


################################
# hsitplot,displot—-multiple
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")
fig=plt.figure()
ax=fig.subplots(2,2)

sns.histplot(data=penguins, x="flipper_length_mm",hue="species",multiple="layer",ax=ax[0][0])
ax[0][0].set_title("layer")
sns.histplot(data=penguins, x="flipper_length_mm",hue="species",multiple="stack",ax=ax[0][1])
ax[0][1].set_title("stack")
sns.histplot(data=penguins, x="flipper_length_mm",hue="species",multiple="dodge",ax=ax[1][0])
ax[1][0].set_title("dodge")
sns.histplot(data=penguins, x="flipper_length_mm",hue="species",multiple="fill",ax=ax[1][1])
ax[1][1].set_title("fill")
fig.tight_layout(pad=1.5)

plt.show()
'''



################################
# kdeplot,displot--基本绘图
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")
fig=plt.figure()
ax=fig.subplots(3,2)

sns.kdeplot(data=penguins, x="flipper_length_mm",bw_method=0.5,ax=ax[0][0])
ax[0][0].set_title("bw_method=0.5")
sns.kdeplot(data=penguins, x="flipper_length_mm",bw_method=0.2,ax=ax[0][1])
ax[0][1].set_title("bw_method=0.2")
sns.kdeplot(data=penguins, x="flipper_length_mm",bw_adjust=0.5,ax=ax[1][0])
ax[1][0].set_title("bw_adjust=0.5")
sns.kdeplot(data=penguins, x="flipper_length_mm",bw_adjust=1,ax=ax[1][1])
ax[1][1].set_title("bw_adjust=1")
sns.kdeplot(data=penguins, x="flipper_length_mm",ax=ax[2][0])
ax[2][0].set_title("default kde")
sns.histplot(data=penguins, x="flipper_length_mm",ax=ax[2][1],kde=True)
ax[2][1].set_title("hist+kde")
fig.tight_layout(pad=1.5)

plt.show()
'''

################################
# kdeplot,displot--kde分组
################################

'''
import matplotlib.pyplot as plt
import seaborn as sns
penguins = sns.load_dataset("penguins")
fig=plt.figure()
ax=fig.subplots(2,2)

sns.kdeplot(data=penguins, x="flipper_length_mm",hue="species",ax=ax[0][0])
ax[0][0].set_title("basic")
sns.kdeplot(data=penguins, x="flipper_length_mm",hue="species",multiple="stack",ax=ax[0][1])
ax[0][1].set_title("stack")
sns.kdeplot(data=penguins, x="flipper_length_mm",hue="species",fill=True,ax=ax[1][0])
ax[1][0].set_title("dodge")
sns.kdeplot(data=penguins, x="flipper_length_mm",hue="species",multiple="fill",ax=ax[1][1])
ax[1][1].set_title("fill")
fig.tight_layout(pad=1)

plt.show()
'''

################################
# kdeplot,displot--kde双参数
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
geyser = sns.load_dataset("geyser")
_,ax=plt.subplots(1,3)

sns.kdeplot(data=geyser, x="waiting", y="duration",ax=ax[0])
ax[0].set_title("kde with x,y")
sns.kdeplot(data=geyser, x="waiting", y="duration", hue="kind",ax=ax[1])
ax[1].set_title("hue='kind'")
sns.kdeplot(data=geyser, x="waiting", y="duration", hue="kind", levels=5,ax=ax[2])
ax[2].set_title("levels=5")

plt.show()
'''

################################
# kdeplot,displot--kde样式
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
geyser = sns.load_dataset("geyser")

sns.kdeplot(data=geyser, x="waiting", y="duration", alpha=.5, linewidth=0, fill=True, cmap="hsv")

plt.show()
'''







################################
# kdeplot,displot--ecdf
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns
geyser = sns.load_dataset("geyser")

#sns.ecdfplot(data=geyser, x="waiting")
#sns.ecdfplot(data=geyser, x="waiting",hue="kind")
#sns.ecdfplot(data=geyser, x="waiting",stat="count")
sns.ecdfplot(data=geyser, x="waiting",complementary=True)

plt.show()
"""

################################
# kdeplot,displot--rug
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
geyser = sns.load_dataset("geyser")
# _,ax=plt.subplots(1,3)
#
# sns.rugplot(data=geyser, x="waiting", ax=ax[0])
# ax[0].set_title("rug")
# sns.rugplot(data=geyser, x="waiting", y="duration", ax=ax[1])
# ax[1].set_title("rug with x,y")
# sns.rugplot(data=geyser, x="waiting", y="duration",hue="kind", ax=ax[2])
# ax[2].set_title("hue='kind'")
sns.rugplot(data=geyser, x="waiting", height=0.5)

plt.show()
'''

import matplotlib.pyplot as plt
import seaborn as sns
geyser = sns.load_dataset("geyser")
tips=sns.load_dataset("tips")
#sns.displot(data=geyser, x="waiting", kind="kde", rug=True)
sns.scatterplot(data=tips, x="total_bill", y="tip")
sns.rugplot(data=tips, x="total_bill", y="tip")

plt.show()