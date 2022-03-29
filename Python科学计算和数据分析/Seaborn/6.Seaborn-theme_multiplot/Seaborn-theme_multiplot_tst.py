#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
# set_theme, set
################################
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
sns.relplot(x=np.arange(5),y=np.arange(5)**2)
plt.show()
"""

################################
# set_theme, set
################################
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure()

with sns.axes_style("darkgrid"):
    ax = fig.add_subplot(231)
    ax.scatter(np.random.rand(10), np.random.rand(10))
    ax.set_title("darkgrid")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
with sns.axes_style("whitegrid"):
    ax = fig.add_subplot(232)
    ax.scatter(np.random.rand(10), np.random.rand(10))
    ax.set_title("whitegrid")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
with sns.axes_style("dark"):
    ax = fig.add_subplot(233)
    ax.scatter(np.random.rand(10), np.random.rand(10))
    ax.set_title("dark")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
with sns.axes_style("white"):
    ax = fig.add_subplot(234)
    ax.scatter(np.random.rand(10), np.random.rand(10))
    ax.set_title("white")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
with sns.axes_style("ticks"):
    ax = fig.add_subplot(235)
    ax.scatter(np.random.rand(10), np.random.rand(10))
    ax.set_title("ticks")
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))

plt.show()
"""


################################
# 设置颜色盘
################################
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



plt.show()
"""

################################
# jointplot
################################
"""
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("penguins")

# sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="scatter")
# sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, hue="species")
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="kde")
sns.jointplot(
    x="bill_length_mm", y="bill_depth_mm", data=data, kind="kde", hue="species"
)

sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="hist")
sns.jointplot(
    x="bill_length_mm", y="bill_depth_mm", data=data, kind="hist", hue="species"
)
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="hex")
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="reg")
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="resid")

plt.show()
"""

################################
# paireplot
################################

import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset("penguins")

sns.pairplot(data=data, hue="species")

plt.show()
