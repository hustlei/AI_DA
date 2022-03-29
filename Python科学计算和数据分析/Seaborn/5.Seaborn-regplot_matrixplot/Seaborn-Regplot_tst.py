#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
# lmplot、regplot
################################
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.lmplot(data=tips, x="total_bill", y="tip")
# sns.lmplot(data=tips, x="total_bill", y="tip", order=3)
# sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex")
# sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex", col="smoker")
# sns.lmplot(data=tips, x="total_bill", y="tip", ci=50)
# sns.lmplot(data=tips, x="total_bill", y="tip", y_jitter=10)
# sns.lmplot(data=tips, x="total_bill", y="tip", x_estimator=np.mean, x_bins=4)

plt.show()
"""


################################
# lmplot、regplot
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns
ans = sns.load_dataset("anscombe")
dat = ans.loc[ans.dataset == "III"]

sns.lmplot(data=dat, x="x", y="y", robust=True, ci=None)

plt.show()
"""

################################
# lmplot、regplot
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns

tips = sns.load_dataset("tips")

# sns.lmplot(data=tips, x="total_bill", y="tip", scatter=False)
# sns.regplot(data=tips, x="total_bill", y="tip")
sns.residplot(data=tips, x="total_bill", y="tip")

plt.show()
"""


################################
# heatmap
################################
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.random.rand(10, 10)

# sns.heatmap(data=data)
# sns.heatmap(data=data, annot=True, fmt=".2f")
sns.heatmap(data=data, cmap="hsv", cbar=False, linewidths=0.5, linecolor="w")

plt.show()
"""

################################
# clustermap
################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = np.random.rand(10, 10)

sns.clustermap(data=data)

plt.show()
