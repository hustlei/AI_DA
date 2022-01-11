import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# plotting

s=pd.Series([1,2,3,1,2,3,1,2,3])

pd.plotting.autocorrelation_plot(s)




plt.show()