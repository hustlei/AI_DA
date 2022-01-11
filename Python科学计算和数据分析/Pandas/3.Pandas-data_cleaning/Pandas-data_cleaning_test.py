#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

# data cleaning test examples

s1=pd.Series([1, 2, 3, 2, 1.5, 2])
s2=pd.Series([1, 2, np.nan, 2, 1.5, np.nan])
df=pd.DataFrame({'col1':s1, 'col2':s2, 'col3':s1})

## 空值判断

s1.isnull()
df.isnull()
df.isnull().any()
df.isnull().all().all()