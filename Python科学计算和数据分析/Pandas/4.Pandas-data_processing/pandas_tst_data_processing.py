#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""example for pandas basic methods

simple test for basic methods of pandas.

author:lei
email:hustlei@sina.cn
"""


import numpy as np
import pandas as pd



"""four fundational arithmetic of pandas"""
s1 = pd.Series([1,2,3],index=['a','b','c'])
s2 = pd.Series([1,3,5])

df1 = pd.DataFrame({'col1':[1,2,3],'col2':[4,5,6]})
df2 = pd.DataFrame({'col2':[1,2,3],'col3':[4,5,6]})

'''
>>> df
   col1  col2
0     1     4
1     2     5
2     3     6
>>> df.melt()
  variable  value
0     col1      1
1     col1      2
2     col1      3
3     col2      4
4     col2      5
5     col2      6
>>> _.pivot(columns='variable',values='value')
variable  col1  col2
0          1.0   NaN
1          2.0   NaN
2          3.0   NaN
3          NaN   4.0
4          NaN   5.0
5          NaN   6.0
'''

'''
>>> df1
   col1  col2
0     1     4
1     2     5
2     3     6
>>> df1.stack()
0  col1    1
   col2    4
1  col1    2
   col2    5
2  col1    3
   col2    6
dtype: int64
>>> _.unstack()
   col1  col2
0     1     4
1     2     5
2     3     6
'''


df3 = pd.DataFrame({'col1':[[1,2],[2,4],3],'col2':[4,5,6]})

'''
>>> df3
     col1  col2
0  [1, 2]     4
1  [2, 4]     5
2       3     6
>>> df3.explode('col1')
  col1  col2
0    1     4
0    2     4
1    2     5
1    4     5
2    3     6
'''

#排序

'''
>>> df2  
   0  1  
0  6  5  
1  5  8  
2  1  9  
3  6  5  
>>> df2.sort_values(0)
   0  1
2  1  9
1  5  8
0  6  5
3  6  5
>>> df2.sort_index()
   0  1
0  6  5
1  5  8
2  1  9
3  6  5
>>> df2.nlargest(1,columns=0)
   0  1
0  6  5
'''