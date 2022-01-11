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
print(s1+s2)

df1 = pd.DataFrame({'a':[1,2,3],'b':[4,5,6]})
df2 = pd.DataFrame({'b':[1,2,3],'c':[4,5,6]})

'''
>>> df1.to_pickle('a.pickle')
>>> pd.read_pickle('a.pickle')
   a  b
0  1  4
1  2  5
2  3  6
'''
'''
>>> df.to_csv('a.csv')
>>> pd.read_csv('a.csv',index_col=[0])
    a  b      c
0 NaN  5   True
1 NaN  7  False
2 NaN  9   True
'''

df1.to_excel('a.xslx')