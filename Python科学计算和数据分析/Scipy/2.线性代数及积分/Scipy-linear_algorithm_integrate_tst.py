#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
test code for scipy linear algorithm and integral
author:lileilei
email:hustlei@sina.cn
"""

################################
#Matrix method
################################
import numpy as np
import scipy.linalg
A=np.array([[1,-4],[-2,3]])
B=scipy.linalg.signm(A)
print(B)
print(np.sign(A))

print(scipy.linalg.funm(A,lambda x:x**2))