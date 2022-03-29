#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''Simplest test for tensorflow

author:lileilei
email:hustlei@sina.cn
'''

import tensorflow as tf

#tensorflow1.0
'''
a=tf.constant(1)
b=tf.constant(2)
c=a+b

with tf.Session() as sess:
    c_val = sess.run(c)
    print(c_val)
'''

#tensorflow2.0

@tf.function  # 使用autograph构建静态图
def printStrs(x,y):
    z =  tf.strings.join([x,y],separator = " ")
    tf.print(z)

printStrs(tf.constant("hello"),tf.constant("world"))

