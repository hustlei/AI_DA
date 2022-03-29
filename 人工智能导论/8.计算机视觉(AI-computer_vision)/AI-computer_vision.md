人工智能基础知识(8)——计算机视觉(Computer Vision)
==============================


@[TOC]

<hr>

# 一、 概述

<font color=#888>计算机视觉是智能感知的最重要技术。

> <font color=#999>为方便记忆和回顾，根据个人学习，总结人工智能基础知识和思维导图形成系列。

# 二、 重点内容

+ <font color=#888>计算机视觉简介
+ <font color=#888>主要计算机视觉技术

# 三、 思维导图
![人工智能基础知识(8)——计算机视觉](https://img-blog.csdnimg.cn/61ee044dd5b34eb5a7f62ab5d982fc58.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



# 四、 重点知识笔记

## 计算机视觉的主要流程

<font color=#888>人的大脑皮层，有差不多70%都是在处理视觉信息，是人类获取信息最主要的渠道。
<font color=#888>计算机视觉（Computer Vision，CV）是研究如何让计算机能够像人类那样“看”的技术。

> 机器视觉是面向应用的计算机视觉系统的设计与实现技术。机器视觉更偏重于产品生产、自动化等行业和工程应用

其基本过程为：

![计算机视觉基本流程](https://img-blog.csdnimg.cn/1ebe59b23aa6427b921300b706e35ec2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 主要计算机视觉技术

<font color=#888>目前计算机视觉主要应用在人脸识别、图像识别方面（包括静态、动态两类信息）。

+ <font color=#888>图像分类(image classification)
+ <font color=#888>目标定位(object localization)
+ <font color=#888>目标检测(object detection)（多个目标）
+ <font color=#888>目标跟踪(Object Tracking)
+ <font color=#888>图像分割(Image Segmentation)
+ <font color=#888>图像生成(Image Generation)
+ <font color=#888>人脸识别(Face Recognition)

> + <font color=#888>图像分类：解决“是什么？”的问题，即给定一张图片或一段视频判断里面包含什么类别的目标。
> + <font color=#888>目标定位：解决“在哪里？”的问题，即定位出这个目标的的位置。
> + <font color=#888>目标检测：解决“是什么？在哪里？”的问题，即定位出这个目标的的位置并且知道目标物是什么。用方框标记。
> + <font color=#888>目标分割：分为实例的分割（Instance-level）和场景分割（Scene-level），解决“每一个像素属于哪个目标物或场景”的问题。类似于抠图。


## 图像分类(image classification)

<font color=#888>图像分类，核心是从给定的分类集合中给图像分配一个标签的任务。也就是说给定一张输入图像，图像分类可以判断该图像中物体所属类别，即是否有猫，是否有狗。

<font color=#888>图像分类根据不同分类标准可以划分为很多种子方向。比如根据类别标签，可以划分为：

+ <font color=#888>二分类问题，比如判断图片中是否包含人脸；
+ <font color=#888>多分类问题，比如鸟类识别；
+ <font color=#888>多标签分类，每个类别都包含多种属性的标签，比如对于服饰分类，可以加上衣服颜色、纹理、袖长等标签。
    + <font color=#888>通用分类，比如简单划分为鸟类、车、猫、狗等类别；
    + <font color=#888>细粒度分类，目前图像分类比较热门的领域，比如鸟类、花卉、猫狗等类别，它们的一些更精细的类别之间非常相似，而同个类别则可能由于遮挡、角度、光照等原因就不易分辨。

<font color=#888>目前较为流行的图像分类架构是卷积神经网络（CNN）——将图像送入网络，然后网络对图像数据进行分类。

## 目标定位(object localization)（单个目标）

<font color=#888>在图像分类的基础上，我们还想知道图像中的目标具体在图像的什么位置。

<font color=#888>基本思路多任务学习，网络带有两个输出分支。

+ <font color=#888>一个分支用于做图像分类，和单纯图像分类区别在于还另外需要一个“背景”类。
+ <font color=#888>另一个分支用于判断目标位置，用方框标记。

<font color=#888>其基本思路是从卷积结果中找到一些较高响应的显著性区域，认为这个区域有对应图像中的目标。


## 目标检测(object detection)（多个目标）

<font color=#888>应用对象定位和特征点检测可以构建对象检测算法。通常可以对多个目标同时进行检测。

<font color=#888>对象检测即识别图像中的对象，常包含两方面的工作

+ <font color=#888>首先是找到目标
+ <font color=#888>然后就是识别目标。

<font color=#888>近年来，主要的目标检测算法已经转向更快、更高效的检测方法。主要有：

+ <font color=#888>Faster R-CNN
+ <font color=#888>基于区域的全卷积网络（R-FCN）算法
+ <font color=#888>You Only Look Once(YOLO)
+ <font color=#888>Single Shot MultiBox Detector(SSD)


## 目标跟踪（Object Tracking）

<font color=#888>目标跟踪，是指在特定场景跟踪某一个或多个特定感兴趣对象的过程。也就是通过动态视频和真实世界的交互，在检测到初始对象之后进行观察。比如无人驾驶领域的目标跟踪。

<font color=#888>根据观察模型，目标跟踪算法可分成2类：生成算法和判别算法。

+ <font color=#888>生成算法使用生成模型来描述表观特征，并将重建误差最小化来搜索目标，如主成分分析算法（PCA）；
+ <font color=#888>判别算法用来区分物体和背景，其性能更稳健，并逐渐成为跟踪对象的主要手段（判别算法也成为Tracking-by-Detection，深度学习也属于这一范畴）


## 图像分割(Image Segmentation)

<font color=#888>图像分割是基于图像检测的，它需要检测到目标物体，然后把物体分割出来。

<font color=#888>图像分割可以分为三种：

+ <font color=#888>普通分割：将不同分属于不同物体的像素区域分开，比如前景区域和后景区域的分割；
+ <font color=#888>语义分割：普通分割的基础上，在像素级别上的分类，属于同一类的像素都要被归为一类，比如分割出不同类别的物体；
+ <font color=#888>实例分割：语义分割的基础上，分割出每个实例物体，比如对图片中的多只狗都分割出来，识别出来它们是不同的个体，不仅仅是属于哪个类别。

典型算法：

+ U-Net，2015
+ DeepLab，2016
+ FCN，2016

**语义分割(semantic segmentation)**

<font color=#888>语义分割是目标检测更进阶的任务，目标检测只需要框出每个目标的包围盒，语义分割需要进一步判断图像中哪些像素属于哪个目标，相当于达到“抠图”的目的。


基本思路 目标检测+语义分割。

<font color=#888>先用目标检测方法将图像中的不同实例框出，再用语义分割方法在不同包围盒内进行逐像素标记。

<font color=#888>Mask R-CNN 用FPN进行目标检测，并通过添加额外分支进行语义分割(额外分割分支和原检测分支不共享参数)，即Mask R-CNN有三个输出分支(分类、坐标回归、和分割)。

## 图像生成(Image Generation)

<font color=#888>图像生成是根据一张图片生成修改部分区域的图片或者是全新的图片的任务。这个应用最近几年快速发展，主要原因也是由于 GANs 是最近几年非常热门的研究方向，而图像生成就是 GANs 的一大应用。


## 人脸识别(Face Recognition)


<font color=#888>人脸识别的过程中有4个关键的步骤：

+ <font color=#888>人脸检测：寻找图片中人脸的位置。标记并分割出来。
+ <font color=#888>人脸对齐：将不同角度的人脸图像对齐成同一种标准的形状。通过几何变换（仿射、旋转、缩放），使各个特征点对齐（将眼睛、嘴等部位移到相同位置）
+ <font color=#888>人脸编码：人脸图像的像素值会被转换成紧凑且可判别的特征向量。理想情况下，同一个主体的所有人脸都应该映射到相似的特征向量。
+ <font color=#888>人脸匹配：在人脸匹配构建模块中，两个特征向量会进行比较，从而得到一个相似度分数，该分数给出了两者属于同一个主体的可能性。

<font color=#888>这应该是计算机视觉方面最热门也是发展最成熟的应用，而且已经比较广泛的应用在各种安全、身份认证等，比如人脸支付、人脸解锁。

<br>
<hr>


> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，
> 持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
