Pandas系列(三)：数据清洗
==============================


@[TOC]

<hr>

# 一、 简介

<font color=#888>数据清洗主要是对空值、重复值、异常值进行处理；对数据标签进行整理；数据类型进行转换；以及数据标准化处理等。

> <font color=#999>Pandas系列将Pandas的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![Pandas 数据清洗]()


<br>

# 三、 Pandas数据清洗

## 1. 空值、缺失值处理

Pandas和Numpy中用np.nan表示空值。

> 本文示例代码中变量s表示Series类型，df表示DataFrame类型，d表示Series或DataFrame均可。

### 1.1 空值、缺失值判断

+ Series或DataFrame每个是否空值判断
    + `d.isnull()` 或 `pd.isnull(d)`：判断每个元素是否空值，返回bool类型的Series或DataFrame
    + `d.isna()` 或 `pd.isna(d)`：判断每个元素是否空值，返回bool类型的Series或DataFrame
+ Series或DataFrame每个是否非空值判断
    + `d.notnull()` 或 `pd.notnull(d)`：每个元素是否非空，返回bool类型的Series或DataFrame
    + `d.isna()` 或 `pd.notna(d)`: 每个元素是否非空，返回bool类型的Series或DataFrame

> + 用s.isnull().all()判断Series是否全部是空值
> + 用df.isnull().any()判断每列是否含空值
> + 用df.isnull().any().any()

### 1.2 空值处理

**填充空值**

+ 指定值填充（替换）空值
    + `d.fillna(0)`:用指定值填充所有空值
    + `d.fillna({'col1':0,'col2':1})`：不同的列指定不同的值填充空值
    + `d.fillna(np.zeros((3,3)), columns=['a','b','c'])`：用矩阵填充指定列中的空值
+ 用相邻的值填充（替换）空值
    + `d.pad()`：用空值前的值填充空值，`[nan,1,nan,2,nan,nan]`pad后变为`[nan,1,1,2,2,2]`
    + `d.bfill()` 或 `d.backfill()`：用NaN之后的数据填充空值，`[nan,nan,2,nan,3,nan]`变为`[2,2,2,3,3,nan]`
+ 插值填充（替换）
    + `d.interpolate()`：对每一列或者行进行插值填充空值
        + axis参数：默认为0，即对每一列插值
        + method参数：插值函数，默认为'linear'。
            - 可以用scipy.interpolate.interp1d中的方法（'nearest','zero','slinear','quadratic','cubic','spline','barycentric','polynomial'等 ）
            - ‘polynomial’ 和 ‘spline’ 需要指定阶数，即order参数(int类型)可以用df.interpolate(method='polynomial', order=5)指定
        + limit_direction参数：插值方向默认为'forward'（如果第一个值为空值则无法插值）。可以改为'backward'或者'both'

**移除空值**

+ `d.dropna()`：移除含空值的行或列
    + axis参数：移除行或列，默认为0，即行含有空值移除行
    + how参数：‘all'所有值为空移除，'any'默认值，包含空值移除
    + thresh参数：包含thresh个空值时移除
    + subset参数：axis轴上，指定需要处理的标签名称列表

## 2. 异常数据替换

异常数据可以修改替换。也可以替换为na，然后用缺失值方法处理

> 异常值指不正常的数据，比如年龄数据出现负值。

### 2.1 条件替换

+ `d.where(cond, rpl)`
    - `d.where(d>3)`：条件为真时，数据保持不变。条件为假替换为NaN。
    - `d.where(d['col1']>3, 3)`：条件为假时，用rpl替换。
    - `d.where(df[0]<3 & df[0] >2)`：多个条件组合。cond为布尔序列。通常用`s>0，df[0]<3 & df[0] >2`形式 
+ `d.mask(cond, rpl)`
    - 与where相反，条件为真时替换数据，条件为假时，数据不变

> `df1.where(df['a']>3,df2)`与`np.where(df['a']>3,df1,df2)`相似。 

### 2.2 replace函数替换

+ 常量替换
    + d.replace(2,3)：所有2替换为3
    + d.replace('a','ccc')：所有为'a'的元素替换为'ccc'
+ 列表替换
    + `df.replace([5,7,'bb'],[3,'bb','ccc'])`：列表中对应的数据替换。5替换为3,7替换为'bb'。
        - 数据类型不同时，替换后列dtype会变为object。不建议用不同数据类型的变量替换
+ 字典替换
    + `s.replace({2:1,3:2})`：把Series中的2替换为1，3替换为2
    + df.replace({'a':np.nan,'b':5},10)：把DataFrame中’a'列中的空值，'b'列中的数值5，都替换为10
    + df.replace({'a':{np.nan:10},'b':{5:20}})：把DataFrame中’a'列中的空值替换为10，'b'列中的数值5替换为20
+ 字符串正则表达式替换
    + `d.replace("[0-9]","abc",regex=True)`：符合正则表达式的字都替换为指定字符串。比如'123'会被替换为'abcabc3'
        + `df.replace(regex=r'[0-9]', value='abc')`：同上
    + `df.replace(["[0-9]",r'[a-b]'],['new1','new2'],regex=True)`：正则表达式列表，对应替换值列表
    + `df.replace({'A': r'^ba.$'}, {'A': 'new'}, regex=True)`：正则表达式字典，类似字典替换

## 3. 重复值处理

+ d.duplicated()：重复的**行**标记为True
    - 重复的n行中，默认第一行标记为False，其他行标记为True。 参数`keep=false`标记所有重复的行 
+ d.drop_duplicated()：删除重复的行。默认重复的行中保留第一行，可以用keep参数修改。
+ s.unique()：删除重复值，返回numpy数组。类似set()操作。DataFrame不支持该函数。

## 4. 字符串处理

+ 使用数据的str属性调用字符串处理函数
    + d.str.lower()
    + d.str.replace()
    + ...
+ 使用pandas数据的replace函数
    + d.replace()：见2.2节
+ 使用map和apply函数，对数据应用指定函数处理：d
    + d.map(str.strip)
    + ...

## 5. 标签修改

+ 修改标签名称(只能为列标签添加前缀或后缀）
    + d.add_prefix('col_')：把字符串添加到标签(DataFrame为列标签)的前面。数值标签操作后变为字符串
    + d.add_suffix('_col')：把字符串添加到标签后边，即给标签加后缀。
    + `df.set_axis([1,2,3],axis='index')`：与`df.index=[1,2,3]`相同
        + `df.index=[1,2,3]`：重命名index标签名称，列表长度必须与行数相同
    + `df.set_axis([1,2,3],axis='columns')`：列名称重新赋值。`df.columns`不支持直接用列表赋值
+ 标签重命名(行标签、列标签均可修改)
    + 直接替换标签名
        + d.rename({'a':'c1','b':'c2'})：把标签'a','b'修改为对应的值。默认修改行标签，axis='columns'参数可以改为列标签
        + d.rename(index={'a':'c1'},columns={0:2})：修改行标签和列标签
    + 用字符串函数处理标签名
        + df.rename(str.upper, axis='columns')：列标签都用指定的函数处理
        + df.rename(index=abs,columns=str.upper)：标签分别用指定的函数修改
    + 重置行标签为整数序号
        + df.reset_index()：重置行标签为序号。drop=True参数会丢失原来的index数据，默认会将原来的index转换为列。
+ 标签排序
    + `d.reindex()`：对所有列标签排序
    + `d.reidnex(['col1','col2'], axis="columns")`：对指定的列排序，输出只包含这两列的数据。
    + `d.reindex(['a','b'])`：对行标签排列，输出只包含这两行。
        - 假如参数中某个标签不存在，则创建行，用NaN填充，或者根据fill_value参数填充 
    + `d.reindex(index=[],columns=[])`：对列标签或行标签排序

## 6. 数据标准化

> 通常对指定列进行标准化处理，方便分析，通常标准化会把数据转换到0-1之间。

常见的标准化方法：

+ 离差标准化
+ 标准差标准化
+ 小数定标标准化

计算公式如下

+ 离差标准化

$$
x=\frac{x-x_{min}}{x_{max}-x_{min}}
$$

+ 标准差标准化

$$
x=\frac{x-x.mean}{x.std}
$$

+ 小数定标标准化

$$
x=\frac{x}{10^{ceil(log10(\max(|x|)))}}
$$

## 7. 数据转换操作

+ pd.to_numeric(s)：把序列或Series转为数值类型
    + 比如`[1,'10']`转换为`[1.0,10.0]`。
    + downcast参数可以指定具体的数据类型，参数值可以是‘integer’, ‘signed’, ‘unsigned’, ‘float’ 
+ pd.to_datetime(s)：把序列或Series转为时间类型。
    + format参数可以指定时间格式，比如"%Y%M%D"
    + unit参数可以指定单位，默认为'ns' 


<br>
<hr>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>