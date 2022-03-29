Pandas系列(二)：IO和常用操作
==============================


@[TOC]

<hr>

# 一、 简介

<font color=#888>Pandas支持多种格式数据文件的读写，接口更加方便，比如：csv，json，excel，xml，sql等。

<font color=#888>Pandas中Series和DataFrame支持很多基本操作，包括：

+ <font color=#888>方便的数据查看工具，条件筛选等。
+ <font color=#888>常用运算，比如数学运算、布尔比较等。
+ <font color=#888>非数字类型操作，比如字符串、时间等，并且pandas还对字符串等处理进行了扩展。
+ <font color=#888>链式调用方法等。

> <font color=#999>Pandas系列将Pandas的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![Pandas IO和常用操作思维导图](https://img-blog.csdnimg.cn/59b26954813844c286d1773166d3f23b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



<br>

# 三、 Pandas IO和常用操作

## 1. IO
### 1.1 csv读写
```python
>>> df.to_csv('abc.csv', float_format="%.3f")
>>> pd.read_csv(u"abc.csv", na_values=['-','_'], index_col=0)
    a  b      c          d
0 NaN  5   True 2020-01-31
1 NaN  7  False 2020-02-29
2 NaN  9   True 2020-03-31
```

+ `df.to_csv(file)`：保存数据到csv文件
    + sep参数：数据分隔符，默认为逗号
    + na_rep参数：空值字符串，默认为''
    + float_format参数：浮点数格式，float_format="%.2f"
    + header参数：bool值，是否保存列标签
    + index参数：bool值，是否保存行标签
    + cols参数：列表参数，指定要保存的列
+ `df=pd.read_csv(file)`：读取csv数据
    + sep参数：指定数据分割符号，可以用正则表达式，默认为逗号
    + header参数：指定列标签所在行。默认第1行为列名。
    	+ header=0表示无列名。多标签用header=[0,1]形式参数
    + index_col参数：指定行标签，可以是数值、字符串、False。默认为None
    + skiprows参数：如果数据文件包含一些说明行，可以用该参数指定数据的开始行号。
    + skipfooter参数：忽略最后几行。
    + na_values, true_values, false_values参数：分别指定NaN、True和False对应的字符串或字符串列表。
    	+ na_values默认值包含了‘’, ‘#N/A’, ‘#N/A N/A’, ‘#NA’, ‘-1.#IND’, ‘-1.#QNAN’, ‘-NaN’, ‘-nan’, ‘1.#IND’, ‘1.#QNAN’, ‘<NA>’, ‘N/A’, ‘NA’, ‘NULL’, ‘NaN’, ‘n/a’, ‘nan’, ‘null’.
    + encoding参数：指定文件的编码，如'utf-8','utf-8-sig','gbk'等
    + usecols参数：指定需要读入的列，其他列暂时不读
    + chunksize参数：指定一次读入的行数。对很大的文件有用
    + parse_dates参数：指定数据类型为时间的列


### 1.2 excel读写

```python
>>> df.to_excel('a.xlsx')   #后缀名不能写错了
>>> pd.read_excel('a.xlsx', index_col=0)
    a  b      c          d
0 NaN  5   True 2020-01-31
1 NaN  7  False 2020-02-29
2 NaN  9   True 2020-03-31
```
> 要读写'xsl'格式excel文件，需要安装xlwt，xlrd库。
> 要读写'xslx'格式excel文件，需要安装openpyxl库。


+ `df.to_excel(file)`：保存excel文件
	+ sheet_name参数：excel标签名称，字符串，默认为Sheet1
    + na_rep参数：空值字符串，默认为''
    + float_format参数：浮点数格式，float_format="%.2f"
    + header参数：bool值，是否保存列标签
    + index参数：bool值，是否保存行标签
+ `df=pd.read_excel(file)`:读取excel文件
    + sheet_name：excel标签名称或序号，可以指定多个，如sheet_name=[0,'sheet1']
    + header参数：指定列标签所在行。默认第1行为列名。
    + index_col参数：指定行标签，可以是数值、字符串、False。默认为None
    + names参数：如果header=None，没有列标签，可以用names指定
    + skiprows参数：如果数据文件包含一些说明行，可以用该参数指定数据的开始行号。
    + na_values, true_values, false_values参数：分别指定NaN、True和False对应的字符串或字符串列表。
    + usecols参数：指定需要读入的列，其他列暂时不读
    + parse_dates参数：指定数据类型为时间的列

### 1.3 json读写
```python
>>> df.to_json('a.json')
>>> pd.read_json('a.json')
    a  b      c              d
0 NaN  5   True  1580428800000
1 NaN  7  False  1582934400000
2 NaN  9   True  1585612800000
```

+ `df.to_json(file)`：保存到json文件
    + double_precision参数：浮点数小数位数
    + force_ascii参数：是否强制字符串编码为ASCII，默认为True。
    + data_unit参数：字符串参数。时间单位，默认为'ms'
+ `df=pd.read_json(file)`：读取json文件
    + encoding参数：文件编码，字符串参数，默认为'utf-8'
    + typ参数：存储的数据类型‘frame'或者'series'，默认是frame


4.1.4 xml

```python
>>> df.to_xml('a.xml')
>>> pd.read_xml('a.xml')  #无法指定index所在列
   index   a  b      c                    d
0      0 NaN  5   True  2020-01-31 00:00:00
1      1 NaN  7  False  2020-02-29 00:00:00
2      2 NaN  9   True  2020-03-31 00:00:00
```

> 需要安装lxml。

+ `df.to_xml(file)`：保存到xml文件
    + encoding参数：文件编码，字符串参数，默认为'utf-8'
    + index参数：bool值，是否保存行标签
    + na_rep参数：空值字符串，默认为''
+ `df=pd.read_xml(file)`：从xml文件读取数据
    + encoding参数：文件编码，字符串参数，默认为'utf-8'
    + xpath参数：要读取的xpath。默认为`./*`


### 1.5 Pickling读写

```python
>>> df.to_pickle('a.dat')
>>> pd.read_pickle('a.dat')
   a  b      c                    d
0  NaN  5   True  2020-01-31 00:00:00
1  NaN  7  False  2020-02-29 00:00:00
2  NaN  9   True  2020-03-31 00:00:00
```

+ `df.to_pickle(file)`:对象进行序列化，保存到文件
+ `df=pd.read_pickle(file)`:读取序列化数据，反序列化

### 1.6 SQL读写
```python
>>> con=sqlite3.connect('sqlite.db')
>>> df.to_sql('tab',con=con)
>>> pd.read_sql("select * from tab", con=con, index_col='index')
          a  b  c                    d
index
0      None  5  1  2020-01-31 00:00:00
1      None  7  0  2020-02-29 00:00:00
2      None  9  1  2020-03-31 00:00:00
>>> con.close()
```

+ `df.to_sql('tab',con=con)`:保存数据到数据库中新建的表中，数据库中如果已有'tab'表会出错
    + name参数：数据库Table名称，比如'table'
    + con参数：数据库连接参数
    + index参数：是否保存行标签
    + index_label：行标签的列名，默认为'index'
+ `df=pd.read_sql("select * from tab",con)`：根据sql语句从数据库读取数据
    + index_col参数：指定行标签的列名。如果用默认的to_sql，read_sql可以取"index"
    + params参数：sql语句中变量参数

### 1.7 html读写

```python
>>> df.to_html()
<table border="1" class="dataframe">
<thead><tr><th></th><th>a</th><th>b</th><th>c</th><th>d</th></tr></thead>
<tbody><tr><th>0</th><td>NaN</td><td>5</td><td>True</td><td>2020-01-31</td></tr>
<tr><th>1</th><td>NaN</td><td>7</td><td>False</td><td>2020-02-29</td></tr>
<tr><th>2</th><td>NaN</td><td>9</td><td>True</td><td>2020-03-31</td></tr>
</tbody></table>
>>> pd.read_html(_,index_col=0)[0]
    a  b      c           d
0 NaN  5   True  2020-01-31
1 NaN  7  False  2020-02-29
2 NaN  9   True  2020-03-31
```

> 需要安装lxml

+ `df.to_html(file)`：保存数据为html表格
    + na_rep参数：空值字符串，默认为'NaN'
    + float_format参数：浮点数格式，float_format="%.2f"
    + header参数：bool值，是否保存列标签
    + index参数：bool值，是否保存行标签
    + col_space参数：html表格单元格宽度。字符串或数值参数。数值默认单位为px
    + justify参数：表头(列标签)对齐方式，比如'left', 'right', 'center', 'justify'等
+ `df=pd.read_html(file)`：从html表格读取数据。返回元素为DataFrame的list
	+ header参数：指定列标签所在行。默认第1行为列名。
    + index_col参数：指定行标签，可以是数值或列表。默认为None
	+ skiprows参数：如果数据文件包含一些说明行，可以用该参数指定数据的开始行号。
	+ parse_dates参数：指定数据类型为时间的列

### 1.8 Latex读写

```python
>>> df.to_latex()
\begin{tabular}{lrrll}
\toprule
{} &   a &  b &      c &          d \\
\midrule
0 & NaN &  5 &   True & 2020-01-31 \\
1 & NaN &  7 &  False & 2020-02-29 \\
2 & NaN &  9 &   True & 2020-03-31 \\
\bottomrule
\end{tabular}
```

+ `df.to_latex(file)`：保存为latex文件。输出的latex是一个tabular。
    + encoding参数：文件编码，字符串参数，默认为'utf-8'
    + index参数：bool值，是否保存行标签
    + na_rep参数：空值字符串，默认为'NaN'
    + float_format参数：浮点数格式，float_format="%.2f"

> 如果要把输出的latex文件转换为pdf或图片，需要引用booktabs包，即增加`\usepackage{booktabs}`


> pandas不支持读取latex文件。但是可以用astropy库读取latex格式文件。
> ```
> from astropy.table import Table
> tab = Table.read('table.tex').to_pandas()
> ```

## 2 查看数据
### 2.1 数据检查

> 常用的数据查看函数Series, DataFrame基本相同

+ d.info()：可以查看DataFrame的信息(列，类型，内存等)
+ d.head(), d.head(n)：返回前5行或前n行数据
+ d.tail(), d.tail(n)：返回后5行或后n行数据
+ d.sample(), d.sample(10)：随机返回1行或n行数据
+ d.idxmax(), d.idxmin()：返回最大最小值的索引。
    - dataframe默认是index轴的最大最小，可以通过axis参数设置为columns

### 2.2 时间过滤

> 适用于Series(必须是 DatetimeIndex)和DataFrame（必须是使用 DatetimeIndex做为index的）。

+ d.first('3H')：获取最早3小时的数据，DataFrame为行数据
+ d.last('3D')：获取最后3天的数据，DataFrame为行数据
+ d.at_time('12:00')：指定每天中的时间，获取数据
+ d.between_time('8:00', '11:00')：根据开始和结束时间点，指定每天的时间段参数可以是datatime.time或者字符串

### 2.3 简单判断

+ in 运算符：数据是否在Series或DataFrame中
+ isin([0,1])：判断参数是否在数据中。参数必须是序列(list,Series或dict,dataframe等)
+ notin(values)：isin的非

### 2.4 筛选过滤

+ **df.query("A>`B a1` and A < @x")**：根据布尔运算字符串，查询。
	+ 其中A为列名`B a1`在名称有空格时可以使用，@x表示引用上下文代码中的变量
	+ 在查询大量数据时速度更快
+ `filter()`:根据行或列标签过滤
    + `df.filter(['a','b'])`：根据数组中的名称过滤
    + `df.filter(like='a')`：查找标签中包含指定字符串的行
    + `df.filter(regex='[a-z]*')`：根据正则表达式过滤标签名称
    + `df.filter(..., axis=0)`：根据行标签过滤，默认是根据列标签

## 3 基本运算函数

### 3.1 pandas广播原则

+ 与常量运算
    - 类似于numpy广播，即每个元素与常量运算
+ 与列表、数组运算  
    - DataFrame每行分别与序列进行运算。序列长度必须与Series或者DataFrame的行长度相同。
    - axis参数可以改为列运算，axis默认为'columns'，即沿1轴运算
+ DataFrame与DataFrame运算     
    - 列名相同的列运算。列长度不同用填充NaN补齐；只有一个数据有的列，另一个数据用NaN填充
+ Series与Series运算
    - 相同标签的元素运算。对应标签无数据的，用NaN代替
+ DataFrame与Series运算    
    - 对DataFrame的每一行，列标签与Series标签相同的元素进行运算。
    - axis参数可调整为列运算，axis默认为'columns'，即沿1轴运算

### 3.2 数学运算

**3.2.1 算术运算**

```python
-d
d + 2
d1 + d2
2 ** d
np.sqare(d)
df['a'] + df['b']
```

+ `+, -, *, /, //, %, **`都可以进行运算，很多numpy函数也都可以。
+ `add(), sub(), mul(), div()`及`radd(), rsub()`等方法。a.rdiv(b)表示b/a

> 除了`+*`外，都只支持数值类型。如果包含字符串等会出错

**3.2.2 汇总运算**

> 多个值运算输出一个值。即reduce操作，中文常翻译为归约、归纳、化简、缩减、折叠(fold)、汇总、合并等。

+ count()：统计数据中非空项个数
+ sum()：求和
+ max()：求最大值
+ min()：求最小值

**3.2.3 方法参数**

+ axis    
    - Series沿指定的轴与DataFrame运算，axis默认为'columns'，即每行与Series运算
    - 比如：df.add(s, axis=1)
+ fill_value
    - 广播时，用fill_value代替NaN填充不存在的数据，然后再运算
    - 比如：df1.add(df2,fill_value=1)

### 3.3 布尔运算

**3.3.1 逻辑对比**

```
d > 2
d1 < d2
d == [1,2,3]
```

+ `>, <, ==, !=, >=, <=`都可以，但是必须数据类型先相同。
    - 如df中有数值列，则与字符串比较则会出错
+ `eq(等于), ne(不等于), lt(小于), gt(大于), le(小于等于), ge(大于等于)`等函数都适用

**3.3.2 汇总运算**

+ all()：所有值为真输出真
+ any()：任意值为真输出真

**3.3.3 方法参数**

+ axis
    + Series沿指定的轴与DataFrame运算，axis默认为'columns'，即每行与Series运算
    + 比如：df.le(s, axis=1) 

## 4 非数值数据处理
### 4.1 字符串方法(Series)

**4.1.1 python字符串函数**

> 对于数据类型为字符串的Series或者DataFrame的列，可以通过str属性调用字符串函数。

+ 合并和分割
    + s.str.cat()：合并字符串
    + s.tr.split()：根据指定的分隔符，分割字符串
        - rsplit()：从字符串尾部开始分割字符串。不指定分割次数count时，与split相同 
    + s.str.join(sep)：对每个字符串中的字符用sep分隔符组合
    + s.str.partition()：同str.partition
    + s.str.rpartition()：同str.rpartition
+ 格式化
    + s.str.center(n)：填充空格到长度为n，居中对齐
    + s.str.ljust(n)：填充空格到长度为n，左对齐
    + s.str.rjust(n)：填充空格到长度为n，右对齐
    + s.str.zfill(n)：左侧填充0到长度为n
+ 修剪
    + s.str.strip()：同str.strip
    + s.str.rstrip()：同str.rstrip
    + s.str.lstrip()：同str.lstrip
+ 判断
    + s.str.contains(pat)：是否包含字符串，或者正则表达式字符串表示的内容
    + s.str.startswith(str)：是否以某个字符串开头。str不能是正则表达式
    + s.str.endswith(str)：是否以某个字符串结尾。str不能是正则表达式
    + s.str.isalnum()：同str.isalnum
    + s.str.isalpha()：同str.isalpha
    + s.str.isdigit()：同str.isdigit
    + s.str.isspace()：同str.isspace
    + s.str.islower()：同str.islower
    + s.str.isupper()：同str.isupper
    + s.str.istitle()：同str.istitle
    + s.str.isnumeric()：同str.isnumeric
    + s.str.isdecimal()：同str.isdecimal
+ 统计
    + s.str.count(str)：统计出现str的次数。
    + s.str.len()：字符串长度
+ 大小写
    + s.str.lower()：同str.lower
    + s.str.casefold()：同str.casefold
    + s.str.upper()：同str.upper
    + s.str.capitalize()：同str.capitalize
    + s.str.swapcase()：同str.swapcase
+ 检索
    + s.str.find()：同str.find
    + s.str.rfind()：同str.rfind
    + s.str.index()：同str.index
    + s.str.rindex()：同str.rindex
+ 编码
    + s.str.normalize()：同unicodedata.normalize
    + s.str.translate()：同str.translate

**4.1.2 正则表达式支持**

+ s.str.match(pat)：为每个字符串执行正则表达式re.match操作
+ s.str.findall(pat)：为每个字符串执行正则表达式re.findall操作
+ s.str.extract(pat)：为每个字符串执行正则表达式re.search操作
+ s.str.extractall(pat)：将re.findall得到的group，作为一个列和s共同组成一个DataFrame
+ s.str.replace(pat,repl)：替换字符串，或者正则表达式字符串指定的字符串

**4.1.3 pandas扩展支持**

> pandas独有的python字符串不支持的方法

+ `s.str[]`：下标或切片访问每个字符串
    + s.str.slice(start=None,end=None,step=None)：获取字符串切片
    + s.str.slice_replace(start=None,end=None,repl=None)：用repl替换切片内容
    + s.str.get(i)：获取第i个字母
+ s.str.repeat(n)：字符重复n次
+ s.str.pad()：在字符串左侧、右侧或者两侧填充空格


### 4.2 时间数据方法(Series)

> 类似字符串，通过dt访问属性或方法（根据数据类型，时间戳、时间段、时间间隔等的属性和方法都可以访问）

比如：

```python
s.dt.date
s.dt.time
s.dt.days
s.dt.to_pydatetime()
```

## 5 方法链

pandas的函数返回值通常还是pandas的数据结构，因此可以通过链式调用实现多个方法的依次处理

比如：`df.sample(10).query(...).where(...).add(2).sum()`



<br>
<hr>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>