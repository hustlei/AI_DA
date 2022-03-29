人工智能导论(9)——自然语言处理(Natural Language Processing)
==============================

@[TOC]

<hr>

# 一、 概述

<font color=#888>人类利用语言进行交流、思想表达和文化传承是人类智能的重要体现。
自然语言处理(NLP)是指用计算机来处理、理解以及运用人类语言，实现人机交流的目的。

> <font color=#999>本文将人工智能"自然语言处理"基础知识整理为思维导图，便于回顾和记忆。

# 二、 重点内容

+ <font color=#888>简介
+ <font color=#888>NLP核心任务
+ <font color=#888>NLP主要应用领域
+ <font color=#888>NLP三个分析层面
+ <font color=#888>NLP分析流程
+ <font color=#888>典型应用简介
    - <font color=#888>语音识别
    - <font color=#888>机器翻译

# 三、 思维导图
![人工智能基础知识(9)——自然语言处理](https://img-blog.csdnimg.cn/def961c7e63c4bd2b3f0699cbcc3bcbd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




# 四、 重点知识笔记

<font color=#888>自然语言处理(NLP)是指用计算机来处理、理解以及运用人类语言，是计算机科学与语言学的交叉学科。

<font color=#888>从应用角度看，自然语言处理的应用有：

+ <font color=#888>语音识别、文字识别
+ <font color=#888>文本生成、图像描述生成
+ <font color=#888>机器翻译：将一种语言翻译成另一种语言。
+ <font color=#888>知识图谱、信息抽取
+ <font color=#888>对话系统、问答系统
+ <font color=#888>文本分类、情感分析
+ <font color=#888>文本推荐

## NLP的两个核心任务

+ <font color=#888>自然语言理解(NaturalLanguage Understanding, NLU)
+ <font color=#888>自然语言生成( Natural LanguageGeneration, NLG)

> + <font color=#888>自然语言理解(NLU):实现人机间自然语言通信意味着要使计算机既能理解自然语言文本的意义
> + <font color=#888>自然语言生成(NLG):以自然语言文本来表达给定的意图、思想等。

<font color=#888>NLU的应用：

+ <font color=#888>机器翻译
+ <font color=#888>机器客服
+ <font color=#888>智能音响

<font color=#888>NLU 的实现方式：

<font color=#888>自然语言理解跟整个人工智能的发展历史类似，一共经历了3次迭代：

+ <font color=#888>基于规则的方法：最早大家通过总结规律来判断自然语言的意图，常见的方法有：CFG、JSGF等。
+ <font color=#888>基于统计的方法：后来出现了基于统计学的 NLU 方式，常见的方法有：SVM、ME等。
+ <font color=#888>基于深度学习的方法：随着深度学习的爆发，CNN、RNN、LSTM 都成为了最新的”统治者”。

> <font color=#888>Transformer是目前“最先进”的方法。BERT 和 GPT-2都是基于TRansformer的。


<font color=#888>NLG 的6个步骤：

+ <font color=#888>内容确定 – Content Determination
+ <font color=#888>文本结构 – Text Structuring
+ <font color=#888>句子聚合 – Sentence Aggregation
+ <font color=#888>语法化 – Lexicalisation
+ <font color=#888>参考表达式生成 – Referring Expression Generation|REG
+ <font color=#888>语言实现 – Linguistic Realisation

<font color=#888>NLG典型应用：

+ <font color=#888>聊天机器人
+ <font color=#888>自动写新闻：对于具有明显规则的领域，比如体育新闻。目前很多新闻已经借助NLG 来完成了
+ <font color=#888>BI报告生成

# NLP处理的三个分析层面

**第一层面：词法分析**

<font color=#888>词法分析包括汉语的分词和词性标注这两部分。

+ <font color=#888>分词：将输人的文本切分为单独的词语
+ <font color=#888>词性标注：为每一个词赋予一个类别
    - <font color=#888>类别可以是名词(noun)、动词（verb）、形容词（adjective）等
    - <font color=#888>属于相同词性的词，在句法中承担类似的角色。

**第二层面:句法分析**

<font color=#888>句法分析是对输人的文本以句子为单位，进行分析以得到句子的句法结构的处理过程。

<font color=#888>三种比较主流的句法分析方法:

+ <font color=#888>短语结构句法体系，作用是识别出句子中的短语结构以及短语之间的层次句法关系;
    - <font color=#888>介于依存句法分析和深层文法句法分析之间
+ <font color=#888>依存结构句法体系（属于浅层句法分析），作用是识别句子中词与词之间的相互依赖关系;
    - <font color=#888>实现过程相对来说比较简单而且适合在多语言环境下应用，但是其所能提供的信息也相对较少
+ <font color=#888>深层文法句法分析，利用深层文法，对句子进行深层的句法以及语义分析
    - <font color=#888>例如词汇化树邻接文法，组合范畴文法等都是深层文法
    - <font color=#888>深层文法句法分析可以提供丰富的句法和语义信息
    - <font color=#888>深层文法相对比较复杂，分析器的运行复杂度也比较高，不太适合处理大规模的数据


**第三个层面:语义分析**

<font color=#888>语义分析的最终目的是理解句子表达的真是语义。语义表示形式至今没有一个统一的方案。

+ <font color=#888>语义角色标注（semantic role labeling)是目前比较成熟的浅层语义分析技术。
    - <font color=#888>语义角色标注一般都在句法分析的基础上完成，句法结构对于语义角色标注的性能至关重要。
    - <font color=#888>通常采用级联的方式，逐个模块分别训练模型
        - <font color=#888>分词
        - <font color=#888>词性标注
        - <font color=#888>句法分析
        - <font color=#888>语义分析
 + <font color=#888>联合模型（新发展的方法）
    - <font color=#888>将多个任务联合学习和解码
        + <font color=#888>分词词性联合
        + <font color=#888>词性句法联合
        + <font color=#888>分词词性句法联合
        + <font color=#888>句法语义联合等
    - <font color=#888>联合模型通常都可以显著提高分析质量
    - <font color=#888>联合模型的复杂度更高，速度也更慢。


## NLP的主要流程

+ <font color=#888>传统机器学习的 NLP 流程
    - <font color=#888>预处理
    - <font color=#888>特征提取
        + <font color=#888>词袋设计
        + <font color=#888>Embedding
    - <font color=#888>特征分类器
+ <font color=#888>深度学习的 NLP 流程
    - <font color=#888>预处理
    - <font color=#888>设计模型
    - <font color=#888>模型训练

<font color=#888>预处理过程：

+ <font color=#888>收集语料库（输入文本）
+ <font color=#888>文本清洗（文本清洗，）
    - <font color=#888>删除所有不相关的字符，例如任何非字母数字字符
    - <font color=#888>分割成单个的单词文本
    - <font color=#888>删除不相关的单词，例如“@”提及或网址链接
    - <font color=#888>将所有字符转换为小写，以便将诸如“hello”，“Hello”和“HELLO”之类的单词视为相同
    - <font color=#888>考虑将拼写错误或交替拼写的单词组合成单个表示（例如“cool”/“kewl”/“cooool”）
    - <font color=#888>考虑词性还原（将诸如“am”，“are”和“is”之类的词语简化为诸如“be”之类的常见形式）
+ <font color=#888>分词
+ <font color=#888>去掉停用词（可选）
+ <font color=#888>标准化和特征提取等。

<font color=#888>英文预处理：

+ <font color=#888>分词 – Tokenization
+ <font color=#888>词干提取 – Stemming
+ <font color=#888>词形还原 – Lemmatization
+ <font color=#888>词性标注 – Parts of Speech
+ <font color=#888>命名实体识别 – NER
+ <font color=#888>分块 – Chunking

<font color=#888>中文NLP预处理：

+ <font color=#888>中文分词 – Chinese Word Segmentation
+ <font color=#888>词性标注 – Parts of Speech
+ <font color=#888>命名实体识别 – NER
+ <font color=#888>去除停用词

<font color=#888>中文分词方法：

1. <font color=#888>经典的基于词典及人工规则：适应性不强，速度快，成本低
    + <font color=#888>基于词典：（正向、逆向、双向最大匹配）
    + <font color=#888>基于规则：（词法结构）
2. <font color=#888>现代的基于统计和机器学习：适应性强，速度较慢，成本较高
    + <font color=#888>基于统计：HMM隐马尔可夫模型
    + <font color=#888>基于机器学习：CRF条件随机场
    
    

**自然语言典型工具和平台**

+ <font color=#888>NLTK  ：全面的python基础NLP库。
+ <font color=#888>StanfordNLP ：学界常用的NLP算法库。
+ <font color=#888>中文NLP工具：THULAC、哈工大LTP、jieba分词。


## 语音识别

<font color=#888>将人类语音中的词汇内容转换为计算机可读的输入。

<font color=#888>语音识别系统的分类主要有：

+ <font color=#888>孤立和连续语音识别系统（主流为连续语音识别）
    - <font color=#888>以单字或单词为单位的孤立的语音识别系统
    - <font color=#888>自然语言只是在句尾或者文字需要加标点的地方有个间断，其他部分都是连续的发音
+ <font color=#888>特定人和非特定人语音识别系统
    - <font color=#888>特定人语音识别系统在前期需要大量的用户发音数据来训练模型。
    - <font color=#888>非特定人语音识别系统则在系统构建成功后，不需要事先进行大量语音数据训练就可以使用
+ <font color=#888>大词汇量和小词汇量语音识别系统
+ <font color=#888>嵌入式和服务器模式语音识别系统

**语音识别的过程**

<font color=#888>语音识别系统一般可以分为前端处理和后端处理两部分：

+ <font color=#888>前端包括
    - <font color=#888>语音信号的输入
    - <font color=#888>预处理：滤波、采样、量化
    - <font color=#888>特征提取
+ <font color=#888>后端是对数据库的搜索过程
    - <font color=#888>训练：对所建模型进行评估、匹配、优化，之后获得模型参数
    - <font color=#888>识别

<font color=#888>语音识别的过程：

+ <font color=#888>根据人的语音特点**建立语音模型**
+ <font color=#888>对输入的语音信号进行分析，并抽取所需的特征，建立语音识别所需要的**模板**
+ <font color=#888>将语音模板与输入的语音信号的特征进行比较，找出一与输入语音**匹配最佳的模板**
+ <font color=#888>通过查表和判决算法给出识别结果

<font color=#888>显然识别结果的准确率与语音特征的选择、语音模型和语音模板的好坏及准确度有关。

<font color=#888>语音识别系统的性能受多个因素的影响

+ <font color=#888>不同的说话人
+ <font color=#888>不同的语言
+ <font color=#888>同一种语言不同的发音和说话方式等

<font color=#888>提高系统的稳定性就是要提高系统克服这些因素的能力，使系统能够适应不同的环境。

<font color=#888>声学模型是识别系统的底层模型，并且是语音识别系统中最关键的一部分。
<font color=#888>声学模型的目的是提供一种有效的方法来计算语音的特征矢量序列和各发音模板之间的距离。

**语音识别关键技术**

+ <font color=#888>语音特征提取
    - <font color=#888>常见的语音特征提取算法有MFCC、FBank、LogFBank等
+ <font color=#888>声学模型与模式匹配
    - <font color=#888>声学模型：对应于语音音节频率的计算，输出计算得到的声学特征
    - <font color=#888>模式匹配：在识别时将输入的语音特征与声学特征同时进行匹配和比较
    - <font color=#888>目前采用的最广泛的建模技术是隐马尔可夫模型（Hidden Markov Model，HMM）。
+ <font color=#888>语音模型与语义理解
    - <font color=#888>进行语法、语义分析
    - <font color=#888>语言模型会计算音节到字的概率
        + <font color=#888>主要分为规则模型和统计模型
        + <font color=#888>语音模型的性能通常通过交叉熵和复杂度来表示，交叉熵表示
            - <font color=#888>交叉熵表示用该模型对文本进行识别的难度
            - <font color=#888>复杂度是指用该模型表示这个文本平均的分支数，其倒数可以看成是每个词的平均概率

## 机器翻译

<font color=#888>机器翻译就是让机器模拟人的翻译过程，利用计算机自动地将一种自然语言翻译为另一种自然语言。

<font color=#888>在机器翻译领域中出现了很多研究方法，包括：

+ <font color=#888>直接翻译方法
+ <font color=#888>句法转换方法
+ <font color=#888>中间语言方法
+ <font color=#888>基于规则的方法
+ <font color=#888>基于语料库的方法
+ <font color=#888>基于实例的方法（含模板与翻译记忆方法）
+ <font color=#888>基于统计的方法
+ <font color=#888>基于深度学习的方法等

<font color=#888>机器翻译过程：

+ <font color=#888>原文输入：按照一定的编码转换成二进制。
+ <font color=#888>原文分析（查词典和语法分析）
    - <font color=#888>查词典：词在语法类别上识别为单功能的词，在词义上成为单义词（某些介词和连词除外）
    - <font color=#888>语法分析：进一步明确某些词的形态特征。找出动词词组、名词词组、形容词词组等
+ <font color=#888>译文综合（调整词序与修辞以及从译文词典中取词）
    - <font color=#888>任务1：把应该以为的成分调动一下
        + <font color=#888>首先加工间接成分：从前向后依次取词加工，从句子的最外层向内加工。
        + <font color=#888>其次加工直接成分：依据成分取词加工，对于复句还需要对各分句进行加工。
    - <font color=#888>任务2：修辞加工
        + <font color=#888>根据修辞的要求增补或删掉一些词。例如英语中的冠词、数次翻译汉语，加上"个"、"只"。
    - <font color=#888>任务3：查目标语言词典，找出目标语言文字的代码。
+ 译文输出
    - <font color=#888>将目标语言的代码转换成文字，打印出译文来

<font color=#888>通用翻译模型：

+ <font color=#888>GNMT（Google NeuralMachine Translation）基于网页和App的神经网络机器翻译
+ <font color=#888>完全基于注意力机制的编解码器模型Transformer
+ <font color=#888>Transformer的升级版—Universal Transformer

> <font color=#888>在Transformer出现之前，多数神经基于神经网络的翻译模型都使用RNN。
> <font color=#888>RNN训练起来很慢，长句子很难训练好。
> <font color=#888>Universal Transformer模型具有了通用计算能力，在更多任务中取得了有力的结果。
> <font color=#888>Universal Transformer的训练和评估代码已开源在了Tensor2Tensor网站。

<br>
<hr>


> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>