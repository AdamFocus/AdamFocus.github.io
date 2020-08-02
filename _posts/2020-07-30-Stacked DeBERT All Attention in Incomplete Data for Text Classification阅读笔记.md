---
layout:     post
title:      Stacked DeBERT_All Attention in Incomplete Data for Text Classification
subtitle:   阅读笔记
date:       2020-07-30
author:     Adam
header-img: img/post-bg-unix-linux.jpg
catalog: true
tags:
    - NLP
    - TextClassification
---


> 本文首次发布于 [Adam's Blog](http://adamfocus.github.io), 作者 [@AdamFocus](http://github.com/AdamFocus) ,转载请保留原文链接.

## Stacked DeBERT: All Attention in Incomplete Data for Text Classification阅读笔记

### 任务分析

不完整数据问题通常认为是一个reconstruction或imputation任务

通常与missing number imputation有关

### 历史工作

针对缺失数据插补

1. Vincent提出将输入映射到有意义的表示上，从而重建出clean data
2. 其他：predictive mean matching, random forest, Support Vector Machine (SVM) and Multiple imputation by Chained Equations (MICE),

### 本文工作

面向tweets和语音转文本生成的句子

利用bert和去除噪声策略来解决不完全意图和情感分析

实现了Stacked Denoising BERT

obtaining richer input representations from input tokens by stacking denoising transformers on an embedding layer with vanilla transformers

（embedding层和vanilla transformer层用于获取输入的中间特征，denoising tranformers再从中提取出特征）

### Model

嵌入层	+	vanilla transformer层（conventional bidirectional transformers）+ denoising bidirectional transformers



![UHSF7F.jpg](https://s1.ax1x.com/2020/07/22/UHSF7F.jpg)

#### 1.常规Bert层

训练时在不完整的文本分类语料库上微调



- 预处理：小写化单词并标记、使用[CLS] [SEP]进行标记
- embedding层与bert相同：

#### 2.Denoising transformer：

- 多层感知机堆叠：

  设置成两组"三层"，每层有两个隐藏层

  **过程**：

  - 第一组：将$$h_{inc}$$压缩成潜在空间表示，提取特征转换成低维向量$$z_1,z_2,z$$

    分别为($$N_{bs}$$,128,128), (Nbs,32,128),(Nbs,12,128)维

    $$N_{bs}$$为batch size

    

  - 第二组：将$$z_1,z_2,z$$转换成$$h_{rec1},h_{rec2},h_{rec}$$

  - 将重建的向量$$h_{rec}$$与完整的向量$$h_{comp}$$借助均方差损失函数比较

    $$L(h_{rec},h_{comp})=\frac{1}{N_{bs}}\sum_{i=1}^n(h_{rec}-h_{comp})^2$$

  

  提取更加抽象和有意义的隐藏特征向量，来重建缺失的词嵌入

  

  训练在==sentence embedding==上进行，将不完全数据作为输入，将对应的完全数据$$h_{comp}$$作为target

  

   Both input and target are obtained after applying the embedding layers and the vanilla transformers, and have shape (Nbs,768,128), where Nbsis the batch size, 768 is the original BERT embedding size for a single token, and 128 is the maximum sequence length in a sentence.

  

- 双向Transformer：

  将上面生成的embedding输入该层，改进嵌入表示

#### 3.feedforward network/softmax 激活函数

### 使用的数据集

#### 1. Twitter Sentiment Classification

[Kaggle’s two-class Sentiment140 dataset][https://www.kaggle.com/kazanova/sentiment140]

错误类型：

|       mistake        |                           examples                           |
| :------------------: | :----------------------------------------------------------: |
|       spelling       |  “teh” (the), “correclty” (correctly), “teusday” (Tuesday)   |
| Casual pronunciation |           “wanna” (want to), “dunno” (don’t know)            |
|     Abbreviation     | “Lit” (Literature), “pls” (please), “u” (you), “idk” (I don’t know) |
|  Repeteated letters  |               thursdayyyyyy”, “sleeeeeeeeeep”                |
|     Onomatopoeia     |                   “Woohoo”, “hmmm”, “yaay”                   |
|        Others        |        “im” (I’m), “your/ur” (you’re), “ryt” (right)         |

使用人工标注正确信息

最终：

一共300条samples（只使用了250？）

- 训练：200sentences，100p，100n
- evaluate：50samples，25p，25n

#### 2.Chatbot Natural Language Unerstanding (NLU) Evaluation Corpus

Intent Classification from Text with STT Error

对拥有完整句子和意图标签的语料库进行TTS和STT处理，从而获取带有STT错误的不完整句子。



原始数据集：100train 106test



处理过程：

- TTS处理：使用gtts库[https://pypi.org/project/gTTS/]和macsay库[https://ss64.com/osx/say.html]

- STT处理：使用witai

  > chosen according to code availability and whether it’s freely available or has high daily usage limitations



不同处理方式在单词缺失和错误的比例不同

使用iBLEU来表示noise的程度，iBLEU在[0,1]取值

iBLEU=1-BLEU

BLEU是机器翻译任务中经常使用的指标

![](G:\MyBlog\AdamFocus.github.io\img\TTS-STT对比.jpg)

### 实验部分

#### baseline：

1. bert

2. NLU service platforms：[Google Dialogflow (formerly Api.ai)][https://dialogflow.com],[ SAP Conversational AI (formerly
   Recast.ai)][https://cai.tools.sap]and [Rasa (spacy and tensorflow backend)][https://rasa.com].

3. Semantic hashing with classifier

   Subword semantic hashing for intent classification on small datasets提到的一种word embedding方法，不会受到out-of-vocabulary的影响，使用在字母表的hash token而不是单个词语，使得vocabulary更加独立。

   分类器使用多层感知机、svm和随机森林

#### train

1. intent classification任务：

   一共训练三次，分别为完整数据、两种TTS-STT结合生成的数据

2. 情感分析任务：

   一共训练三次，分别为original text、corrected text、incorrect with correct texts

   选取十次中最好的F1值

3. Semantic hashing with classifier设置：

   采用3-gram，特征向量size设置为768

   13种分类器参数按照论文内容设置

4. bert设置：

   12 transformer block L

   hidden size H-768

   12 self-attention heads A

   3 epochs with Adam Optimizer

   learning rate of 2 ∗ 10−5

   maximum sequence length -128

   warm up proportion - 0.1

   train batch size is 4 for the Twitter Sentiment Corpus and 8 for the Chatbot Intent Classification Corpus

   

5. Stacked DeBert设置：

   trained in end-to-end manner

   training time depending on the size of the dataset and train batch size

   The stack of multilayer perceptrons are trained for 100 and 1,000 epochs with Adam Optimizer

   learning rate of 10−3,

   weight decay of 10−5,

   MSE loss criterion and batch size the same as BERT (4 for the Twitter Sentiment
   Corpus and 8 for the Chatbot Intent Classification Corpus).

### RESULT

准确率

F1值

混淆矩阵

### 未来工作

针对其他类型的噪声，比如单词重新排序，单词插入，拼写错误等等

使用其他网络代替前馈神经网络