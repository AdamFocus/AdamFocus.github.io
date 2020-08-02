---
layout:     post
title:      Pruning and Sparsemax Methods for Hierarchical Attention Networks
subtitle:   阅读笔记
date:       2020-08-02
author:     Adam
header-img: img/post-bg-unix-linux.jpg
catalog: true
tags:
    - NLP
    - TextClassification
---


> 本文首次发布于 [Adam's Blog](http://adamfocus.github.io), 作者 [@AdamFocus](http://github.com/AdamFocus) ,转载请保留原文链接.

## Pruning and Sparsemax Methods for Hierarchical Attention Networks阅读笔记

### 摘要

介绍并评价了两种层次注意力网络模型：

- Hierarchical Pruned Attention Networks：

  在分类过程中删除掉不相关的单词和句子，以减少文档分类中潜在的噪声

- Hierarchical Sparsemax Attention Networks：

  将注意力机制中的softmax替换成sparsemax，能够更好处理低概率的单词和句子的重要分布

### 任务分析

面向文档分类

### introduction

HAN层次注意力网络能够借助文档的结构和注意力机制更好地进行分类，但它不能区分文档中的弱关联的噪声信息。

作者将低分值的内容直接进行删除，提出了两种方法。

### Hierarchical Attention Networks

![](G:\MyBlog\AdamFocus.github.io\img\HAN.png)

给定一个flat padded document，形状为(S,L),S代表句子数量，L代表每句中单词数目

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=softmax(u^Tu)$$
5. 获得句子向量$$s=\alpha\bigodot h$$
6. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
7. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
8. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=softmax(u_s^Tu_s)$$
9. 获得document features $$v=\alpha\bigodot h_s$$
10. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
11. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度。



通过$$softmax(z):=\frac {exp(z)}{\sum_{z'} exp(z')}$$计算attention权重



### HPAN

给定一个阈值

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=softmax(u^Tu)$$
5. 低于阈值$$\alpha_{min}$$的attention权重都设置为0，然后对剩余的权重进行归一化，使其总和仍为0
6. 获得句子向量$$s=\alpha\bigodot h$$
7. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
8. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
9. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=softmax(u_s^Tu_s)$$
10. 低于阈值$$\alpha_{min}$$的attention权重都设置为0，然后对剩余的权重进行归一化，使其总和仍为0
11. 获得document features $$v=\alpha\bigodot h_s$$
12. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
13. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度

### HSAN

使用SparseMax函数代替softmax

通过returning the "Euclidean projection of the input vector z onto the probability simplex，能够更好地处理大量近零概率的分布

$$sparsemax(z):=argmin_{p∈∆^{k+1}}||p−z||^2$$

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=sparsemax(u^Tu)$$
5. 获得句子向量$$s=\alpha\bigodot h$$
6. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
7. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
8. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=sparsemax(u_s^Tu_s)$$
9. 获得document features $$v=\alpha\bigodot h_s$$
10. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
11. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度。

### Evaluation

| hyperparameter                | values | Yang's value |
| ----------------------------- | ------ | ------------ |
| Word2vec embeddings size      | 200    | 200          |
| GRU layers                    | 1      | 1            |
| GRU layers hidden sizes       | 50     | 50           |
| Dropout                       | 0.1    |              |
| Training epochs               | 3      |              |
| Optimizer                     | Adam   | SGD          |
| Learning rate                 | 0.001  |              |
| Batch size                    | 64     | 64           |
| HPAN min. attention threshold | 0.05   |              |

#### 数据集

IMDB(relatively low samplesize (which allows both faster loading and training) and its simplicity.)

#### Experimental Procedure

1. Train a model on the IMDB dataset for three epochs on the training set.
2. Evaluate, at each epoch, the classification accuracy on the validation set.
3. Evaluate, after all epochs, the classification accuracy on test dataset

metric：

M: Document classification accuracy (%) on obtained by a trained model on the test dataset, after three training epochs.

作者提出的两个模型效果都不如HAN原模型

### issue

HPAN在训练时使用cpu，如果使用GPU，在反向传播时会生成nanPruning and Sparsemax Methods for Hierarchical Attention Networks阅读笔记

### 摘要

介绍并评价了两种层次注意力网络模型：

- Hierarchical Pruned Attention Networks：

  在分类过程中删除掉不相关的单词和句子，以减少文档分类中潜在的噪声

- Hierarchical Sparsemax Attention Networks：

  将注意力机制中的softmax替换成sparsemax，能够更好处理低概率的单词和句子的重要分布

### 任务分析

面向文档分类

### introduction

HAN层次注意力网络能够借助文档的结构和注意力机制更好地进行分类，但它不能区分文档中的弱关联的噪声信息。

作者将低分值的内容直接进行删除，提出了两种方法。

### Hierarchical Attention Networks

![](G:\MyBlog\AdamFocus.github.io\img\HAN.png)

给定一个flat padded document，形状为(S,L),S代表句子数量，L代表每句中单词数目

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=softmax(u^Tu)$$
5. 获得句子向量$$s=\alpha\bigodot h$$
6. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
7. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
8. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=softmax(u_s^Tu_s)$$
9. 获得document features $$v=\alpha\bigodot h_s$$
10. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
11. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度。



通过$$softmax(z):=\frac {exp(z)}{\sum_{z'} exp(z')}$$计算attention权重



### HPAN

给定一个阈值

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=softmax(u^Tu)$$
5. 低于阈值$$\alpha_{min}$$的attention权重都设置为0，然后对剩余的权重进行归一化，使其总和仍为0
6. 获得句子向量$$s=\alpha\bigodot h$$
7. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
8. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
9. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=softmax(u_s^Tu_s)$$
10. 低于阈值$$\alpha_{min}$$的attention权重都设置为0，然后对剩余的权重进行归一化，使其总和仍为0
11. 获得document features $$v=\alpha\bigodot h_s$$
12. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
13. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度

### HSAN

使用SparseMax函数代替softmax

通过returning the "Euclidean projection of the input vector z onto the probability simplex，能够更好地处理大量近零概率的分布

$$sparsemax(z):=argmin_{p∈∆^{k+1}}||p−z||^2$$

步骤：

1. 进行word embedding，得到一个词嵌入表示w(S,L,E),E代表词嵌入的维数，设置为200，使用glove预训练向量。
2. w输入到一个双向的GRU中，计算hidden word scores $$h=[\overrightarrow h,\overleftarrow h]$$
3. 将h输入到单层前馈网络中，获得隐藏word表示 $$u=tanh(W_wh+b_w)$$
4. 将隐藏表示输入到注意力机制中，借助可训练的词级上下文向量$$u$$计算注意力权重$$\alpha=sparsemax(u^Tu)$$
5. 获得句子向量$$s=\alpha\bigodot h$$
6. 将s传入到双向GRU中，计算hidden sentence scores $$h_s=[\overrightarrow h_s,\overleftarrow h_s]$$
7. 将隐藏句子scores传入单层前馈网络中，获得隐藏句子表示$$u_s=tanh(W_sh_s+b_s)$$
8. 隐藏句子表示输入注意力机制中，使用可训练的句子级上下文向量$$u_s$$，计算注意力权重$$\alpha=sparsemax(u_s^Tu_s)$$
9. 获得document features $$v=\alpha\bigodot h_s$$
10. 将文档特征输入到affite transformation，获得class logit scores $$z=W_vv+b_v$$
11. 通过softmax函数计算最终的分类$$p=softmax(z)$$，通过交叉熵损失计算所有参数的梯度。

### Evaluation

| hyperparameter                | values | Yang's value |
| ----------------------------- | ------ | ------------ |
| Word2vec embeddings size      | 200    | 200          |
| GRU layers                    | 1      | 1            |
| GRU layers hidden sizes       | 50     | 50           |
| Dropout                       | 0.1    |              |
| Training epochs               | 3      |              |
| Optimizer                     | Adam   | SGD          |
| Learning rate                 | 0.001  |              |
| Batch size                    | 64     | 64           |
| HPAN min. attention threshold | 0.05   |              |

#### 数据集

IMDB(relatively low samplesize (which allows both faster loading and training) and its simplicity.)

#### Experimental Procedure

1. Train a model on the IMDB dataset for three epochs on the training set.
2. Evaluate, at each epoch, the classification accuracy on the validation set.
3. Evaluate, after all epochs, the classification accuracy on test dataset

metric：

M: Document classification accuracy (%) on obtained by a trained model on the test dataset, after three training epochs.

作者提出的两个模型效果都不如HAN原模型

### issue

HPAN在训练时使用cpu，如果使用GPU，在反向传播时会生成nan