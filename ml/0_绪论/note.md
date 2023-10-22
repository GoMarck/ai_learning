# 绪论
本章节分为以下几点：
- 什么是机器学习
- 监督学习
- 无监督学习

什么是机器学习
---
机器学习有新旧两种定义，分别如下：
- Arthur Samuel (1959). Machine Learning: Field of study that gives computers the ablity to leran without being explicitly programmed.
- Tom Mitchell (1998). Well-posed Learning Problem: A computer program is said to leran from experience E with respect to some task T and some performance measure P, if its performace on T, as measured by P, improves with experience E.

以下棋为例：当使用机器学习的思路开发一个下棋程序时，会让程序自己与自己下几万次棋局，最终把程序训练成一个下棋高手。在这里面：
- E：程序与自己下几万次棋就是经验
- T：下棋就是这个程序的任务
- P：与新对手下棋时赢的概率

监督学习
---
监督学习的算法如下：
- Regression: Predict continuous valued output.
- Classificaction: Discrete valued output.

回归问题例子：给定一组房子大小和房价的关系数据，根据数据使用某种函数进行拟合，最终预测出某个大小的房子的房价。

分类问题例子：给出一个肿瘤样本的数据集，里面描述了肿瘤的特征：大小、平滑度、薄厚等，以及对应的类型：良性肿瘤和恶性肿瘤。通过对数据集进行学习之后，程序就能根据输入的数据推测该肿瘤是良性肿瘤还是恶性肿瘤。

无监督学习
---
相较于监督学习的数据集有明确的标签，无监督学习的最大特点就是没有数据标签。我们只有一个数据集，我们不知道他们要拿来做什么，也不知道它们每一个数据点究竟是什么，而无监督学习则是需要从它们当中找到某种数据结构。
- 聚类算法：给定一个数据集，找出当中的数据结构，将它们分为不同的簇（cluster）。
- 鸡尾酒会算法：给定一个数据集，找到当中的数据结构，将不同类别的数据分离出来。例如在鸡尾酒会上一个麦克风收录了多个人的声音以及背景音乐，通过鸡尾酒会算法，将不同的人声分离出来。
