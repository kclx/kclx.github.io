---
title: DecisionTree算法简析
tags: [ 'ML', 'Decision Tree']
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: DecisionTree算法简析
swiper: false
swiperDesc: DecisionTree算法简析
tocOpen: true
onlyTitle: false
share: true
copyright: true
donate: true
bgImgTransition: fade
bgImgDelay: 180000
prismjs: default
mathjax: ture
imgTop: ture
date: 2023-07-29 10:25:32
updated: 2023-07-29 10:25:32
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/07/29/2023-07-29-13.35.04.png
---

# DecisionTree算法简析

决策树（Decision Tree）是一种常见的机器学习算法，它可用于分类和回归任务。决策树通过对数据进行一系列逻辑判断，构建一个树状结构来进行决策。在分类问题中，决策树用于预测输入数据所属的类别；在回归问题中，它用于预测连续型的输出值。

决策树的构建过程基于训练数据，通过选择最佳的特征来进行节点分裂，直到达到某个停止条件。构建好的决策树能够提供清晰的决策路径，易于解释，并且可以很好地处理非线性关系。

以下是决策树算法的简要分析：

1. 特征选择：决策树的核心是选择最佳的特征来进行节点的分裂。通常采用信息增益、信息增益比、基尼不纯度等指标来评估特征的重要性，选择能够最好地分隔数据的特征。

2. 节点分裂：决策树根据特征的取值将数据划分为不同的子集。分裂过程持续进行，直到达到停止条件，例如达到最大树深度、子集中的样本数量过少等。

3. 树的构建：决策树的构建是一个递归的过程，从根节点开始，根据特征选择和节点分裂的原则逐步构建分支，直到生成一个完整的决策树。

4. 剪枝：构建好的决策树可能存在过拟合问题（对训练数据过度拟合）。剪枝是一种通过去除一些不重要的节点来减少复杂度、提高泛化能力的技术。

5. 预测：用新的数据样本在决策树上进行遍历，根据叶节点的类别（分类问题）或预测值（回归问题）进行预测。

决策树算法的优点包括易于解释、处理非线性关系和对缺失值不敏感。然而，它也有一些缺点，如容易过拟合、对噪声敏感等。为了改进决策树算法，人们发展了一些变种，如随机森林（Random Forest）和梯度提升决策树（Gradient Boosting Decision Tree），以提高模型的性能和鲁棒性。

总的来说，决策树是一种强大的算法，特别适用于解决分类和回归问题，同时也是学习机器学习算法基础的重要内容之一。

### 设计到的数学公式

当涉及到决策树及其相关算法时，可能会使用以下公式。下面是这些公式的使用示例，使用MathJax进行表示：

1. **信息熵（Entropy）**：衡量数据集的混乱程度，对于分类问题的特征选择很重要。

    $$
    H(X) = -\sum_{i=1}^{n} p(x_i) \log(p(x_i))
    $$

    其中，$H(X)$表示数据集$X$的信息熵，$p(x_i)$是数据集$X$中类别为$x_i$的样本占比。

2. **信息增益（Information Gain）**：表示使用某个特征对数据集进行划分所获得的信息熵减少的程度，用于特征选择。

    $$
    \text{Information Gain}(X, \text{feature}) = H(X) - \sum_{v \in \text{values}(\text{feature})} \frac{|X_v|}{|X|} H(X_v)
    $$
    
    其中，$H(X)$表示数据集$X$的信息熵，$X_v$表示使用特征$\text{feature}$中取值为$v$的样本子集，$\text{values}(\text{feature})$表示特征$\text{feature}$的所有取值。

3. **基尼不纯度（Gini Impurity）**：另一种衡量数据集混乱程度的方法，用于特征选择。

    $$
    \text{Gini}(X) = 1 - \sum_{i=1}^{n} p(x_i)^2
    $$
    
    其中，$\text{Gini}(X)$表示数据集$X$的基尼不纯度，$p(x_i)$是数据集$X$中类别为$x_i$的样本占比。

4. **回归树的均方误差（Mean Squared Error，MSE）**：回归树使用的损失函数，用于衡量预测值与实际值之间的差异。
    $$
    \text{MSE}(X) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
    $$

    其中，$\text{MSE}(X)$表示数据集$X$的均方误差，$n$是样本数量，$y_i$是第$i$个样本的实际输出值，$\bar{y}$是样本输出值的均值。

5. **决策树的预测**：对于回归树，预测值为叶节点上训练样本输出值的平均值；对于分类树，预测值为叶节点上出现最频繁的类别。

### 构建步骤

1. **数据准备**：
    - 收集数据集：获取用于训练和测试的数据集。
    - 数据清洗：处理缺失值、异常值等数据问题，确保数据质量。
    - 数据划分：将数据集划分为训练集和测试集，用于模型的训练和评估。

2. **特征选择**：
    - 根据具体任务和特征的性质，选择合适的特征用于构建决策树。
    - 使用信息增益、基尼不纯度等指标评估特征的重要性，选择最佳的划分特征。

3. **构建决策树**：
    - 从根节点开始，根据选定的特征选择标准进行节点分裂，生成子节点。
    - 递归地对子节点进行分裂，直到满足停止条件，例如达到最大深度或节点样本数量过少。

4. **剪枝**（可选）：
    - 构建好的决策树可能对训练数据过拟合，剪枝是一种降低过拟合风险的技术。
    - 可以采用预剪枝（在构建树的过程中进行剪枝）或后剪枝（构建完整树后再进行剪枝）。

5. **预测**：
    - 使用训练好的决策树对新数据进行预测。
    - 对于分类问题，根据决策树的分支和叶节点的类别进行分类预测。
    - 对于回归问题，根据决策树的叶节点上训练样本输出值的平均值进行回归预测。

6. **模型评估**：
    - 使用测试集对训练好的决策树进行性能评估，可以使用准确率（分类问题）或均方误差（回归问题）等指标。

7. **优化**：
    - 根据模型评估结果，可以调整决策树的参数，选择不同的特征，或尝试其他优化方法，以提高模型性能。

8. **应用**：
    - 将训练好的决策树应用于实际问题中，进行分类、回归等预测任务。

### 示例

```python
import operator
from math import log

from matplotlib import pyplot as plt

import decisionTreePlot as dtPlot


class DecisionTree:

    def __init__(self):
        self.myTree = None

    def calcShannonEnt(self, dataSet):
        """
        计算给定数据集的香农熵
        """
        numEntries = len(dataSet)
        labelCounts = {}
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1

        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)

        return shannonEnt

    def splitDataSet(self, dataSet, index, value):
        """
        划分数据集
        """
        retDataSet = []
        for featVec in dataSet:
            if featVec[index] == value:
                reducedFeatVec = featVec[:index]
                reducedFeatVec.extend(featVec[index + 1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        """
        选择切分数据集的最佳特征
        """
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain, bestFeature = 0.0, -1

        for i in range(numFeatures):
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0

            for value in uniqueVals:
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)

            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeature = i

        return bestFeature

    def majorityCnt(self, classList):
        """
        选择出现次数最多的一个结果
        """
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1

        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        """
        创建决策树
        """
        classList = [example[-1] for example in dataSet]

        if classList.count(classList[0]) == len(classList):
            return classList[0]

        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList)

        bestFeat = self.chooseBestFeatureToSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel: {}}

        del (labels[bestFeat])

        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)

        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)

        self.myTree = myTree
        return self.myTree

    def classify(self, inputTree, featLabels, testVec):
        """
        对新数据进行分类
        """
        firstStr = list(inputTree.keys())[0]
        secondDict = inputTree[firstStr]
        featIndex = featLabels.index(firstStr)
        key = testVec[featIndex]
        valueOfFeat = secondDict[key]

        if isinstance(valueOfFeat, dict):
            classLabel = self.classify(valueOfFeat, featLabels, testVec)
        else:
            classLabel = valueOfFeat

        return classLabel

    def storeTree(self, inputTree, filename):
        """
        将训练好的决策树模型存储起来，使用 pickle 模块
        """
        import pickle
        with open(filename, 'wb') as fw:
            pickle.dump(inputTree, fw)

    def grabTree(self, filename):
        """
        将之前存储的决策树模型使用 pickle 模块还原出来
        """
        import pickle
        with open(filename, 'rb') as fr:
            return pickle.load(fr)

    def ContactLensesTest(self):
        """
        预测隐形眼镜的测试代码，并将结果画出来
        """
        print(self.myTree)
        self.createPlot()

    def createPlot(self):
        fig = plt.figure(1, facecolor='green')
        fig.clf()

        axprops = dict(xticks=[], yticks=[])
        dtPlot.createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)

        dtPlot.plotTree.totalW = float(dtPlot.getNumLeafs(self.myTree))
        dtPlot.plotTree.totalD = float(dtPlot.getTreeDepth(self.myTree))
        dtPlot.plotTree.xOff = -0.5 / dtPlot.plotTree.totalW
        dtPlot.plotTree.yOff = 1.0
        dtPlot.plotTree(self.myTree, (0.5, 1.0), '')
        plt.show()


if __name__ == "__main__":
    fr = open('3.DecisionTree/lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    dt = DecisionTree()
    dt.createTree(lenses, lensesLabels)
    print(dt.myTree)
    dt.createPlot()
```

示例数据

```text
young	myope	no	reduced	no lenses
young	myope	no	normal	soft
young	myope	yes	reduced	no lenses
young	myope	yes	normal	hard
young	hyper	no	reduced	no lenses
young	hyper	no	normal	soft
young	hyper	yes	reduced	no lenses
young	hyper	yes	normal	hard
pre	myope	no	reduced	no lenses
pre	myope	no	normal	soft
```