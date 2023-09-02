---
title: Regression算法简析
tags: [ 'ML', 'Regression' ]
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: 回归算法是一类用于预测数值型输出（连续变量）的机器学习算法，通过分析特征与输出之间的关系建立模型，以预测未知数据的输出值，常见的回归算法包括线性回归、岭回归、Lasso回归等。
swiper: false
swiperDesc: 回归算法是一类用于预测数值型输出（连续变量）的机器学习算法，通过分析特征与输出之间的关系建立模型，以预测未知数据的输出值，常见的回归算法包括线性回归、岭回归、Lasso回归等。
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
date: 2023-08-03 13:43:27
updated: 2023-08-03 13:43:27
swiperImg:
bgImg:
img:
---

# Regression算法简析

## 概述

回归（Regression）是机器学习中的一种重要算法和任务，其主要目标是预测连续数值型的输出（或称为目标变量）在输入特征（或称为自变量）的基础上。回归可以被认为是一种建模技术，它用于分析变量之间的关系并预测未来的趋势。

在回归中，我们假设输入特征和输出之间存在某种函数关系，回归算法的目标就是找到这个函数，以便能够在新的输入数据上进行预测。这个函数通常被称为回归模型，它可以用来预测连续的输出值。

回归问题可以分为以下几种类型：

1. **线性回归（Linear Regression）：** 这是最简单的回归方法之一。它假设输入特征和输出之间存在线性关系。线性回归试图找到最佳拟合的直线（或超平面），使得输入特征和输出的拟合误差最小。

2. **多项式回归（Polynomial Regression）：** 在某些情况下，线性模型无法准确地捕捉输入和输出之间的复杂关系。多项式回归通过引入多项式项来扩展线性模型，从而更好地拟合数据。

3. **岭回归（Ridge Regression）和Lasso回归（Lasso Regression）：** 这些是正则化的回归方法，用于控制模型的复杂度，防止过拟合。它们通过添加正则化项来限制模型参数的大小，从而在拟合数据时更加稳定。

4. **支持向量回归（Support Vector Regression，SVR）：** 类似于支持向量机（SVM），SVR通过在输入空间中找到一个“边界带”来预测输出。它着重于在允许一定误差的情况下拟合数据。

5. **决策树回归（Decision Tree Regression）：** 这种方法使用决策树来建立输入特征和输出之间的映射关系。它将输入空间划分成不同的区域，每个区域对应一个输出值。

6. **随机森林回归（Random Forest Regression）：** 随机森林是基于决策树的集成方法，它通过组合多个决策树来改善预测性能，并减少过拟合风险。

7. **神经网络回归（Neural Network Regression）：** 借助深度神经网络，神经网络回归能够建模复杂的非线性关系。它适用于更大规模和更复杂的数据集。

回归问题的评估通常使用各种性能指标，如均方误差（Mean Squared Error）、平均绝对误差（Mean Absolute Error）、决定系数（Coefficient of Determination，R-squared）等来衡量模型的预测性能。

{% note info, 逻辑回归虽然名字中包含“回归”，但实际上它是一种分类算法，而不是回归算法。逻辑回归用于解决二分类问题，即将输入特征映射到一个概率值，表示属于某个类别的可能性，然后根据阈值来进行分类决策。 %}

## 应用场景

回归在许多不同领域都有广泛的应用，它主要用于预测和建模连续数值型的输出变量。

1. **经济学和金融：** 回归可用于预测股票价格、商品价格、通货膨胀率等经济和金融指标，帮助投资决策和风险管理。

2. **房地产：** 用于预测房价，考虑到各种房屋特征（如面积、地理位置、楼层等）对价格的影响。

3. **医学：** 回归可以应用于疾病风险预测、药物剂量选择以及医疗成本预测等领域。

4. **自然科学：** 在物理、化学、地理等领域，回归可用于分析实验数据、模拟物理过程，并帮助理解自然现象。

5. **社会科学：** 用于预测人口统计数据、社会行为趋势、选民投票倾向等。

6. **工程和技术：** 回归可用于预测产品寿命、材料强度、工程参数等，有助于设计和制造过程的优化。

7. **市场营销：** 回归在市场营销中用于预测销售量、市场份额，以及了解广告投放和促销策略对销售的影响。

8. **环境科学：** 回归可用于分析环境数据，例如预测气温变化、大气污染水平等。

9. **农业：** 用于预测农作物产量、土壤质量，以及农业实践对收成的影响。

10. **教育：** 在教育领域，回归可以用于预测学生的学术成绩，以及了解不同因素对学生表现的影响。

11. **运输与物流：** 用于预测货运量、交通拥堵情况，以及运输成本的变化。

## 必备知识

1. 矩阵求逆

   矩阵求逆是一个在线性代数中非常重要的操作，它用于解决一系列问题，例如线性方程组的求解、线性变换的逆变换等。对于一个方阵（即行数和列数相等的矩阵），如果它是可逆的（也称为非奇异的），则可以求出它的逆矩阵。

   设A是一个n阶方阵，如果存在一个n阶矩阵B，使得AB = BA = I（其中I是n阶单位矩阵），则称B是A的逆矩阵，记作A⁻¹。

   然而，并不是所有矩阵都是可逆的。一个矩阵A可逆的条件是其行列式不为零，即det(A) ≠ 0。

   对于一个可逆矩阵A，它的逆矩阵A⁻¹可以通过以下方式求解：

   如果A是一个2x2矩阵：$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$
   则A的逆矩阵为：$$A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

   对于更高维的矩阵，可以使用高斯-约旦消元法、LU分解等方法来求逆矩阵。另外，许多数值计算库和软件提供了求逆矩阵的函数，如Python的NumPy库中的`numpy.linalg.inv()`函数。

   需要注意的是，矩阵求逆可能会涉及到数值稳定性的问题，特别是在计算机中使用有限的浮点数表示。对于大型或病态（ill-conditioned）的矩阵，求逆可能会导致数值不稳定，需要谨慎处理。

   {% note warning, 这里矩阵的mathjax格式不兼容，bc位置应该是换行显示的！ %}

2. 最小二乘法

   最小二乘法（Least Squares Method）是一种用于拟合数据和估计模型参数的统计技术。它的主要思想是通过最小化实际观测值与模型预测值之间的误差的平方和，来找到最佳的模型参数。最小二乘法在回归分析、曲线拟合等问题中得到广泛应用。

   在回归分析中，最小二乘法用于拟合一个线性模型，使模型的预测值与观测值之间的误差最小化。具体来说，对于一个包含 $m$ 个样本的数据集，每个样本包含一个或多个输入特征 $x$ 和一个输出 $y$。我们希望找到一个线性模型 $y = f(x)$，其中 $f(x)$ 表示输入特征 $x$ 对应的模型预测值。然后，通过最小二乘法，我们寻找最佳的模型参数（回归系数）使得观测值与预测值之间的误差平方和最小。

   数学上，对于一个线性模型 $y = w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n$，最小二乘法的目标是找到最佳的回归系数 $w_0, w_1, w_2, \ldots, w_n$，使得误差平方和最小，即最小化以下的损失函数：

   $$\text{Loss} = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

   其中，$y_i$ 是第 $i$ 个样本的实际输出，$\hat{y}_i$ 是模型预测的输出。

   最小二乘法的求解过程通常涉及对损失函数进行求导，然后将导数设置为零，从而得到模型参数的解析解。这个解析解可以用于求解最佳的回归系数。

   假设有以下变量：
   - $x_{\text{test}}$: 待预测的样本点的特征向量
   - $x_{\text{arr}}$: 所有样本的特征数据矩阵，每一行是一个样本的特征向量
   - $y_{\text{arr}}$: 所有样本的目标变量向量
   - $\tau$: 带宽参数，控制权重的衰减速率
   - $w^{(i)}$: 样本点的权重矩阵，表示样本点与待预测点的距离所计算的权重
   - $\mathbf{x}_{\text{mat}}$: 样本特征数据矩阵
   - $\mathbf{y}_{\text{mat}}$: 样本目标变量矩阵
   - $\mathbf{x}_ {\text{tx}}$: 矩阵 $\mathbf{x}_{\text{mat}}$ 的转置与自身的乘积
   - $\mathbf{w}_{\text{mat}}$: 权重矩阵 $\mathbf{w}$，对角矩阵，由 $w^{(i)}$ 组成
   - $\mathbf{y}_{\text{hat}}$: 预测点估计值矩阵

   最小二乘法计算回归系数的数学公式为：
   $$ \mathbf{w}_{\text{mat}} = \text{diag}(w^{(1)}, w^{(2)}, \ldots, w^{(m)}) \quad \text{(对角矩阵)} $$

   $$ \mathbf{x} _ {\text{tx}} = \mathbf{x} _ {\text{mat}}^T \cdot \mathbf{x} _ {\text{mat}} $$

   $$ \hat{\beta} = (\mathbf{x}_ {\text{tx}} \cdot \mathbf{w}_ {\text{mat}} \cdot \mathbf{x}_ {\text{mat}})^{-1} \cdot \mathbf{x}_ {\text{mat}}^T \cdot \mathbf{w}_ {\text{mat}} \cdot \mathbf{y}_ {\text{mat}} $$

   $$ \mathbf{y}_ {\text{hat}} = \mathbf{x}_ {\text{test}} \cdot \hat{\beta} $$

## 线性回归

### 原理

线性回归是一种常见且简单的回归算法，用于建立输入特征和输出之间的线性关系。其原理可以概括为以下几个步骤：

1. **假设线性关系：** 首先，线性回归假设输入特征和输出之间存在线性关系，即可以用线性方程来表示。这个线性方程的形式可以表示为：$$y = w_0 + w_1x_1 + w_2x_2 + \ldots + w_nx_n$$其中，$y$是预测的输出，$x_1, x_2, \ldots, x_n$是输入特征，$w_0, w_1, w_2, \ldots, w_n$是回归系数（权重）。

2. **最小化误差：** 线性回归的目标是找到最佳的回归系数，使得预测值与真实输出之间的误差最小化。通常使用均方误差（Mean Squared Error，MSE）作为衡量误差的指标。MSE的定义为：$$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$其中，$m$是样本数量，$y_i$是第$i$个样本的真实输出，$\hat{y}_i$是线性模型预测的输出。

3. **求解回归系数：** 最小化MSE的过程可以使用不同的优化算法，常见的方法是最小二乘法（Ordinary Least Squares，OLS）。最小二乘法的目标是找到能够最小化误差平方和的回归系数，从而得到最佳拟合的直线（或超平面）。

4. **特征处理：** 在进行线性回归之前，通常需要对输入特征进行预处理，包括特征缩放、特征选择、特征变换等。这有助于提高模型的性能和稳定性。

5. **预测和评估：** 在训练完成后，使用训练得到的回归系数来进行预测。对于新的输入特征，通过线性方程计算预测值。模型的性能可以通过各种指标（如R-squared、均方误差等）来进行评估。

## 局部加权线性回归

### 原理

局部加权线性回归（Locally Weighted Linear Regression，LWLR）是一种非参数回归方法，用于在回归分析中拟合数据并预测目标变量。与传统的线性回归方法不同，LWLR 在预测时考虑了样本点附近的权重，从而能够更好地捕捉数据中的局部非线性关系。

LWLR 的主要思想是，对于待预测的数据点，通过对训练数据中的每个点赋予不同的权重，来进行加权线性回归。在预测时，距离待预测点近的训练样本会被赋予较高的权重，而距离较远的训练样本则会被赋予较低的权重。这样可以使得模型在局部区域内更关注于邻近样本，从而更好地适应局部的数据分布。

具体来说，对于待预测点 $(x_{\text{test}}, y_{\text{test}})$，LWLR 的回归模型为：

$$y_{\text{test}} = x_{\text{test}}^T \cdot \hat{\beta}$$

其中，$\hat{\beta}$ 是通过加权最小二乘法计算得到的回归系数，权重 $w^{(i)}$ 由核函数计算得出：

$$w^{(i)} = \exp\left(-\frac{(x^{(i)} - x_{\text{test}})^T \cdot (x^{(i)} - x_{\text{test}})}{2\tau^2}\right)$$

其中，$x^{(i)}$ 是训练样本的特征向量，$\tau$ 是一个用户指定的参数，控制权重的衰减速度。核函数通常使用高斯核函数（Gaussian Kernel），但也可以选择其他类型的核函数。

需要注意的是，LWLR 是一种非参数方法，因为它不对模型形式做出明确的假设，而是在每个预测点处都进行加权线性回归。虽然 LWLR 在局部能够更好地拟合数据，但也存在计算复杂度高和参数选择的问题。

LWLR 在处理数据中存在局部非线性关系时具有优势，但在全局趋势较为明显的情况下，可能会导致过拟合。

## 岭回归

### 原理

岭回归（Ridge Regression）是一种用于处理多重共线性问题的线性回归技术，它在最小二乘法的基础上引入了正则化项，有助于提高模型的稳定性和泛化能力。岭回归的主要思想是通过对回归系数的大小进行限制，来减小模型的复杂度，从而降低过拟合的风险。

在普通的最小二乘线性回归中，我们试图最小化损失函数 $J(\beta) = \sum_{i=1}^{m}(y^{(i)} - \mathbf{x}^{(i)T}\beta)^2$，其中 $\beta$ 是回归系数，$y^{(i)}$ 是目标变量，$\mathbf{x}^{(i)}$ 是特征向量。而岭回归则在这个基础上引入了一个正则化项，使得最小化的目标变为：

$$J_{\text{ridge}}(\beta) = \sum_{i=1}^{m}(y^{(i)} - \mathbf{x}^{(i)T}\beta)^2 + \lambda \sum_{j=1}^{n}\beta_j^2$$

其中 $\lambda$ 是一个用户指定的正则化参数，$n$ 是特征的数量。第二项 $\lambda \sum_{j=1}^{n}\beta_j^2$ 被称为岭惩罚项（Ridge Penalty），它惩罚了回归系数的大小。当 $\lambda$ 较大时，回归系数会被强烈惩罚，趋向于接近于零，从而减小模型的复杂度。

岭回归的求解过程可以使用类似最小二乘法的方法，通过求解下面的最优化问题来得到回归系数 $\beta$：

$$ \hat{\beta}_ {\text{ridge}} = \arg\min_{\beta} \left( \sum_{i=1}^{m} \left(y^{(i)} - \mathbf{x}^{(i)T}\beta\right)^2 + \lambda \sum_{j=1}^{n} \beta_j^2 \right) $$

岭回归的一个重要特点是，它可以处理多重共线性问题，即特征之间存在强相关性的情况。在普通的线性回归中，当特征之间存在共线性时，回归系数的估计可能会变得不稳定，而岭回归通过引入正则化项可以在一定程度上缓解这个问题。

选择合适的正则化参数 $\lambda$ 对岭回归的性能具有重要影响。较小的 $\lambda$ 可能会导致模型过拟合，而较大的 $\lambda$ 可能会导致模型欠拟合。通常可以通过交叉验证等方法来选择最优的 $\lambda$ 值。

## 前向逐步回归

### 原理

前向逐步回归是一个基于贪婪算法的特征选择方法。它从一个包含无特征的模型开始，并逐步添加或减少特征，直到达到某个终止条件（例如，达到预定的特征数量或误差不再显著减少）。

原理：
1. 开始时，模型不包含任何特征，即所有回归系数都设为0。
2. 每一步，考虑添加一个特征或减去一个特征，或者增加/减少一个特征的系数。
3. 选择那些能够使误差最小化的特征或系数更新。

公式如下：

1. 计算当前模型的误差 $E_0$（例如，均方误差）:

   $$ E_0 = \sum_{i=1}^N (y_i - \mathbf{x}_i^T \mathbf{w})^2 $$
   
   其中，$y_i$ 是第 $i$ 个观测值的响应变量，$\mathbf{x}_i$ 是第 $i$ 个观测值的特征向量，$\mathbf{w}$ 是回归系数向量。

2. 对于每一个特征 $j$，计算增加或减少这个特征（或改变它的系数）时的新误差 $E_j$:

   $$ E_j = \sum_{i=1}^N (y_i - \mathbf{x}_i^T \mathbf{w} + \Delta w_j \mathbf{x} _{ij})^2 $$

   其中，$\Delta w_j$ 是对第 $j$ 个系数的改变量，$\mathbf{x}_{ij}$ 是第 $i$ 个观测值的第 $j$ 个特征。

3. 选择那个使 $E_j - E_0$ 最小的特征进行更新。

4. 重复上述步骤，直到满足终止条件。

   在每一步，前向逐步回归会尝试增加、减少或不改变每一个特征的系数，然后选择那个使误差最小化的更新。这是一个迭代的、基于贪婪策略的过程，旨在逐步构建一个误差最小的模型。


## 案例

### 数据集

{% getFiles citation/Regression, txt %}

### 线性回归

```python
from numpy import *
import matplotlib.pylab as plt

def loadDataSet(fileName):
    """
    加载数据
    解析以 tab 键分隔的文件中的浮点数

    Returns:
        dataMat :   feature 对应的数据集
        labelMat :  feature 对应的分类标签，即类别标签
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

def standRegres(xArr, yArr):
    """
    线性回归

    Args:
        xArr : 输入的样本数据，包含每个样本数据的 feature
        yArr : 对应于输入数据的类别标签，也就是每个样本对应的目标变量

    Returns:
        ws: 回归系数
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def regression1():
    xArr, yArr = loadDataSet("8.Regression/data.txt")
    xMat = mat(xArr)
    yMat = mat(yArr)
    ws = standRegres(xArr, yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten(), yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

if __name__ == '__main__':
    regression1()
```

### 局部加权线性回归

```python
from numpy import *
import matplotlib.pylab as plt


def loadDataSet(fileName):
    """
    加载数据
    解析以 tab 键分隔的文件中的浮点数

    Returns:
        dataMat :   feature 对应的数据集
        labelMat :  feature 对应的分类标签，即类别标签
    """
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def lwlr(testPoint, xArr, yArr, k=1.0):
    """
    局部加权线性回归，在待预测点附近的每个点赋予一定的权重，在子集上基于最小均方差来进行普通的回归。

    Args:
        testPoint: 样本点
        xArr: 样本的特征数据，即 feature
        yArr: 每个样本对应的类别标签，即目标变量
        k: 关于赋予权重矩阵的核的一个参数，与权重的衰减速率有关

    Returns:
        testPoint * ws: 数据点与具有权重的系数相乘得到的预测点
    """
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    """
    测试局部加权线性回归，对数据集中每个点调用 lwlr() 函数

    Args:
        testArr: 测试所用的所有样本点
        xArr: 样本的特征数据，即 feature
        yArr: 每个样本对应的类别标签，即目标变量
        k: 控制核函数的衰减速率

    Returns:
        yHat: 预测点的估计值
    """
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def lwlrTestPlot(xArr, yArr, k=1.0):
    """
    首先将 X 排序，其余的都与 lwlrTest 相同，这样更容易绘图

    Args:
        xArr: 样本的特征数据，即 feature
        yArr: 每个样本对应的类别标签，即目标变量，实际值
        k: 控制核函数的衰减速率的有关参数，这里设定的是常量值 1

    Returns:
        yHat: 样本点的估计值
        xCopy: xArr 的复制
    """
    yHat = zeros(shape(yArr))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i], xArr, yArr, k)
    return yHat, xCopy


def regression2():
    xArr, yArr = loadDataSet("8.Regression/data.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()


if __name__ == '__main__':
    regression2()
```

### 岭回归

```python
from numpy import *
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    '''
    加载数据
    解析以tab键分隔的文件中的浮点数
    Args:
        fileName: 文件名
    Returns:
        dataMat : 特征数据集
        labelMat : 类别标签
    '''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def ridgeRegres(xMat, yMat, lam=0.2):
    '''
    进行岭回归求解
    Args:
        xMat: 特征数据
        yMat: 类别标签，实际值
        lam: λ值，用于使矩阵非奇异
    Returns:
        回归系数
    '''
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    '''
    在一组 λ 上测试岭回归结果
    Args:
        xArr: 特征数据
        yArr: 类别标签，真实数据
    Returns:
        所有回归系数输出到矩阵并返回
    '''
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def regression3():
    abX, abY = loadDataSet("8.Regression/abalone.txt")
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()


if __name__ == '__main__':
    regression3()
```

### 前向逐步回归

```python
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))  # 测试代码删除
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat
    
def regularize(xMat):  # 按列进行规范化
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)  # 计算平均值然后减去它
    inVar = var(inMat, 0)  # 计算除以Xi的方差
    inMat = (inMat - inMeans) / inVar
    return inMat
    
def rssError(yArr, yHatArr):
    '''
        Desc:
            计算分析预测误差的大小
        Args:
            yArr: 真实的目标变量
            yHatArr: 预测得到的估计值
        Returns:
            计算真实值和估计值得到的值的平方和作为最后的返回值
    '''
    return ((yArr - yHatArr) ** 2).sum()
    
def standRegres(xArr, yArr):
    '''
    Description: 
        线性回归
    Args:
        xArr : 输入的样本数据，包含每个样本数据的 feature
        yArr : 对应于输入数据的类别标签，也就是每个样本对应的目标变量
    Returns:
        ws: 回归系数
    '''

    # mat()函数将xArr，yArr转换为矩阵 mat().T 代表的是对矩阵进行转置操作
    xMat = mat(xArr)
    yMat = mat(yArr).T
    # 矩阵乘法的条件是左矩阵的列数等于右矩阵的行数
    xTx = xMat.T * xMat
    # 因为要用到xTx的逆矩阵，所以事先需要确定计算得到的xTx是否可逆，条件是矩阵的行列式不为0
    # linalg.det() 函数是用来求得矩阵的行列式的，如果矩阵的行列式为0，则这个矩阵是不可逆的，就无法进行接下来的运算
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 最小二乘法
    # http://cwiki.apachecn.org/pages/viewpage.action?pageId=5505133
    # 书中的公式，求得w的最优解
    ws = xTx.I * (xMat.T * yMat)
    return ws

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     # 也可以规则化ys但会得到更小的coef
    xMat = regularize(xMat)
    m,n=shape(xMat)
    #returnMat = zeros((numIt,n)) # 测试代码删除
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print (ws.T)
        lowestError = inf; 
        for j in range(n):
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat


def regression4():
    xArr, yArr = loadDataSet("8.Regression/abalone.txt")
    stageWise(xArr, yArr, 0.01, 200)
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xMat = regularize(xMat)
    yM = mean(yMat, 0)
    yMat = yMat - yM
    weights = standRegres(xMat, yMat.T)
    print(weights.T)
    
if __name__ == '__main__':
    regression4()
```