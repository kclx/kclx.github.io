---
title: SVM简析
tags: [ 'ML', 'SVM', 'Lagrange Multiplier Method' ]
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: 支持向量机（SVM）是一种机器学习算法，其目标是在数据点中找到一个超平面，将不同类别的数据点分隔开，并使得边距最大化，以实现有效的分类。
swiper: false
swiperDesc: 支持向量机（SVM）是一种机器学习算法，其目标是在数据点中找到一个超平面，将不同类别的数据点分隔开，并使得边距最大化，以实现有效的分类。
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
date: 2023-08-01 11:00:35
updated: 2023-08-01 11:00:35
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/08/01/FOH_Ab2aQAo8JkV.jpeg
---

# SVM支持向量机

## 概述

SVM（支持向量机）是一种常用的监督学习算法，用于分类和回归任务。其主要目标是找到一个超平面，将不同类别的数据点有效地分隔开，并使得边距最大化。在分类问题中，SVM将数据点映射到特征空间，并通过支持向量寻找最优超平面，以在未知数据上取得良好的泛化性能。

SVM的核心原理包括以下几点：

1. 支持向量：在数据点中，距离超平面最近的一些数据点被称为支持向量，它们对超平面的位置和间隔起着关键作用。

2. 最大边距：SVM通过最大化超平面与支持向量之间的边距来寻找最优解。最大边距可以提高模型的鲁棒性，使其对新数据更具预测能力。

3. 分类器构建：SVM的目标是解决一个优化问题，通过数学方法找到最优的超平面。这个优化问题是一个凸二次规划问题，在线性可分的情况下有解。

4. 核函数：对于线性不可分的数据，SVM可以通过核函数将数据映射到高维特征空间，在高维空间中找到超平面，从而实现线性分类。常用的核函数有线性核、多项式核和高斯核等。

5. 正则化参数：SVM引入一个正则化参数C，用于平衡最大化边距和分类错误之间的权衡。较小的C值允许一些分类错误，而较大的C值对分类错误施加更严格的惩罚。

SVM具有良好的泛化能力，对于小样本数据和高维特征空间都表现出较好的性能。它在图像分类、文本分类、生物信息学等领域取得了广泛的应用。虽然SVM在处理大规模数据时存在一定的挑战，但在许多实际问题中，它仍然是一个有效且强大的分类器和回归器。

## 涉及到的数学公式

1. SVM分类器：
    - 假设超平面的方程为：$$f(x) = w^T x + b$$
    - 其中，$f(x)$ 是输入样本 $x$ 的预测输出，$w$ 是超平面的法向量（权重），$b$ 是超平面的偏置项。

2. 超平面到数据点的距离：
    - 数据点 $x$ 到超平面 $f(x)$ 的带符号距离为：$$d = \frac{w^T x + b}{\|w\|}$$，其中 $\|w\|$ 是 $w$ 的L2范数。

3. 分类决策规则：
    - 如果 $f(x) \geq 0$，则预测 $x$ 属于正类别。
    - 如果 $f(x) < 0$，则预测 $x$ 属于负类别。

4. 间隔（边距）：
    - 对于超平面 $(w^T x + b) = 0$，它到最近的正类别支持向量和最近的负类别支持向量之间的距离为 $\frac{2}{\|w\|}$。
    - SVM 的目标是最大化边距，即最小化 $\|w\|$。

5. 优化问题：
      SVM的优化问题是一个凸二次规划（Quadratic Programming，QP）问题。对于线性可分和线性不可分的情况，可以使用不同的方法来解决SVM的优化问题。

   1. 线性可分情况下的优化问题：
      对于线性可分的SVM，优化问题可以表示为：
      $$\text{Minimize:} \quad \frac{1}{2} \|w\|^2$$
      $$\text{Subject to:} \quad y_i (w^T x_i + b) \geq 1, \text{for all} \ i$$
      其中，$x_i$ 是训练样本，$y_i$ 是样本 $x_i$ 的标签（+1 或 -1），$w$ 是超平面的法向量，$b$ 是超平面的偏置项。

      这是一个凸二次规划问题，可以使用现有的凸优化求解器来求解。常用的凸优化求解器包括：
      - [序列最小优化算法](/2023/08/01/SVM简析/#序列最小优化SMO)（Sequential Minimal Optimization，SMO）
      - 内点法（Interior Point Method）
      - 梯度下降法（Gradient Descent）
      - 坐标下降法（Coordinate Descent）等

      这些方法可以找到拉格朗日乘子 $\alpha$ 的最优解，进而求解出超平面的参数 $w$ 和 $b$。

   2. 线性不可分情况下的优化问题：
      当数据线性不可分时，SVM引入松弛变量 $\xi_i$，使得部分数据点可以位于边界区域内。优化问题变为：
      $$\text{Minimize:} \quad \frac{1}{2} \|w\|^2 + C \sum \xi_i$$
      $$\text{Subject to:} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \text{for all} \ i$$
      $$\xi_i \geq 0, \text{for all} \ i$$
      其中，$C$ 是一个正则化参数，控制了分类错误的惩罚。

      这也是一个凸二次规划问题，可以使用相同的凸优化求解器来求解。通过求解优化问题，可以得到拉格朗日乘子 $\alpha$ 和松弛变量 $\xi_i$ 的最优解，进而求解出超平面的参数 $w$ 和 $b$。

   需要注意的是，当遇到非线性问题时，需要使用核函数来将数据映射到高维特征空间，然后再应用上述方法来求解对偶问题。这样，可以将非线性SVM转化为一个线性SVM问题来求解。

6. 核函数：
    - SVM通过核函数将数据映射到高维特征空间，以实现非线性分类。常用的核函数有线性核、多项式核和高斯核。
    - 线性核：$$K(x_i, x_j) = x_i^T x_j$$
    - 多项式核：$$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$$
    - 高斯核（径向基核）：$$K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$$

## 实现步骤

1. 数据预处理：
    - 收集和准备训练数据集，确保数据是有标签的，并且特征值和标签值已经转换为数值形式。
    - 如果数据集中存在缺失值或异常值，进行数据清洗和处理。

2. 特征缩放：
    - 对特征值进行缩放，确保所有特征在相同的尺度范围内。常见的缩放方法包括标准化（均值为0，方差为1）或者归一化（将特征缩放到0到1的范围内）。

3. 训练集和测试集划分：
    - 将数据集划分为训练集和测试集，用于模型的训练和评估。

4. 核函数选择（可选）：
    - 如果数据集是线性不可分的，需要选择合适的核函数对数据进行映射。常见的核函数有线性核、多项式核和高斯核等。

5. 优化问题解决：
    - 使用优化算法（例如凸二次规划问题的求解器）来找到SVM的最优超平面。
    - 线性可分的情况下，使用硬间隔SVM；线性不可分的情况下，使用软间隔SVM。

6. 正则化参数选择（可选）：
    - 对于软间隔SVM，需要选择合适的正则化参数C，该参数控制了分类错误的惩罚程度。C值较大将更严格惩罚分类错误，C值较小则更容忍分类错误。

7. 训练模型：
    - 使用训练集来训练SVM模型。模型的训练过程就是优化问题求解的过程，找到最优的超平面和参数。

8. 预测：
    - 使用训练好的模型对新的样本进行预测，计算预测输出的值 $f(x)$。

9. 评估模型性能：
    - 使用测试集来评估模型的性能。常见的评估指标包括准确率、精确率、召回率、F1-score等。

10. 调优（可选）：
    - 可以根据模型在测试集上的表现进行调优，例如尝试不同的核函数、调整正则化参数C等。

11. 应用模型：
    - 在实际应用中，使用训练好的SVM模型来进行分类任务，对新的未知样本进行分类预测。

## SVM与Logistic Regression

SVM（支持向量机）和逻辑回归在一些方面的数学公式和算法确实有相似之处，但它们在目标函数、决策边界和优化算法等方面存在明显的区别。下面列出它们的相似之处和区别：

相似之处：

1. 都是监督学习算法：SVM和逻辑回归都属于监督学习算法，都是用于解决分类问题。

2. 都使用线性模型：SVM和逻辑回归在默认情况下都使用线性模型进行分类。线性模型是通过特征的线性组合来进行分类的。

3. 都基于概率：逻辑回归可以输出样本属于某一类别的概率，而SVM通过决策函数的符号来判断样本的类别。

区别：

1. 目标函数和决策边界：
    - 逻辑回归的目标是最小化Logistic损失函数，并使用sigmoid函数将线性模型的输出映射到[0, 1]的概率值。决策边界是线性的。
    - SVM的目标是找到一个能够将不同类别的数据点有效分隔开的超平面，并最大化边距。决策边界是距离支持向量最近的超平面。

2. 损失函数：
    - 逻辑回归使用Logistic损失函数（也称为交叉熵损失函数），用于衡量模型预测与实际标签之间的差异。
    - SVM使用Hinge损失函数，它对正确分类的样本施加较小的损失，对于错误分类的样本施加较大的损失。

3. 优化算法：
    - 逻辑回归通常使用梯度下降等优化算法来最小化损失函数。
    - SVM使用凸二次规划等优化算法来找到最优的超平面。

4. 处理线性不可分数据：
    - 逻辑回归可以处理线性不可分的数据，但在处理非线性问题时可能需要引入特征工程或使用多项式特征。
    - SVM通过使用核函数将数据映射到高维特征空间，从而实现处理非线性分类问题。

## 判断线性可分

在SVM中，线性可分是指数据集在特征空间中存在一个超平面，能够将不同类别的数据点完全正确地分隔开，而不会出现任何错误分类。判断数据是否线性可分可以通过以下方法：

1. 可视化数据：将数据绘制在二维或三维空间中，观察数据点的分布情况。如果数据点可以被一个直线（在二维空间）或一个平面（在三维空间）完全分开，那么数据集很可能是线性可分的。

2. 线性SVM的结果：使用线性SVM对数据进行训练并绘制决策边界。如果线性SVM能够找到一个超平面，使得所有数据点都被正确分类，那么数据集是线性可分的。

3. 检查线性约束条件：对于二分类问题，在SVM中，线性可分的条件是所有数据点满足以下线性约束条件：
   $$y_i (w^T x_i + b) \geq 1, \text{for all} \ i$$
   其中，$x_i$ 是数据点，$y_i$ 是数据点的标签（+1 或 -1），$w$ 是超平面的法向量，$b$ 是超平面的偏置项。

4. 优化问题的结果：如果使用线性SVM解决优化问题，得到了一组满足上述约束条件的最优权重$w$和偏置项$b$，而且目标函数值为0（即最大化边距的结果），则说明数据是线性可分的。

需要注意的是，在实际应用中，数据可能并不是完全线性可分的，而是存在一些噪声或重叠情况。在这种情况下，可以使用软间隔SVM来处理部分分类错误，或者使用核函数将数据映射到高维特征空间来处理非线性可分问题。

## 序列最小优化SMO

序列最小优化（Sequential Minimal Optimization，简称SMO）是一种用于求解支持向量机（SVM）的优化算法，特别适用于处理大规模数据集的情况。SMO算法由John Platt于1998年提出，是一种迭代算法，通过不断选择两个变量进行优化，以逐步收敛到SVM的最优解。

SMO算法的基本思想是将SVM的对偶问题转化为一个二次规划问题，并采用启发式的方法来解决这个二次规划问题。SMO算法的主要步骤如下：

1. 初始化：初始化拉格朗日乘子 $\alpha$ 和偏置项 $b$ 为0，选择一个迭代次数的阈值或设置最大迭代次数。

2. 选择两个变量：在每一次迭代中，选择两个需要更新的拉格朗日乘子 $\alpha_i$ 和 $\alpha_j$。选择的方法可以采用启发式的方式，例如通过最大步长来选择 $\alpha_i$ 和 $\alpha_j$。

3. 优化两个变量：固定其他拉格朗日乘子 $\alpha$，将问题转化为一个只有两个变量的二次规划问题，并通过解析求解或优化算法（如SMO内循环）来找到 $\alpha_i$ 和 $\alpha_j$ 的最优解。

4. 更新参数：根据得到的 $\alpha_i$ 和 $\alpha_j$ 的最优解，更新拉格朗日乘子 $\alpha$ 和偏置项 $b$。

5. 终止条件：检查迭代次数是否达到设定的阈值，或者拉格朗日乘子的变化是否小于设定的容差。如果满足终止条件，则算法结束；否则，返回步骤2，继续下一轮迭代。

SMO算法的优点在于，它每次只更新两个拉格朗日乘子，因此在每一步的计算量相对较小。同时，SMO算法还使用了一些启发式的策略，帮助在高维空间中搜索更快地收敛到SVM的最优解。这使得SMO算法在大规模数据集下表现出了较好的计算效率。

值得注意的是，SMO算法用于求解线性SVM，当遇到非线性问题时，需要使用核函数来将数据映射到高维特征空间，然后再应用SMO算法来求解对偶问题。这样，SMO算法也可以用于求解非线性SVM。

## 案例

### 数据集

{% getFiles citation/svm, txt %}

### 实现

```python
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    b = 0
    alphas = mat(zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
    return b, alphas

def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

def plotfig_SVM(xMat, yMat, ws, b, alphas):
    xMat = mat(xMat)
    yMat = mat(yMat)
    b = array(b)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 0].flatten().A[0], xMat[:, 1].flatten().A[0])
    x = arange(-1.0, 10.0, 0.1)
    y = (-b - ws[0, 0] * x) / ws[1, 0]
    ax.plot(x, y)
    for i in range(shape(yMat[0, :])[1]):
        if yMat[0, i] > 0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'cx')
        else:
            ax.plot(xMat[i, 0], xMat[i, 1], 'kp')
    for i in range(100):
        if alphas[i] > 0.0:
            ax.plot(xMat[i, 0], xMat[i, 1], 'ro')
    plt.show()

if __name__ == "__main__":
    dataArr, labelArr = loadDataSet('6.SVM/testSet.txt')
    b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    plotfig_SVM(dataArr, labelArr, ws, b, alphas)
```

{% note info, 来源于 [ApacheCN](https://www.apachecn.org/) %}