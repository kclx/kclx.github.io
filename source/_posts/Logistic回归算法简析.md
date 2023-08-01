---
title: Logistic Regression算法简析
tags: ['ML', 'Logistic Regression']
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: Logistic Regression算法是一种用于解决二分类问题的线性分类算法，通过将线性模型的输出映射到概率空间并使用sigmoid函数对概率进行转换，从而进行分类预测。
swiper: false
swiperDesc: Logistic Regression算法是一种用于解决二分类问题的线性分类算法，通过将线性模型的输出映射到概率空间并使用sigmoid函数对概率进行转换，从而进行分类预测。
tocOpen: true
onlyTitle: false
share: true
copyright: true
donate: true
bgImgTransition: fade
bgImgDelay: 180000
prismjs: default
mathjax: true
imgTop: ture
date: 2023-07-31 17:10:22
updated: 2023-07-31 17:10:22
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/07/31/FOH-7wHaIAYlGGo.jpeg
---

# Logistic Regression算法简析

## 原理
Logistic Regression是一种广义线性模型，用于处理二分类问题。它通过建立一个线性模型，并使用逻辑函数（sigmoid函数）将线性输出转换为概率值，从而对输入数据进行分类。

模型假设：
- 假设输入特征和输出类别之间存在线性关系。
- 用sigmoid函数将线性输出映射到0和1之间的概率值。

**sigmoid函数：**
sigmoid函数（也称为Logistic函数）定义如下：
```
sigmoid(z) = 1 / (1 + exp(-z))
```
其中z是输入的线性输出。

**梯度提升与梯度下降**

梯度提升（Gradient Boosting）和梯度下降（Gradient Descent）是两个不同的机器学习概念，虽然它们都涉及梯度（Gradient）这一术语，但在方法和应用上有很大的区别。

**梯度提升（Gradient Boosting）：**
梯度提升是一种集成学习技术，它通过将多个弱学习器（通常是决策树）进行串行训练，每次训练都尝试纠正前一轮训练中模型的错误，从而逐步提高整体模型的性能。这种技术通过迭代的方式构建一个强大的集成模型，每一步都关注先前模型的残差（预测值与真实值之间的差异），然后训练一个新的模型来纠正这些残差。梯度提升的主要代表算法是Gradient Boosting Machine（GBM）和XGBoost。

**梯度下降（Gradient Descent）：**
梯度下降是一种用于优化目标函数的迭代优化算法。它的主要目标是在参数空间中找到目标函数的最小值点（或最大值点）。在机器学习中，这通常涉及到最小化损失函数，例如平方误差损失或交叉熵损失等。梯度下降的基本思想是通过计算目标函数关于参数的梯度（导数），朝着梯度的反方向调整参数，以使目标函数值逐步减小。梯度下降有多种变种，如批量梯度下降（Batch Gradient Descent）、随机梯度下降（Stochastic Gradient Descent）和小批量梯度下降（Mini-batch Gradient Descent）等。

**主要区别：**
- 梯度提升是一种集成学习算法，用于构建强大的预测模型，通过串行训练多个弱学习器来逐步提高整体模型的性能。
- 梯度下降是一种优化算法，用于找到目标函数的最小值点，通过计算梯度并朝着梯度的反方向调整参数来实现优化。

尽管它们涉及到梯度这个共同点，但是梯度提升和梯度下降是两个不同的概念，分别用于解决集成学习和优化问题。

## 优缺点

*优点*：
- 实现简单，计算效率高。
- 可解释性强，能够看到每个特征的权重对预测的影响。
- 在特征空间较简单的问题上表现良好。

*缺点*：
- 不能处理复杂的数据关系，对于非线性问题表现不佳。
- 对异常值比较敏感，容易受到噪声的影响。
- 不能直接处理多分类问题，通常需要使用一对多（One-vs-Rest）或Softmax等策略来扩展到多分类。

## 应用
Logistic Regression在许多领域有广泛的应用，尤其在二分类问题中常见，如：
- 垃圾邮件分类：判断一封邮件是否为垃圾邮件。
- 金融风险预测：预测客户是否有违约风险。
- 医学诊断：判断患者是否患有某种疾病。
- 自然语言处理：情感分析，判断文本的情感倾向。

## 涉及到的数学公式

1. 假设函数：
    逻辑回归的假设函数表示为：
    $$ h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}} $$
    
    其中，
      - $ h_\theta(x) $ 是根据输入特征 $ x $ 和模型参数 $ \theta $ 得到的预测值（类别为1的概率）。
      - $ \theta $ 是模型的参数向量。
      - $ \theta^T $ 表示 $ \theta $ 的转置。
      - $ e $ 是自然对数的底数。

2. 损失函数：
   逻辑回归使用交叉熵损失函数来衡量预测值与真实类别之间的差异。对于二分类问题，交叉熵损失函数表示为：
   $$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] $$

    其中，
      - $ J(\theta) $ 是损失函数。
      - $ m $ 是训练样本的数量。
      - $ x^{(i)} $ 表示第 $ i $ 个训练样本的特征向量。
      - $ y^{(i)} $ 是第 $ i $ 个训练样本的真实类别（0或1）。
      - $ h_\theta(x^{(i)}) $ 是根据假设函数预测的类别为1的概率。

3. 梯度下降更新规则：
   梯度下降的目标是最小化损失函数 $ J(\theta) $。为了更新模型参数 $ \theta $，我们需要计算损失函数对于每个参数的偏导数，然后根据梯度的方向和学习率来更新参数。更新规则如下：
   $$ \theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j} $$

    其中，
      - $ \alpha $ 是学习率（控制参数更新的步长）。
      - $ \frac{\partial J(\theta)}{\partial \theta_j} $ 是损失函数对于参数 $ \theta_j $ 的偏导数。

    对于逻辑回归，梯度下降的具体更新规则是：
    $$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} $$

    其中，$ x_j^{(i)} $ 是第 $ i $ 个训练样本的第 $ j $ 个特征值。

通过不断迭代更新参数，梯度下降会使损失函数逐渐减小，从而找到最优的模型参数，使得逻辑回归模型能够较好地拟合训练数据并作出准确的预测。

## 案例：从疝气病症预测病马的死亡率

### 数据集

{% getFiles citation/HorseColic,  txt, %}

### Python实现

```python
# -------从疝气病症预测病马的死亡率------
def colic_test():
    """
    打开测试集和训练集，并对数据进行格式化处理,其实最主要的的部分，比如缺失值的补充（真的需要学会的），人家已经做了
    :return: 
    """
    f_train = open('5.Logistic/HorseColicTraining.txt', 'r')
    f_test = open('5.Logistic/HorseColicTest.txt', 'r')
    training_set = []
    training_labels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in f_train.readlines():
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1:
            continue    # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    train_weights = stoc_grad_ascent1(np.array(training_set), training_labels, 500)
    error_count = 0
    num_test_vec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率
    for line in f_test.readlines():
        num_test_vec += 1
        curr_line = line.strip().split('\t')
        if len(curr_line) == 1: 
            continue    # 这里如果就一个空的元素，则跳过本次循环
        line_arr = [float(curr_line[i]) for i in range(21)]
        if int(classify_vector(np.array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = error_count / num_test_vec
    print('the error rate is {}'.format(error_rate))
    return error_rate
   
def stoc_grad_ascent1(data_mat, class_labels, num_iter=150):
    """
    改进版的随机梯度上升，使用随机的一个样本来更新回归系数
    :param data_mat: 输入数据的数据特征（除去最后一列）,ndarray
    :param class_labels: 输入数据的类别标签（最后一列数据
    :param num_iter: 迭代次数
    :return: 得到的最佳回归系数
    """
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for j in range(num_iter):
        # 这里必须要用list，不然后面的del没法使用
        data_index = list(range(m))
        for i in range(m):
            # i和j的不断增大，导致alpha的值不断减少，但是不为0
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机产生一个 0～len()之间的一个值
            # random.uniform(x, y) 方法将随机生成下一个实数，它在[x,y]范围内,x是这个范围内的最小值，y是这个范围内的最大值。
            rand_index = int(np.random.uniform(0, len(data_index)))
            h = sigmoid(np.sum(data_mat[data_index[rand_index]] * weights))
            error = class_labels[data_index[rand_index]] - h
            weights = weights + alpha * error * data_mat[data_index[rand_index]]
            del(data_index[rand_index])
    return weights

def sigmoid(x):
    # 这里其实非常有必要解释一下，会出现的错误 RuntimeWarning: overflow encountered in exp
    # 这里是因为我们输入的有的 x 实在是太小了，比如 -6000之类的，那么计算一个数字 np.exp(6000)这个结果太大了，没法表示，所以就溢出了
    return 1.0 / (1 + np.exp(-x))

def classify_vector(in_x, weights):
    """
    最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    :param in_x: 特征向量，features
    :param weights: 根据梯度下降/随机梯度下降 计算得到的回归系数
    :return: 
    """
    # print(np.sum(in_x * weights))
    prob = sigmoid(np.sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    return 0.0

if __name__ == '__main__':
    colic_test()
```

{% note info, 来源于 [ApacheCN](https://www.apachecn.org/) %}