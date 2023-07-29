---
title: KNN算法简析
tags: [ 'ML', 'KNN' ]
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: KNN算法简析
swiper: false
swiperDesc: KNN算法简析
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
date: 2023-07-27 17:43:29
updated: 2023-07-27 17:43:29
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/07/27/2023-07-27-19.23.37.png
---

# KNN算法简析

## 概述

K最近邻算法（K-Nearest Neighbors，KNN）是一种常用的监督学习算法，用于分类和回归问题。它是一种简单而有效的算法，在实践中广泛应用。KNN算法基于实例之间的相似性来进行分类或回归预测。

以下是KNN算法的概述：

1. 算法原理：
    - 对于分类问题，KNN算法通过测量新样本点与训练数据集中已标记样本点之间的距离来进行分类。它选择与新样本点距离最近的K个已标记样本点，并根据这K个点中占多数的类别来对新样本进行分类。
    - 对于回归问题，KNN算法计算新样本点与训练数据集中已标记样本点之间的距离，然后选择距离最近的K个样本点，并使用这K个样本点的平均值（或加权平均值）作为新样本点的预测值。

2. 距离度量：
    - 在KNN算法中，常用的距离度量方法包括欧氏距离（Euclidean distance）、曼哈顿距离（Manhattan distance）和闵可夫斯基距离（Minkowski distance）。欧氏距离在连续特征空间中广泛使用，而曼哈顿距离对于具有稀疏特征的数据集更为适用。

3. 确定K值：
    - K值是KNN算法中的一个重要参数。它表示选择多少个最近邻来进行分类或回归。较小的K值可能会导致模型复杂，容易受到噪声的影响，而较大的K值可能会导致模型过于简单，忽略了局部的数据模式。通常，K值的选择通过交叉验证等技术来进行调优。

4. 优缺点：
    - 优点：KNN算法简单易懂，易于实现，对于多类别问题表现良好。它不需要对数据进行假设，适用于非线性数据。
    - 缺点：KNN算法计算复杂度较高，特别是在大规模数据集上。对于高维数据，容易受到维度灾难的影响。此外，KNN算法对于不平衡数据集的处理效果可能较差。

5. 实现步骤：
    - 加载数据集。
    - 选择距离度量方法。
    - 选择K值。
    - 对于分类问题，计算新样本与训练样本的距离，选择最近的K个样本进行投票得出分类结果。
    - 对于回归问题，计算新样本与训练样本的距离，选择最近的K个样本进行平均（或加权平均）得出预测值。

总结：KNN算法是一种简单而强大的算法，适用于许多分类和回归问题。它的实现简单，但在大规模数据集上计算效率较低。合理选择K值以及适当的距离度量方法是使用KNN算法时需要注意的要点。

## 涉及到的数学公式

1. 欧氏距离（Euclidean distance）：
   $$ d(A, B) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} $$

2. 曼哈顿距离（Manhattan distance）：
   $$ d(A, B) = \sum_{i=1}^{n}|x_i - y_i| $$

3. KNN分类公式：
   $$ \text{Class}(\text{new\_sample}) = \text{argmax}\left(\sum_{i}{\mathbb{I}(label_i = c) \text{ for } i \text{ in neighbors}}\right) $$

    其中，$\text{Class}(\text{new\_sample})$ 表示新样本点的预测类别，$label_i$ 是第 $i$ 个邻居样本的类别，$c$ 表示所有可能的类别，$\mathbb{I}(condition)$ 是指示函数，如果 $condition$ 为真，则 $\mathbb{I}(condition) = 1$，否则为 0。 $\text{argmax}$ 函数返回使括号内表达式最大的类别 $c$。

4. KNN回归公式：
   $$ \text{Prediction}(\text{new\_sample}) = \frac{1}{K}\sum_{i}{label_i \text{ for } i \text{ in neighbors}} $$

其中，$\text{Prediction}(\text{new\_sample})$ 表示新样本点的回归预测值，$label_i$ 是第 $i$ 个邻居样本的回归标签。

### 欧式距离与曼哈顿距离

欧式距离（Euclidean distance）和曼哈顿距离（Manhattan distance）是两种常用的距离度量方法，用于衡量在n维空间中两个点之间的距离。它们在计算方法和性质上有一些异同：

异同点如下：

1. 计算方法：
   - 欧式距离：欧式距离是两点之间直线距离的度量，也称为直线距离。在n维空间中，欧式距离计算公式为：
     $$ d_{\text{euclidean}}(A, B) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} $$

   - 曼哈顿距离：曼哈顿距离是两点之间沿坐标轴方向的距离之和，也称为城市街区距离。在n维空间中，曼哈顿距离计算公式为：
     $$ d_{\text{manhattan}}(A, B) = \sum_{i=1}^{n}|x_i - y_i| $$

2. 距离形状：
   - 欧式距离：欧式距离在几何上对应于直线的长度，因此在n维空间中呈现圆形等距离轮廓。
   - 曼哈顿距离：曼哈顿距离在几何上对应于沿坐标轴方向的步行距离，因此在n维空间中呈现方形等距离轮廓。

3. 敏感度：
   - 欧式距离：欧式距离对各个维度的差异比较敏感，当特征维度之间的差异很大时，欧式距离可能会受到影响。
   - 曼哈顿距离：曼哈顿距离对各个维度的差异不那么敏感，因为它只考虑了沿坐标轴方向的距离。

4. 计算效率：
   - 欧式距离：计算欧式距离需要进行平方和开方的运算，计算相对较慢。
   - 曼哈顿距离：计算曼哈顿距离只需要进行绝对值和求和运算，计算相对较快。

在实际应用中，选择使用欧式距离还是曼哈顿距离取决于数据的特点和具体问题。通常情况下，欧式距离更适合用于连续特征空间，而曼哈顿距离更适合用于具有稀疏特征或特征之间具有明显分隔的数据。在KNN算法中，我们可以根据实际情况选择合适的距离度量方法来获得更好的分类或回归结果。

## 实现

### Python实现

```python
import numpy as np
from collections import Counter


class KNNClassifier:
    def __init__(self, k=3):
        """
        Initialize the KNNClassifier.

        Parameters:
            k (int): Number of nearest neighbors to consider for classification. Default is 3.
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Train the KNN classifier.

        Parameters:
            X_train (numpy.ndarray): Training feature matrix.
            y_train (numpy.ndarray): Labels for the training data.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Parameters:
            point1 (numpy.ndarray): First data point.
            point2 (numpy.ndarray): Second data point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def predict(self, X_test, predict_type="classify"):
        """
        Predict class labels or regression values for the given input data.

        Parameters:
            X_test (numpy.ndarray): Input feature matrix for prediction.
            predict_type (str): Type of prediction, either "classify" or "regress". Default is "classify".

        Returns:
            numpy.ndarray: Predicted class labels or regression values.
        """
        if predict_type == "classify":
            y_pred = [self._predict_single_classify(x) for x in X_test]
            return np.array(y_pred)
        if predict_type == "regress":
            y_pred = [self._predict_single_regress(x) for x in X_test]
            return y_pred

    def _predict_single_classify(self, x):
        """
        Classify a single data point.

        Parameters:
            x (numpy.ndarray): Input data point for classification.

        Returns:
            int: Predicted class label for the input data point.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]  # Get indices of k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]  # Get class labels of k nearest neighbors
        most_common = Counter(k_nearest_labels).most_common(
            1)  # Find the most common class label among k nearest neighbors
        return most_common[0][0]

    def _predict_single_regress(self, x):
        """
        Make a regression prediction for a single data point.

        Parameters:
            x (numpy.ndarray): Input data point for regression prediction.

        Returns:
            float: Predicted regression value for the input data point.
        """
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]  # Get indices of k nearest neighbors
        k_nearest_y = [self.y_train[i] for i in k_indices]  # Get target values of k nearest neighbors
        return np.mean(k_nearest_y)  # Use the mean of nearest target values as the prediction
```

### 说明

类的使用方式

1. 导入该类：`from knn_classifier import KNNClassifier` (假设该类的定义保存在名为 `knn_classifier.py` 的文件中)
2. 创建KNN分类器对象：`knn = KNNClassifier(k=k_value)`，其中 `k_value` 是最近邻居的数量，可以选择性地指定，默认为 3。
3. 使用训练数据对分类器进行训练：`knn.fit(X_train, y_train)`，其中 `X_train` 是训练数据的特征矩阵，`y_train` 是对应的类标签。
4. 对新数据进行预测：`predictions = knn.predict(X_test, predict_type="classify")`，其中 `X_test` 是包含新数据点的特征矩阵，`predict_type` 可以选择 "classify" 或 "regress"，分别用于分类和回归预测。

方法说明

1. `__init__(self, k=3)`: 构造函数，初始化 KNN 分类器对象，可以传入最近邻居的数量 `k`，默认为 3。
2. `fit(self, X_train, y_train)`: 训练 KNN 分类器，将特征矩阵 `X_train` 和对应的类标签 `y_train` 用于模型训练。
3. `predict(self, X_test, predict_type="classify")`: 预测给定输入数据的类标签或回归值，根据 `predict_type` 参数选择分类预测或回归预测。
4. `_euclidean_distance(self, point1, point2)`: 计算两个数据点之间的欧几里得距离。
5. `_predict_single_classify(self, x)`: 对单个数据点进行分类预测，返回预测的类标签。
6. `_predict_single_regress(self, x)`: 对单个数据点进行回归预测，返回预测的回归值。

属性说明

1. `k`: 最近邻居的数量，在构造函数中初始化，用于 KNN 算法中指定最近的邻居数目。
2. `X_train`: 训练数据的特征矩阵，用于存储训练过程中的输入特征。
3. `y_train`: 训练数据的类标签，用于存储训练过程中的目标标签。

### 示例用法
```python
# 生成一些示例数据
X_train = np.array([[1, 2], [2, 3], [3, 4], [5, 1], [6, 2], [7, 3]])
y_train = np.array([0, 0, 0, 1, 1, 1])

knn = KNNClassifier(k=3)  # 创建KNN分类器对象
knn.fit(X_train, y_train)  # 使用示例数据进行训练

# 预测新数据点
X_test = np.array([[2, 1], [7, 4]])
predictions = knn.predict(X_test, predict_type="classify")
print("预测结果:", predictions)
```

以上代码展示了使用 `KNNClassifier` 类的示例用法。首先，我们创建一个包含两个类的示例数据集 `X_train` 和 `y_train`，然后创建了一个 `KNNClassifier` 对象，并使用示例数据进行训练。接着，我们使用新数据 `X_test` 对模型进行分类预测，并打印预测结果。

## 注意事项

1. **特征缩放**：KNN算法基于距离度量来进行分类或回归，因此特征的尺度会影响距离计算的结果。如果特征具有不同的尺度范围，建议进行特征缩放，确保各个特征对距离计算的影响相对均衡。常用的特征缩放方法包括标准化（z-score标准化）和归一化（min-max缩放）。

2. **K值的选择**：KNN算法中的K值决定了用于分类或回归的最近邻样本的数量。选择适当的K值对算法的性能至关重要。较小的K值可能会导致模型过拟合，过于敏感，容易受到噪声的影响。较大的K值可能会导致模型过于简单，忽略了局部的数据模式。通常，K值的选择通过交叉验证等技术来进行调优。

3. **距离度量方法的选择**：KNN算法需要选择适当的距离度量方法，例如欧氏距离、曼哈顿距离或其他距离度量方法。不同的距离度量方法适用于不同的数据类型和特征空间，因此在选择时需要根据数据的特点进行合理选择。

4. **数据量和计算复杂度**：KNN算法在大规模数据集上的计算复杂度较高，因为它需要计算新样本与所有训练样本之间的距离。在处理大规模数据时，需要考虑使用更高效的数据结构和算法来加速计算过程。

5. **处理不平衡数据集**：对于不平衡的数据集（某些类别样本数量较少），KNN算法可能会偏向于占主导地位的类别。这时可以考虑采用过采样、欠采样或合成新样本的方法来平衡数据集，以获得更好的分类结果。

6. **避免数据泄露**：在使用KNN算法时，应确保测试集和训练集的样本是独立的，避免数据泄露。数据泄露可能导致模型在测试集上的表现过于乐观，无法真实地评估算法的性能。

7. **模型的解释性**：KNN算法是一种基于实例的学习方法，它不生成显式的模型，因此模型的解释性较差。在某些应用场景中，模型的解释性可能是非常重要的，此时需要考虑其他类型的算法。

综上所述，KNN算法是一种简单而强大的分类和回归算法，在实践中有广泛的应用。但要获得好的性能，需要合理选择K值、距离度量方法，并注意数据预处理等注意事项。