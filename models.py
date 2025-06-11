import numpy as np


class MultinomialNB:
    """手动实现的多项式朴素贝叶斯分类器"""

    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        alpha: 拉普拉斯平滑参数，默认为1.0
        """
        self.alpha = alpha
        self.classes = None
        self.class_priors = None
        self.feature_probs = None
        self.n_features = None

    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组 (n_samples,)
        """
        # 转换为numpy数组
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)
        y = np.array(y)

        # 获取类别信息
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_samples, self.n_features = X.shape

        # 计算类先验概率 P(y)
        self.class_priors = {}
        for class_label in self.classes:
            class_count = np.sum(y == class_label)
            self.class_priors[class_label] = class_count / n_samples

        # 计算特征条件概率 P(x_i|y)
        self.feature_probs = {}
        for class_label in self.classes:
            # 获取当前类别的所有样本
            class_mask = (y == class_label)
            class_samples = X[class_mask]

            # 计算每个特征在当前类别下的词频总和
            feature_counts = np.sum(class_samples, axis=0)

            # 应用拉普拉斯平滑
            # P(x_i|y) = (count(x_i, y) + alpha) / (sum(count(x_j, y)) + alpha * n_features)
            total_count = np.sum(feature_counts)
            smoothed_probs = (feature_counts + self.alpha) / (total_count + self.alpha * self.n_features)

            self.feature_probs[class_label] = smoothed_probs

        return self

    def predict_proba(self, X):
        """
        预测每个类别的概率
        X: 特征矩阵 (n_samples, n_features)
        返回: 概率矩阵 (n_samples, n_classes)
        """
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        probabilities = np.zeros((n_samples, n_classes))

        for i, class_label in enumerate(self.classes):
            # 计算类先验概率的对数
            log_prior = np.log(self.class_priors[class_label])

            # 计算特征条件概率的对数
            feature_log_probs = np.log(self.feature_probs[class_label])

            # 对于每个样本，计算 log P(y) + sum(x_j * log P(x_j|y))
            log_likelihood = np.dot(X, feature_log_probs)

            probabilities[:, i] = log_prior + log_likelihood

        # 使用log-sum-exp技巧避免数值下溢
        max_log_prob = np.max(probabilities, axis=1, keepdims=True)
        exp_probs = np.exp(probabilities - max_log_prob)
        normalized_probs = exp_probs / np.sum(exp_probs, axis=1, keepdims=True)

        return normalized_probs

    def predict(self, X):
        """
        预测类别标签
        X: 特征矩阵 (n_samples, n_features)
        返回: 预测标签数组 (n_samples,)
        """
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]