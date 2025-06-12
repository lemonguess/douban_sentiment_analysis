import numpy as np
from collections import Counter

class DecisionTree:
    """手动实现的决策树分类器"""
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        """
        初始化决策树分类器
        max_depth: 树的最大深度
        min_samples_split: 分裂所需的最小样本数
        criterion: 不纯度衡量标准（gini或entropy）
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
    
    def fit(self, X, y):
        """训练决策树"""
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)
        return self
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples = X.shape[0]
        
        # 终止条件
        if (depth == self.max_depth or n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            leaf_value = self._most_common_class(y)
            return {'type': 'leaf', 'class': leaf_value}
        
        # 找到最佳分裂
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            leaf_value = self._most_common_class(y)
            return {'type': 'leaf', 'class': leaf_value}
        
        # 分裂数据集
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth+1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth+1)
        
        return {
            'type': 'split',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _find_best_split(self, X, y):
        """寻找最佳分裂特征和阈值"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(self.n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.unique(feature_values)
            
            for threshold in thresholds:
                gain = self._calculate_gain(feature_idx, threshold, X, y)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _calculate_gain(self, feature_idx, threshold, X, y):
        """计算信息增益"""
        # 分割数据
        left_indices = X[:, feature_idx] <= threshold
        right_indices = ~left_indices
        
        if sum(left_indices) == 0 or sum(right_indices) == 0:
            return 0
        
        # 计算基尼指数或熵
        if self.criterion == 'gini':
            impurity_func = self._gini_impurity
        else:
            impurity_func = self._entropy
            
        parent_impurity = impurity_func(y)
        
        left_impurity = impurity_func(y[left_indices])
        right_impurity = impurity_func(y[right_indices])
        
        # 计算信息增益
        n = len(y)
        gain = parent_impurity - (len(y[left_indices])/n * left_impurity + 
                                 len(y[right_indices])/n * right_impurity)
        return gain
    
    def _gini_impurity(self, y):
        """计算基尼不纯度"""
        counts = np.array(list(Counter(y).values()))
        proportions = counts / len(y)
        return 1 - np.sum(proportions**2)
    
    def _entropy(self, y):
        """计算熵"""
        counts = np.array(list(Counter(y).values()))
        proportions = counts / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-10))
    
    def _most_common_class(self, y):
        """返回出现次数最多的类别"""
        return Counter(y).most_common(1)[0][0]
    
    def predict(self, X):
        """预测样本类别"""
        predictions = np.array([self._predict_sample(x) for x in X])
        return predictions
    
    def _predict_sample(self, x):
        """预测单个样本类别"""
        node = self.tree
        while node['type'] == 'split':
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['class']