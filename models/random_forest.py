import numpy as np
from collections import Counter

class DecisionTree:
    """自定义决策树分类器"""
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # 特征随机选择数量
        self.tree = None  # 存储决策树结构

    def fit(self, X, y):
        """训练决策树"""
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.tree = self._build_tree(X, y)
        
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_feats = X.shape
        label_counts = Counter(y)
        
        # 停止条件
        if (depth >= self.max_depth or n_samples < self.min_samples_split or len(label_counts) == 1):
            leaf_value = self._most_common_label(y)
            return leaf_value
            
        # 随机选择特征子集
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # 寻找最佳分割
        best_gain, best_threshold, best_idx = -1, None, None
        for feat_idx in feat_idxs:
            unique_vals = np.unique(X[:, feat_idx])
            for threshold in unique_vals:
                gain = self._information_gain(X[:, feat_idx], y, threshold)
                if gain > best_gain:
                    best_gain, best_threshold, best_idx = gain, threshold, feat_idx
        
        # 如果无法提升则返回叶节点
        if best_gain == 0:
            return self._most_common_label(y)
            
        # 分割数据集
        left_idxs = X[:, best_idx] <= best_threshold
        right_idxs = X[:, best_idx] > best_threshold
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return (best_idx, best_threshold, left, right)
    
    def _most_common_label(self, y):
        """获取出现次数最多的标签"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _information_gain(self, X_col, y, threshold):
        """计算信息增益"""
        parent_entropy = self._entropy(y)
        
        left_mask = X_col <= threshold
        right_mask = X_col > threshold
        
        n_left, n_right = len(left_mask), len(right_mask)
        if n_left == 0 or n_right == 0:
            return 0
            
        e_left = self._entropy(y[left_mask])
        e_right = self._entropy(y[right_mask])
        
        child_entropy = (n_left * e_left + n_right * e_right) / (n_left + n_right)
        return parent_entropy - child_entropy
    
    def _entropy(self, y):
        """计算熵"""
        counts = np.bincount(y.astype(int))
        probs = counts / len(y)
        return -np.sum(p * np.log2(p) for p in probs if p > 0)
    
    def predict(self, X):
        """预测单个样本"""
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _traverse_tree(self, x, node):
        """递归遍历树"""
        if not isinstance(node, tuple):  # 叶节点
            return node
            
        idx, threshold, left, right = node
        if x[idx] <= threshold:
            return self._traverse_tree(x, left)
        else:
            return self._traverse_tree(x, right)


class RandomForestClassifier:
    """自定义随机森林分类器"""
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        
    def fit(self, X, y):
        """训练随机森林"""
        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            
            # 自助采样
            idxs = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[idxs]
            y_bootstrap = y[idxs]
            
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        return self
        
    def predict(self, X):
        """预测函数"""
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 投票选择最终结果
        return np.apply_along_axis(
            lambda x: Counter(x).most_common(1)[0][0],
            axis=0,
            arr=tree_preds
        )