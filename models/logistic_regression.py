import numpy as np

class LogisticRegression:
    """手动实现的逻辑回归分类器"""
    
    def __init__(self, learning_rate=0.01, num_iterations=10, reg_lambda=0.01):
        """
        初始化逻辑回归分类器
        learning_rate: 学习率
        num_iterations: 迭代次数
        reg_lambda: L2正则化系数
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """训练模型"""
        # 转换为numpy数组
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.random.randn(n_features, 1) * 0.01
        self.bias = 0
        
        # 梯度下降
        for i in range(self.num_iterations):
            # 前向传播
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)
            
            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y)) + (self.reg_lambda / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 打印训练信息（可选）
            # if i % 100 == 0:
            print(f"当前训练轮数:【{i+1}/{self.num_iterations}】")
            loss = self._compute_loss(y, y_pred)
            print(f"Iteration {i}, Loss: {loss:.4f}")
        
        return self
    
    def _sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))
    
    def _compute_loss(self, y_true, y_pred):
        """计算带L2正则化的交叉熵损失"""
        n_samples = y_true.shape[0]
        epsilon = 1e-15  # 防止log(0)
        loss = -(1 / n_samples) * np.sum(
            y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        reg_term = (self.reg_lambda / (2 * n_samples)) * np.sum(np.square(self.weights))
        return loss + reg_term
    
    def predict_proba(self, X):
        """预测概率"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)