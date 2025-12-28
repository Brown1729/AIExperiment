import random
import numpy as np
import math
from collections import Counter


def load_dna_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    加载DNA数据集
    因为数据集的属性是 1-180 的数字，而值总是 1，所以将DNA数据集转换为01矩阵
    而标签则写成列向量的形式
    这样虽然增加了空间复杂度，但是代码更加简洁
    :param filename:
    :return:
    """
    features = []  # 存储特征向量
    labels = []  # 存储标签

    # 打开文件读取数据
    with open(filename, 'r') as f:
        # 逐行读取文件内容
        for line in f:
            line = line.strip()  # 去除行首尾的空白字符
            if not line:  # 跳过空行
                continue

            # 分割标签和特征
            parts = line.split()  # 按空格分割行内容
            label = int(parts[0]) - 1  # 将标签从1,2,3转换为0,1,2
            feature_vector = np.zeros(180)  # 创建180维的特征向量，初始化为0

            # 处理特征部分（从第2个元素开始）
            for part in parts[1:]:
                idx, val = part.split(':')  # 分割特征索引和值（索引:值）
                feature_vector[int(idx) - 1] = float(val)  # 属性索引从1开始，转换为0开始索引

            # 添加到列表
            features.append(feature_vector)
            labels.append(label)

    return np.array(features), np.array(labels)  # 转成 np.ndarray 便于计算


# 决策树节点类
class TreeNode:
    def __init__(self, feature_idx=None, threshold=None, value=None, left=None, right=None):
        self.feature_idx = feature_idx  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.value = value  # 叶节点的预测值
        self.left = left  # 左子树
        self.right = right  # 右子树


# 通用决策树实现
class DecisionTree:
    def __init__(self, algorithm='cart', max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=None):
        self.algorithm = algorithm  # 'id3', 'c45', 'cart'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.root = None
        self.tree_depth = 0
        self.leaf_nodes = 0

        if random_state:
            np.random.seed(random_state)

    def _information_entropy(self, y):
        """
        计算信息熵
        :param y:
        :return:
        """
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probs = class_counts / len(y)
        entropy = 0
        for prob in class_probs:
            if prob > 0:  # 0 log_2(0) = 0
                entropy -= prob * math.log2(prob)
        return entropy

    def _gini(self, y):
        """
        计算基尼值
        :param y:
        :return:
        """
        if len(y) == 0:
            return 0
        class_counts = np.bincount(y)
        class_probs = class_counts / len(y)
        return 1 - np.sum(class_probs ** 2)  # 计算基尼值：1 - Σ(p_i^2)

    def _information_gain(self, y, y_left, y_right):
        """
        计算信息增益（二分类）
        :param y:
        :param y_left:
        :param y_right:
        :return:
        """
        # 父节点的熵
        parent_entropy = self._information_entropy(y)
        # 计算子节点的加权熵
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n == 0:
            return 0
        child_entropy = (n_left / n) * self._information_entropy(y_left) + (n_right / n) * self._information_entropy(
            y_right)
        # 信息增益
        return parent_entropy - child_entropy

    def _gain_ratio(self, y, y_left, y_right):
        """
        计算增益率（二分类）
        :param y:
        :param y_left:
        :param y_right:
        :return:
        """
        # 信息增益
        information_gain = self._information_gain(y, y_left, y_right)
        # 分裂信息
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        # 固有值
        intrinsic_value = 0
        if n_left > 0:
            intrinsic_value -= (n_left / n) * math.log2(n_left / n)
        if n_right > 0:
            intrinsic_value -= (n_right / n) * math.log2(n_right / n)
        # 避免除以0
        if intrinsic_value == 0:
            return 0
        # 增益率
        return information_gain / intrinsic_value

    def _gini_index(self, y, y_left, y_right):
        """
        计算基尼指数（二分类）
        :param y:
        :param y_left:
        :param y_right:
        :return:
        """
        # 计算子节点的加权基尼不纯度
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        if n == 0:
            return 0
        gini_index = (n_left / n) * self._gini(y_left) + (n_right / n) * self._gini(y_right)
        # 基尼增益
        return gini_index

    def _calculate_split_criterion(self, y, y_left, y_right):
        """根据选择的算法计算分裂准则"""
        if self.algorithm == 'id3':
            return self._information_gain(y, y_left, y_right)
        elif self.algorithm == 'c45':
            return self._gain_ratio(y, y_left, y_right)
        elif self.algorithm == 'cart':
            return self._gini_index(y, y_left, y_right)
        else:
            raise ValueError(f"不支持的算法: {self.algorithm}")

    def _best_split(self, X, y):
        """找到最佳分裂特征和阈值"""
        best_criterion = -float('inf')
        best_feature = None
        best_threshold = None

        n_samples, n_features = X.shape

        if self.algorithm == "c45":
            # 对于C4.5算法，先从候选划分属性中找出信息增益高于平均水平的属性，再从中选择增益率最高的
            information_gain_list = []
            for feature_idx in range(n_features):
                threshold = 0.5
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                information_gain_list.append(self._information_gain(y, y[left_mask], y[right_mask]))

            average_information_gain = np.mean(information_gain_list)

            over_average_feature_indexes = [feature_idx for feature_idx, gain in enumerate(information_gain_list) if
                                            gain > average_information_gain]

            for feature_idx in over_average_feature_indexes:
                threshold = 0.5
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                # 根据选择的算法计算分裂准则
                criterion = self._gain_ratio(y, y[left_mask], y[right_mask])

                if criterion > best_criterion:
                    best_criterion = criterion
                    best_feature = feature_idx
                    best_threshold = threshold

        elif self.algorithm == "id3":
            for feature_idx in range(n_features):
                # 对于二值特征，只需要考虑0和1作为阈值
                threshold = 0.5  # 由于是二值特征，用0.5作为阈值
                # 根据阈值分割数据
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # 根据选择的算法计算分裂准则
                criterion = self._information_gain(y, y[left_mask], y[right_mask])

                if criterion > best_criterion:
                    best_criterion = criterion
                    best_feature = feature_idx
                    best_threshold = threshold

        elif self.algorithm == "cart":
            best_criterion = float('inf')  # 初始化最佳基尼不纯度为无穷大

            for feature_idx in range(n_features):
                # 对于二值特征，只需要考虑0和1作为阈值
                threshold = 0.5  # 由于是二值特征，用0.5作为阈值
                # 根据阈值分割数据
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue

                # 根据选择的算法计算分裂准则
                criterion = self._gini_index(y, y[left_mask], y[right_mask])

                # 基尼指数越小，数据的纯度越高
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_criterion

    def _generate_tree(self, X, y, depth=0):
        """
        构建决策树
        :param X:
        :param y:
        :param depth:
        :return:
        """
        n_samples, n_features = X.shape

        # 终止条件检查，这里将书上的（1）（2）合并
        if (depth >= self.max_depth or  # 达到最大深度
                n_samples < self.min_samples_split or  # 样本数小于分裂要求
                len(np.unique(y)) == 1):  # 所有样本属于同一类别
            # 创建叶结点
            leaf_node = TreeNode(value=Counter(y).most_common(1)[0][0])
            self.leaf_nodes += 1
            self.tree_depth = max(self.tree_depth, depth)
            return leaf_node

        # 寻找最佳分裂
        feature_idx, threshold, criterion = self._best_split(X, y)

        # 如果没有找到合适的分裂，返回叶节点
        if feature_idx is None:
            leaf_node = TreeNode(value=Counter(y).most_common(1)[0][0])
            self.leaf_nodes += 1
            self.tree_depth = max(self.tree_depth, depth)
            return leaf_node

        # 根据分裂分割数据（由于是二分类，所以只需要对左子树取反就可得到右子树）
        left_mask = X[:, feature_idx] <= threshold  # 左子树样本
        right_mask = ~left_mask  # 右子树样本

        # 递归构建左右子树
        left_subtree = self._generate_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._generate_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_idx=feature_idx, threshold=threshold,
                        left=left_subtree, right=right_subtree)

    def fit(self, X, y, X_val=None, y_val=None):
        """训练决策树，可选择使用验证集进行剪枝"""
        # 重置树统计信息
        self.tree_depth = 0
        self.leaf_nodes = 0

        # 构建树
        print(f"构建决策树 (算法: {self.algorithm})...")
        self.root = self._generate_tree(X, y)

        return self

    def _predict_sample(self, x, node):
        """预测单个样本"""
        if node is None:
            return 0  # 默认值
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        """预测"""
        return np.array([self._predict_sample(x, self.root) for x in X])

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_tree_info(self):
        """获取树的信息"""
        return {
            'depth': self.tree_depth,
            'leaf_nodes': self.leaf_nodes
        }


# 随机森林实现
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=None):
        """随机森林初始化"""
        self.n_estimators = n_estimators  # 树的数量
        self.max_depth = max_depth  # 每棵树的最大深度
        self.min_samples_split = min_samples_split  # 节点分裂所需的最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶节点所需的最小样本数
        self.max_features = max_features  # 每棵树使用的特征数量策略
        self.random_state = random_state  # 随机种子
        self.trees = []  # 存储所有决策树

        # 设置随机种子
        if random_state:
            random.seed(random_state)
            np.random.seed(random_state)

    def _bootstrap_samples(self, X, y):
        """生成自助样本（有放回抽样）"""
        n_samples = X.shape[0]  # 样本数量
        indices = np.random.choice(n_samples, n_samples, replace=True)  # 随机选择索引（有放回）
        return X[indices], y[indices]  # 返回自助样本

    def _get_feature_subset(self, n_features):
        """获取特征子集"""
        # 根据策略确定选择的特征数量
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))  # 平方根策略
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))  # 对数策略
        elif isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)  # 比例策略
        else:
            n_selected = self.max_features  # 固定数量策略

        # 确保选择的特征数量在合理范围内
        n_selected = max(1, min(n_selected, n_features))
        # 随机选择特征索引
        return np.random.choice(n_features, n_selected, replace=False)

    def fit(self, X, y, algorithms=['cart', 'id3', 'c45']):
        """训练随机森林"""
        self.trees = []  # 清空树列表
        n_samples, n_features = X.shape  # 获取样本数和特征数
        # 训练每棵树
        for i in range(self.n_estimators):
            # 生成自助样本
            X_bootstrap, y_bootstrap = self._bootstrap_samples(X, y)
            algorithm = random.choice(algorithms)
            # 创建决策树，限制特征选择
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                algorithm=algorithm
            )
            # 训练决策树
            tree.fit(X_bootstrap, y_bootstrap)
            # 添加到树列表
            self.trees.append(tree)
            # 打印训练进度
            if (i + 1) % 10 == 0:
                print(f"已训练 {i + 1}/{self.n_estimators} 棵树")
        return self

    def predict(self, X):
        """预测"""
        # 收集所有树的预测
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # 多数投票
        final_predictions = []
        for sample_idx in range(X.shape[0]):
            votes = tree_predictions[:, sample_idx]  # 所有树对该样本的预测
            # 选择票数最多的类别
            final_predictions.append(Counter(votes).most_common(1)[0][0])

        return np.array(final_predictions)

    def score(self, X, y):
        """计算准确率"""
        predictions = self.predict(X)  # 获取预测结果
        return np.mean(predictions == y)  # 计算准确率


if __name__ == '__main__':
    # 加载数据
    print("正在加载DNA数据集...")
    X_train, y_train = load_dna_data('dna.scale.tr')  # 训练集
    X_val, y_val = load_dna_data('dna.scale.val')  # 验证集
    X_test, y_test = load_dna_data('dna.scale.t')  # 测试集

    # 打印数据集信息
    print(f"训练集: {X_train.shape}, 标签: {y_train.shape}")
    print(f"验证集: {X_val.shape}, 标签: {y_val.shape}")
    print(f"测试集: {X_test.shape}, 标签: {y_test.shape}")
    print(f"类别分布 - 训练集: {np.bincount(y_train)}, 验证集: {np.bincount(y_val)}, 测试集: {np.bincount(y_test)}")

    # 1. 决策树分类
    print("\n" + "=" * 50)
    print("1. 决策树分类")
    print("=" * 50)

    # 超参数调优
    max_depths = [5, 10, 15]  # 最大深度候选值
    min_samples_splits = [2, 5, 10]  # 最小分裂样本数候选值
    algorithms = ['cart', 'id3', 'c45']  # 算法候选值

    best_dt_val_score = 0  # 最佳验证集准确率
    best_dt_params = {}  # 最佳参数
    best_dt = None  # 最佳决策树模型

    print("在验证集上调整决策树超参数...")
    # 网格搜索寻找最佳参数
    for algorithm in algorithms:
        for max_depth in max_depths:
            for min_samples_split in min_samples_splits:
                # 创建决策树模型
                dt = DecisionTree(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=1,
                    algorithm=algorithm
                )
                dt.fit(X_train, y_train)  # 训练模型
                val_score = dt.score(X_val, y_val)  # 验证集准确率

                # 更新最佳模型
                if val_score > best_dt_val_score:
                    best_dt_val_score = val_score
                    best_dt_params = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'best_algorithm': algorithm
                    }
                    best_dt = dt

    # 打印最佳参数
    print(f"最佳验证集准确率: {best_dt_val_score:.4f}")
    print(f"最佳参数: {best_dt_params}")

    # 计算准确率
    train_acc_dt = best_dt.score(X_train, y_train)  # 训练集准确率
    val_acc_dt = best_dt.score(X_val, y_val)  # 验证集准确率
    test_acc_dt = best_dt.score(X_test, y_test)  # 测试集准确率

    # 打印性能结果
    print(f"\n决策树性能:")
    print(f"训练集准确率: {train_acc_dt:.4f}")
    print(f"验证集准确率: {val_acc_dt:.4f}")
    print(f"测试集准确率: {test_acc_dt:.4f}")

    # 2. 随机森林分类
    print("\n" + "=" * 50)
    print("2. 随机森林分类")
    print("=" * 50)

    # 超参数调优
    n_estimators_list = [50, 100]  # 树的数量候选值
    max_depths_rf = [10, 15]  # 最大深度候选值
    max_features_list = ['sqrt', 0.5]  # 特征选择策略候选值

    best_rf_val_score = 0  # 最佳验证集准确率
    best_rf_params = {}  # 最佳参数
    best_rf = None  # 最佳随机森林模型

    print("在验证集上调整随机森林超参数...")
    # 网格搜索寻找最佳参数
    for n_estimators in n_estimators_list:
        for max_depth in max_depths_rf:
            for max_features in max_features_list:
                # 创建随机森林模型
                rf = RandomForest(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features=max_features,
                    random_state=42  # 设置随机种子保证可重复性
                )
                rf.fit(X_train, y_train)  # 训练模型
                val_score = rf.score(X_val, y_val)  # 验证集准确率

                # 更新最佳模型
                if val_score > best_rf_val_score:
                    best_rf_val_score = val_score
                    best_rf_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'max_features': max_features
                    }
                    best_rf = rf

    # 打印最佳参数
    print(f"最佳验证集准确率: {best_rf_val_score:.4f}")
    print(f"最佳参数: {best_rf_params}")

    # 计算准确率
    train_acc_rf = best_rf.score(X_train, y_train)  # 训练集准确率
    val_acc_rf = best_rf.score(X_val, y_val)  # 验证集准确率
    test_acc_rf = best_rf.score(X_test, y_test)  # 测试集准确率

    # 打印性能结果
    print(f"\n随机森林性能:")
    print(f"训练集准确率: {train_acc_rf:.4f}")
    print(f"验证集准确率: {val_acc_rf:.4f}")
    print(f"测试集准确率: {test_acc_rf:.4f}")
