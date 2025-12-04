import random
import numpy as np


class NaiveBayes:
    def __init__(self, laplace_smoothing=False, alpha=1):
        """
        初始化朴素贝叶斯分类器
        :param laplace_smoothing: 是否使用拉普拉斯平滑
        """
        self.laplace_smoothing = laplace_smoothing

        # 存储训练得到的参数
        self.classes = None  # 类别标签
        self.priors = None  # 先验概率
        self.conditionals = None  # 条件概率

        # 超参，laplace平滑系数
        self.alpha = alpha

    def train(self, X, y):
        """
        训练模型
        :param X: 特征矩阵，n_samples × n_features
        :param y: 标签向量，n_samples
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # 只有三个类别
        n_classes = len(self.classes)

        # 计算先验概率 P(c)
        self.priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            if self.laplace_smoothing:
                self.priors[i] = (np.sum(y == c) + 1) / (n_samples + n_classes)  # 拉普拉斯平滑
            else:
                self.priors[i] = np.mean(y == c)

        # 计算条件概率 P(x_i|c)
        self.conditionals = np.zeros((n_classes, n_features))
        # conditionals 中存储的是有该类含有该特征的概率，如果不含该特征，则应该乘上1减去这个值

        for i, c in enumerate(self.classes):
            X_c = X[y == c]  # 获取 C 的所有样本（ y==c 构造一个mask）
            n_c = X_c.shape[0]  # C 的样本数量

            for j in range(n_features):
                if self.laplace_smoothing:
                    self.conditionals[i, j] = (np.sum(X_c[:, j] == 1) + 1 * self.alpha) / (
                            n_c + 2 * self.alpha)  # 拉普拉斯平滑（只有两个属性）
                else:
                    # 标准朴素贝叶斯
                    count = np.sum(X_c[:, j] == 1)
                    # 避免零概率问题（如果count为0，则概率为0，可能导致后续计算问题）
                    if count == 0:
                        self.conditionals[i, j] = 1e-10  # 一个很小的正数
                    elif count == n_c:
                        self.conditionals[i, j] = 1 - 1e-10  # 一个非常接近1的数
                    else:
                        self.conditionals[i, j] = count / n_c
        return self

    def predict(self, X):
        """
        预测 P = P(c) * P(x_1|c) * P(x_2|c) * ... * P(x_n|c)
        :param X: 特征矩阵，n_samples × n_features
        :return: 预测结果
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        # 每个样本属于每个类别的概率
        probs = np.zeros((n_samples, len(self.classes)))

        for i in range(n_samples):
            for j, c in enumerate(self.classes):
                probs[i, j] = self.priors[j]
                for k in range(X.shape[1]):
                    # X[i, k] == 1 表示该样本的第 k 个特征为 1
                    if X[i, k] == 1:
                        probs[i, j] *= self.conditionals[j, k]
                    else:
                        probs[i, j] *= 1 - self.conditionals[j, k]

        # 选择概率最大的类别
        predictions = self.classes[np.argmax(probs, axis=1)]

        return predictions

    def score(self, X, y):
        """
        计算准确率
        :param X: 特征矩阵
        :param y: 真实标签
        :return: 准确率
        """
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def find_best_alpha(self, X_train, y_train, X_val, y_val, alpha_list):
        best_val_accuracy = 0
        best_alpha = 0
        for index, alpha in enumerate(alpha_list):
            print(f"正在进行第 {index + 1} 训练")
            self.alpha = alpha
            self.train(X_train, y_train)
            val_accuracy = self.score(X_val, y_val)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_alpha = alpha
            print(f"alpha: {alpha}, val_accuracy: {val_accuracy:.4f}")

        return best_alpha, best_val_accuracy


def load_dna_data(filename):
    """
    加载DNA数据
    （标准的LIBSVM数据会出现小数，而dna数据只分有和没有，这里用离散的贝叶斯分类方式）
    :param filename: 数据文件名
    :return: 特征矩阵和标签向量
    """
    features = []
    labels = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割标签和特征
            parts = line.split()
            label = int(parts[0]) - 1  # 将标签从1,2,3转换为0,1,2
            feature_vector = np.zeros(180)  # 180个属性

            for part in parts[1:]:
                idx, val = part.split(':')
                feature_vector[int(idx) - 1] = float(val)  # 属性索引从1开始

            features.append(feature_vector)
            labels.append(label)

    return np.array(features), np.array(labels)


# 主程序
if __name__ == "__main__":
    # 加载数据
    X_train, y_train = load_dna_data(r"D:\Program\python\AIexp\dna\dna.scale.tr")
    X_val, y_val = load_dna_data(r"D:\Program\python\AIexp\dna\dna.scale.val")
    X_test, y_test = load_dna_data(r"D:\Program\python\AIexp\dna\dna.scale.t")

    # 标准朴素贝叶斯
    print("训练标准朴素贝叶斯...")
    nb = NaiveBayes(laplace_smoothing=False)
    nb.train(X_train, y_train)

    # 验证集评估
    val_accuracy = nb.score(X_val, y_val)
    print(f"验证集准确率（标准朴素贝叶斯）: {val_accuracy:.4f}")

    # 测试集评估
    test_accuracy = nb.score(X_test, y_test)
    print(f"测试集准确率（标准朴素贝叶斯）: {test_accuracy:.4f}")

    # 拉普拉斯修正的朴素贝叶斯
    print("\n训练拉普拉斯修正的朴素贝叶斯...")
    nb_laplace = NaiveBayes(laplace_smoothing=True)
    nb_laplace.train(X_train, y_train)

    # 验证集评估
    val_accuracy_laplace = nb_laplace.score(X_val, y_val)
    print(f"验证集准确率（拉普拉斯修正）: {val_accuracy_laplace:.4f}")

    # 测试集评估
    test_accuracy_laplace = nb_laplace.score(X_test, y_test)
    print(f"测试集准确率（拉普拉斯修正）: {test_accuracy_laplace:.4f}")

    # 调整α值
    print("\n调整α值...")
    alpha_list = [0.9, 0.8, 0.85, 0.7, 0.6]
    best_alpha, best_val_accuracy = nb_laplace.find_best_alpha(X_train, y_train, X_val, y_val, alpha_list)
    print(f"最佳α值: {best_alpha}, 验证集准确率: {best_val_accuracy:.4f}")

    # 测试集评估
    test_accuracy_laplace = nb_laplace.score(X_test, y_test)
    print(f"测试集准确率（拉普拉斯修正）: {test_accuracy_laplace:.4f}")
