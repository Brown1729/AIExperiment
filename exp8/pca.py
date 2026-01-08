import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PCAImageReconstruction:
    """PCA图像重建实验类"""

    def __init__(self, data_path=None):
        """
        初始化PCA图像重建实验

        参数:
            data_path: yalefaces数据集路径
        """
        self.data_path = data_path
        self.images = None
        self.images_flat = None
        self.original_images = None
        self.image_shape = None
        self.pca_model = None
        self.mean_vector = None
        self.eigenvectors = None
        self.eigenvalues = None

    def load_yale_faces(self, img_size=(100, 100), num_samples=100):
        """
        加载yalefaces数据集

        参数:
            img_size: 图像调整大小
        """
        if self.data_path is None:
            # 如果没有提供路径，创建一个简单的示例数据集
            print("警告: 没有提供数据集路径，使用随机生成的示例数据")
            self._create_sample_data(img_size)
            return

        print(f"正在从 {self.data_path} 加载yalefaces数据集...")

        image_files = []
        # 搜索常见的图像格式
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.pgm', '*.bmp', '*.tif']:
            image_files.extend(list(Path(self.data_path).rglob(ext)))

        if len(image_files) == 0:
            print("警告: 没有找到图像文件")
            assert (len(image_files) == 0)
            return

        images = []
        for img_file in image_files[:num_samples]:  # 限制加载数量以加速处理
            try:
                img = Image.open(img_file)
                img = img.convert('L')  # 转换为灰度
                img = img.resize(img_size)
                img_array = np.array(img, dtype=np.float32)
                images.append(img_array)
            except Exception as e:
                print(f"无法加载图像 {img_file}: {e}")

        if len(images) == 0:
            print("警告: 没有成功加载任何图像")
            assert (len(images) == 0)
            return

        self.images = np.array(images)
        self.image_shape = self.images[0].shape
        print(f"成功加载 {len(self.images)} 张图像，图像尺寸: {self.image_shape}")

    def preprocess_data(self):
        """数据预处理：归一化并展平"""
        if self.images is None:
            raise ValueError("请先加载数据")

        # 归一化到0-1范围
        self.original_images = self.images.copy()
        self.images = self.images / 255.0 if self.images.max() > 1.0 else self.images

        # 展平图像
        n_samples, height, width = self.images.shape
        self.images_flat = self.images.reshape(n_samples, height * width)

        print(f"数据预处理完成，形状: {self.images_flat.shape}")

        return self.images_flat

    def add_gaussian_noise(self, images, sigma):
        """
        添加高斯噪声

        参数:
            images: 输入图像
            sigma: 噪声标准差

        返回:
            带噪声的图像
        """
        noise = np.random.normal(0, sigma, images.shape)
        noisy_images = images + noise
        # 裁剪到0-1范围
        noisy_images = np.clip(noisy_images, 0, 1)
        return noisy_images

    def manual_pca_fit(self, X):
        """
        手动实现PCA训练

        参数:
            X: 输入数据，形状 (n_samples, n_features)
        """
        print("正在训练PCA模型...")

        # 计算均值
        self.mean_vector = np.mean(X, axis=0)

        # 中心化数据
        X_centered = X - self.mean_vector

        # 计算协方差矩阵
        # 使用(X^T X) / (n-1)
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 计算特征值和特征向量
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(cov_matrix)

        # 排序特征值和特征向量（降序）
        idx = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = self.eigenvectors[:, idx]

        print(f"PCA训练完成，特征值数量: {len(self.eigenvalues)}")

        return self.eigenvectors, self.eigenvalues

    def manual_pca_transform(self, X, n_components):
        """
        手动PCA变换

        参数:
            X: 输入数据
            n_components: 主成分数量

        返回:
            降维后的数据
        """
        if self.eigenvectors is None:
            raise ValueError("请先训练PCA模型")

        # 中心化数据
        X_centered = X - self.mean_vector

        # 投影到主成分空间
        components = self.eigenvectors[:, :n_components]
        X_transformed = np.dot(X_centered, components)

        return X_transformed

    def manual_pca_inverse_transform(self, X_transformed, n_components):
        """
        手动PCA逆变换

        参数:
            X_transformed: 降维后的数据
            n_components: 主成分数量

        返回:
            重建后的数据
        """
        if self.eigenvectors is None:
            raise ValueError("请先训练PCA模型")

        # 使用前n_components个主成分重建
        components = self.eigenvectors[:, :n_components]
        X_reconstructed = np.dot(X_transformed, components.T) + self.mean_vector

        return X_reconstructed

    def calculate_psnr(self, original, reconstructed, max_pixel=1.0):
        """
        计算峰值信噪比(PSNR)

        参数:
            original: 原始图像
            reconstructed: 重建图像
            max_pixel: 最大像素值

        返回:
            PSNR值
        """
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')

        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr

    def run_experiment(self, noise_levels=None, n_components_range=None):
        """
        运行完整实验

        参数:
            noise_levels: 噪声水平列表
            n_components_range: 主成分数量范围

        返回:
            实验结果
        """
        if noise_levels is None:
            noise_levels = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]

        if n_components_range is None:
            n_components_range = range(5, 201, 5)

        print("=" * 60)
        print("开始PCA图像重建实验")
        print(f"噪声水平: {noise_levels}")
        print(f"主成分数量范围: {list(n_components_range)}")
        print("=" * 60)

        # 预处理数据
        X = self.preprocess_data()

        # 训练PCA模型
        self.manual_pca_fit(X)

        # 存储PSNR结果
        psnr_results = {sigma: [] for sigma in noise_levels}

        # 对每个噪声水平进行实验
        for sigma in noise_levels:
            print(f"\n处理噪声水平 σ={sigma}...")

            # 添加噪声
            X_noisy = self.add_gaussian_noise(X, sigma)

            # 对每个主成分数量进行重建并计算PSNR
            for r in n_components_range:
                # PCA变换
                X_transformed = self.manual_pca_transform(X_noisy, r)

                # 重建
                X_reconstructed = self.manual_pca_inverse_transform(X_transformed, r)

                # 计算平均PSNR
                psnr_values = []
                for i in range(len(X)):
                    psnr_i = self.calculate_psnr(X[i], X_reconstructed[i])
                    psnr_values.append(psnr_i)

                avg_psnr = np.mean(psnr_values)
                psnr_results[sigma].append(avg_psnr)

                if r % 50 == 0 or r == max(n_components_range):
                    print(f"  r={r:3d}, 平均PSNR: {avg_psnr:.2f} dB")

        return psnr_results, X, X_noisy

    def plot_results(self, psnr_results, n_components_range=None):
        """
        绘制PSNR随主成分数量变化的曲线

        参数:
            psnr_results: PSNR结果
            n_components_range: 主成分数量范围
        """
        if n_components_range is None:
            n_components_range = range(5, 201, 5)

        plt.figure(figsize=(12, 8))

        for sigma, psnr_values in psnr_results.items():
            plt.plot(list(n_components_range), psnr_values,
                     marker='o', markersize=3, linewidth=2,
                     label=f'σ={sigma}')

        plt.xlabel('主成分数量 (r)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title('不同噪声水平下PSNR随主成分数量的变化', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # 保存图像
        plt.savefig('psnr_vs_components.png', dpi=300)
        plt.show()

        print("PSNR曲线图已保存为 'psnr_vs_components.png'")

    def visualize_reconstruction(self, X, X_noisy, sigma=0.05, n_components=20, n_examples=5):
        """
        可视化重建效果

        参数:
            X: 原始数据
            X_noisy: 带噪声数据
            sigma: 噪声水平
            n_components: 主成分数量
            n_examples: 显示示例数量
        """
        print(f"\n可视化重建效果: σ={sigma}, r={n_components}")

        # 重建带噪声图像
        X_transformed = self.manual_pca_transform(X_noisy, n_components)
        X_reconstructed = self.manual_pca_inverse_transform(X_transformed, n_components)

        # 重塑为图像形状
        height, width = self.image_shape
        original_images = X.reshape(-1, height, width)
        noisy_images = X_noisy.reshape(-1, height, width)
        reconstructed_images = X_reconstructed.reshape(-1, height, width)

        # 计算PSNR
        psnr_values = []
        for i in range(min(n_examples, len(original_images))):
            psnr_i = self.calculate_psnr(original_images[i], reconstructed_images[i])
            psnr_values.append(psnr_i)

        # 创建可视化
        fig, axes = plt.subplots(n_examples, 3, figsize=(12, 4 * n_examples))

        if n_examples == 1:
            axes = axes.reshape(1, -1)

        for i in range(min(n_examples, len(original_images))):
            # 原始图像
            axes[i, 0].imshow(original_images[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'原始图像 {i + 1}')
            axes[i, 0].axis('off')

            # 带噪声图像
            axes[i, 1].imshow(noisy_images[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'带噪声图像 (σ={sigma})')
            axes[i, 1].axis('off')

            # 重建图像
            axes[i, 2].imshow(reconstructed_images[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'重建图像 (r={n_components}, PSNR={psnr_values[i]:.2f} dB)')
            axes[i, 2].axis('off')

        plt.suptitle(f'PCA图像重建效果对比 (σ={sigma}, r={n_components})', fontsize=16)
        plt.tight_layout()

        # 保存图像
        plt.savefig(f'reconstruction_visualization_sigma{sigma}_r{n_components}.png', dpi=300)
        plt.show()

        print(f"重建效果可视化已保存为 'reconstruction_visualization_sigma{sigma}_r{n_components}.png'")
        print(f"示例图像的平均PSNR: {np.mean(psnr_values):.2f} dB")


# 主程序
def main():
    """主函数"""

    # 1. 初始化PCA图像重建实验
    print("初始化PCA图像重建实验...")

    # 如果您有yalefaces数据集，请将路径替换为实际路径
    # data_path = "path/to/yalefaces/dataset"
    data_path = "./yalefaces"  # 设置为None以使用示例数据

    experiment = PCAImageReconstruction(data_path)

    # 2. 加载数据
    experiment.load_yale_faces(img_size=(100, 100), num_samples=1102)

    # 3. 运行实验
    noise_levels = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1]
    n_components_range = range(5, 201, 5)

    psnr_results, X, X_noisy = experiment.run_experiment(
        noise_levels=noise_levels,
        n_components_range=n_components_range
    )

    # 4. 绘制PSNR曲线
    print("\n绘制PSNR曲线...")
    experiment.plot_results(psnr_results, n_components_range)

    # 5. 可视化特定条件下的重建效果
    print("\n可视化重建效果...")
    # 为σ=0.05创建带噪声的数据
    X_noisy_specific = experiment.add_gaussian_noise(X, sigma=0.05)
    experiment.visualize_reconstruction(
        X, X_noisy_specific,
        sigma=0.05, n_components=20, n_examples=5
    )

    # 6. 保存实验结果
    print("\n保存实验结果...")
    results_summary = {
        'noise_levels': noise_levels,
        'n_components_range': list(n_components_range),
        'psnr_results': psnr_results,
        'image_shape': experiment.image_shape,
        'n_samples': len(experiment.images)
    }

    # 可以将结果保存为npy文件
    np.save('pca_experiment_results.npy', results_summary)
    print("实验结果已保存为 'pca_experiment_results.npy'")

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

    # 7. 打印一些关键发现
    print("\n关键发现:")
    for sigma in noise_levels:
        best_r_idx = np.argmax(psnr_results[sigma])
        best_r = list(n_components_range)[best_r_idx]
        best_psnr = psnr_results[sigma][best_r_idx]
        print(f"σ={sigma}: 最佳r={best_r}, 最大PSNR={best_psnr:.2f} dB")


if __name__ == "__main__":
    # 运行主实验
    main()
