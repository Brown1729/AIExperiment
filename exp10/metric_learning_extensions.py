from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _dataset_labels(dataset: Any) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("数据集缺少 targets 属性，无法做分层采样。")
    if torch.is_tensor(targets):
        return targets.cpu().numpy()
    return np.asarray(targets)


def make_stratified_indices(dataset: Any, max_samples: Optional[int], seed: int = 42) -> np.ndarray:
    labels = _dataset_labels(dataset)
    total = len(labels)
    if max_samples is None or max_samples >= total:
        return np.arange(total)

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    per_class = max_samples // len(classes)
    remainder = max_samples % len(classes)
    sampled = []

    for offset, class_id in enumerate(classes):
        class_indices = np.where(labels == class_id)[0].copy()
        rng.shuffle(class_indices)
        take = min(len(class_indices), per_class + (1 if offset < remainder else 0))
        sampled.append(class_indices[:take])

    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def make_loader(
    dataset: Any,
    indices: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    subset = Subset(dataset, indices.astype(int).tolist())
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_center_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += data.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": 100.0 * correct / max(total, 1),
    }


@torch.no_grad()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features_list = []
    labels_list = []

    for data, target in loader:
        data = data.to(device)
        _, features = model(data, return_feature=True)
        features_list.append(features.cpu().numpy())
        labels_list.append(target.cpu().numpy())

    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def train_centerloss_model(
    model_cls: Callable[..., nn.Module],
    center_loss_cls: Callable[..., nn.Module],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    feature_dim: int = 2,
    lambda_center: float = 0.1,
    epochs: int = 4,
    lr: float = 1e-3,
    center_lr: float = 0.5,
    num_classes: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[nn.Module, nn.Module, list[Dict[str, float]]]:
    set_seed(seed)

    model = model_cls(feature_dim=feature_dim).to(device)
    center_loss_fn = center_loss_cls(num_classes=num_classes, feat_dim=feature_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=center_lr)

    history = []

    for epoch in range(epochs):
        model.train()
        center_loss_fn.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            center_optimizer.zero_grad()

            output, features = model(data, return_feature=True)
            loss_ce = criterion(output, target)
            loss_center = center_loss_fn(features, target)
            loss = loss_ce + lambda_center * loss_center

            loss.backward()
            optimizer.step()
            center_optimizer.step()

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

        train_metrics = {
            "train_loss": total_loss / max(total, 1),
            "train_acc": 100.0 * correct / max(total, 1),
        }
        test_metrics = evaluate_center_model(model, test_loader, criterion, device)

        record = {
            **train_metrics,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        history.append(record)

        if verbose:
            print(
                f"Epoch {epoch + 1:2d}/{epochs} | "
                f"Train Loss: {record['train_loss']:.4f} Acc: {record['train_acc']:.2f}% | "
                f"Test Loss: {record['test_loss']:.4f} Acc: {record['test_acc']:.2f}%"
            )

    return model, center_loss_fn, history


def _pca_preprocess(features: np.ndarray, out_dim: int = 30) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x = x / (x.std(axis=0, keepdims=True) + 1e-12)
    if x.shape[1] <= out_dim:
        return x
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:out_dim].T


def _squared_distances(x: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(x * x, axis=1, keepdims=True)
    dist = squared_norm + squared_norm.T - 2.0 * (x @ x.T)
    return np.maximum(dist, 0.0)


def _joint_probabilities(
    distances: np.ndarray,
    perplexity: float,
    tol: float = 1e-5,
    max_iter: int = 50,
) -> np.ndarray:
    n = distances.shape[0]
    target_entropy = math.log(perplexity)
    probs = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        beta = 1.0
        beta_min = -np.inf
        beta_max = np.inf
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dist_i = distances[i, mask]

        for _ in range(max_iter):
            p = np.exp(-dist_i * beta)
            sum_p = p.sum()
            if sum_p <= 1e-12:
                p = np.full_like(dist_i, 1.0 / len(dist_i))
                entropy = target_entropy
            else:
                p = p / sum_p
                entropy = -np.sum(p * np.log(p + 1e-12))

            entropy_diff = entropy - target_entropy
            if abs(entropy_diff) < tol:
                break

            if entropy_diff > 0:
                beta_min = beta
                beta = beta * 2.0 if np.isinf(beta_max) else 0.5 * (beta + beta_max)
            else:
                beta_max = beta
                beta = beta / 2.0 if np.isinf(beta_min) else 0.5 * (beta + beta_min)

        probs[i, mask] = p

    probs = (probs + probs.T) / (2.0 * n)
    probs = np.maximum(probs, 1e-12)
    probs /= probs.sum()
    return probs


def tsne_reduce(
    features: np.ndarray,
    *,
    perplexity: float = 30.0,
    n_iter: int = 350,
    learning_rate: float = 200.0,
    early_exaggeration: float = 8.0,
    random_state: int = 42,
) -> np.ndarray:
    x = _pca_preprocess(features, out_dim=min(30, features.shape[1]))
    n_samples = x.shape[0]
    if n_samples < 3:
        return x[:, :2].astype(np.float32)

    usable_perplexity = min(perplexity, max(5.0, (n_samples - 1) / 3.0))

    try:
        from sklearn.manifold import TSNE

        embedding = TSNE(
            n_components=2,
            perplexity=usable_perplexity,
            learning_rate="auto",
            init="pca",
            max_iter=n_iter,
            random_state=random_state,
        ).fit_transform(x)
        return np.asarray(embedding, dtype=np.float32)
    except Exception:
        pass

    distances = _squared_distances(x)
    joint_p = _joint_probabilities(distances, usable_perplexity)

    generator = torch.Generator(device="cpu").manual_seed(random_state)
    p = torch.tensor(joint_p * early_exaggeration, dtype=torch.float64)
    y = torch.randn((n_samples, 2), generator=generator, dtype=torch.float64) * 1e-4
    y.requires_grad_(True)
    optimizer = torch.optim.Adam([y], lr=learning_rate)

    release_step = max(50, n_iter // 4)

    for step in range(n_iter):
        squared_norm = torch.sum(y * y, dim=1, keepdim=True)
        dist = squared_norm + squared_norm.T - 2.0 * (y @ y.T)
        dist = torch.clamp(dist, min=0.0)

        inv_dist = 1.0 / (1.0 + dist)
        inv_dist.fill_diagonal_(0.0)
        q = inv_dist / torch.sum(inv_dist)
        q = torch.clamp(q, min=1e-12)

        loss = torch.sum(p * torch.log(p / q))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y -= y.mean(dim=0, keepdim=True)
            if step + 1 == release_step:
                p = torch.tensor(joint_p, dtype=torch.float64)

    return y.detach().cpu().numpy().astype(np.float32)


def try_umap_reduce(
    features: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    try:
        import umap
    except Exception as exc:  # pragma: no cover - depends on environment
        return None, f"UMAP 当前不可用，已自动跳过：{exc}"

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    embedding = reducer.fit_transform(_pca_preprocess(features, out_dim=min(50, features.shape[1])))
    return np.asarray(embedding, dtype=np.float32), None


def compute_cluster_metrics(embedding: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    emb = np.asarray(embedding, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    centers = []
    radii = []

    for class_id in classes:
        class_points = emb[labels == class_id]
        center = class_points.mean(axis=0)
        centers.append(center)
        radii.append(np.linalg.norm(class_points - center, axis=1))

    centers = np.stack(centers, axis=0)
    radii_all = np.concatenate(radii, axis=0)
    center_dists = np.sqrt(_squared_distances(centers))
    inter_dists = center_dists[np.triu_indices_from(center_dists, k=1)]

    compactness = float(radii_all.mean())
    mean_inter = float(inter_dists.mean())
    min_inter = float(inter_dists.min())
    global_spread = float(np.mean(np.std(emb, axis=0)))

    return {
        "compactness": compactness,
        "radius_std": float(radii_all.std()),
        "mean_inter": mean_inter,
        "min_inter": min_inter,
        "separation_ratio": mean_inter / (compactness + 1e-12),
        "global_spread": global_spread,
    }


def _scatter_embedding(ax: plt.Axes, embedding: np.ndarray, labels: np.ndarray, title: str) -> None:
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for class_id in range(10):
        mask = labels == class_id
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=10,
            alpha=0.70,
            c=[colors[class_id]],
            label=str(class_id),
        )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)


def _print_metric_table(records: Sequence[Dict[str, Any]], name_key: str = "name") -> None:
    header = (
        f"{name_key:>14} | {'Acc(%)':>7} | {'类内半径':>8} | {'最小类间距':>10} | "
        f"{'分离比':>8} | {'全局扩张':>8} | {'阶段':>6}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        name = str(record.get(name_key, "-"))
        phase = str(record.get("phase", "-"))
        print(
            f"{name:>14} | "
            f"{record.get('test_acc', float('nan')):7.2f} | "
            f"{record['compactness']:8.4f} | "
            f"{record['min_inter']:10.4f} | "
            f"{record['separation_ratio']:8.4f} | "
            f"{record['global_spread']:8.4f} | "
            f"{phase:>6}"
        )


def _default_sizes(device: torch.device) -> Dict[str, int]:
    is_cpu = getattr(device, "type", str(device)) == "cpu"
    if is_cpu:
        return {
            "epochs": 4,
            "train_samples": 4000,
            "test_samples": 1000,
            "tsne_iters": 300,
        }
    return {
        "epochs": 8,
        "train_samples": 12000,
        "test_samples": 2000,
        "tsne_iters": 450,
    }


def run_advanced_feature_visualization(
    *,
    model_cls: Callable[..., nn.Module],
    center_loss_cls: Callable[..., nn.Module],
    train_dataset: Any,
    test_dataset: Any,
    device: torch.device,
    lambda_center: float = 0.1,
    highdim_feature_dim: int = 32,
    epochs: Optional[int] = None,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    batch_size: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    defaults = _default_sizes(device)
    epochs = defaults["epochs"] if epochs is None else epochs
    train_samples = defaults["train_samples"] if train_samples is None else train_samples
    test_samples = defaults["test_samples"] if test_samples is None else test_samples
    tsne_iters = defaults["tsne_iters"]

    train_indices = make_stratified_indices(train_dataset, train_samples, seed=seed)
    test_indices = make_stratified_indices(test_dataset, test_samples, seed=seed + 1)
    train_loader = make_loader(train_dataset, train_indices, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(test_dataset, test_indices, batch_size=batch_size, shuffle=False)

    print(
        f"高级可视化对比：使用 {len(train_indices)} 个训练样本、"
        f"{len(test_indices)} 个测试样本，保证 2D 与高维模型在同一数据子集上训练。"
    )

    print("\n[1/2] 训练直接输出 2D 特征的模型")
    model_2d, _, history_2d = train_centerloss_model(
        model_cls,
        center_loss_cls,
        train_loader,
        test_loader,
        device,
        feature_dim=2,
        lambda_center=lambda_center,
        epochs=epochs,
        seed=seed,
    )

    print("\n[2/2] 训练高维特征模型")
    model_hd, _, history_hd = train_centerloss_model(
        model_cls,
        center_loss_cls,
        train_loader,
        test_loader,
        device,
        feature_dim=highdim_feature_dim,
        lambda_center=lambda_center,
        epochs=epochs,
        seed=seed + 7,
    )

    features_2d, labels = collect_features(model_2d, test_loader, device)
    features_hd, labels_hd = collect_features(model_hd, test_loader, device)

    if not np.array_equal(labels, labels_hd):
        raise RuntimeError("直接 2D 模型和高维模型提取到的标签顺序不一致。")

    print("\n开始执行自定义 t-SNE 降维...")
    tsne_embedding = tsne_reduce(
        features_hd,
        perplexity=30.0,
        n_iter=tsne_iters,
        learning_rate=max(80.0, len(labels) / 4.0),
        random_state=seed,
    )

    umap_embedding, umap_message = try_umap_reduce(features_hd, random_state=seed)

    embedding_records: list[tuple[str, np.ndarray]] = [
        ("Direct 2D Output", features_2d),
        (f"{highdim_feature_dim}D + t-SNE", tsne_embedding),
    ]
    if umap_embedding is not None:
        embedding_records.append((f"{highdim_feature_dim}D + UMAP", umap_embedding))

    metrics = []
    for name, embedding in embedding_records:
        record = {"name": name, **compute_cluster_metrics(embedding, labels)}
        metrics.append(record)

    metrics[0]["test_acc"] = history_2d[-1]["test_acc"]
    for record in metrics[1:]:
        record["test_acc"] = history_hd[-1]["test_acc"]

    fig, axes = plt.subplots(1, len(embedding_records), figsize=(6 * len(embedding_records), 5))
    if len(embedding_records) == 1:
        axes = [axes]

    for ax, (name, embedding) in zip(axes, embedding_records):
        _scatter_embedding(ax, embedding, labels, name)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:10], legend_labels[:10], loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.suptitle("Advanced Clustering Visualization", fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\n聚类质量指标对比：")
    _print_metric_table(metrics)

    best_compact = min(metrics, key=lambda item: item["compactness"])
    best_margin = max(metrics, key=lambda item: item["min_inter"])
    best_ratio = max(metrics, key=lambda item: item["separation_ratio"])
    print(
        f"\n结论提示：类内最紧凑的是“{best_compact['name']}”，"
        f"类别边界最清晰的是“{best_margin['name']}”，"
        f"综合分离比最高的是“{best_ratio['name']}”。"
    )
    if umap_message is not None:
        print(umap_message)

    return {
        "history_2d": history_2d,
        "history_highdim": history_hd,
        "metrics": metrics,
        "labels": labels,
        "features_2d": features_2d,
        "features_highdim": features_hd,
        "tsne_embedding": tsne_embedding,
        "umap_embedding": umap_embedding,
    }


def _label_lambda_phases(records: list[Dict[str, float]]) -> Dict[str, Optional[float]]:
    records.sort(key=lambda item: item["lambda"])
    if not records:
        return {"balanced_lambda": None, "collapse_lambda": None}

    best_acc = max(record["test_acc"] for record in records)
    baseline_compactness = records[0]["compactness"]
    baseline_spread = records[0]["global_spread"]
    best_ratio = max(record["separation_ratio"] for record in records)
    collapse_lambda = None

    for record in records:
        acc_drop = best_acc - record["test_acc"]
        compactness_rel = record["compactness"] / (baseline_compactness + 1e-12)
        ratio_rel = record["separation_ratio"] / (best_ratio + 1e-12)
        spread_rel = record["global_spread"] / (baseline_spread + 1e-12)

        # Center Loss 会主动把特征整体缩到更小的尺度上，
        # 所以只看“绝对距离变小”会把正常收缩误判成过坍缩。
        # 这里把“明显掉精度”作为过坍缩的必要条件。
        if acc_drop > 3.0 and (spread_rel < 0.45 or compactness_rel < 0.35):
            record["phase"] = "过坍缩"
            if collapse_lambda is None:
                collapse_lambda = record["lambda"]
        elif compactness_rel > 0.75 and ratio_rel < 0.90:
            record["phase"] = "扩散"
        else:
            record["phase"] = "平衡"

    non_collapse_records = [record for record in records if record["phase"] != "过坍缩"]
    if non_collapse_records:
        near_best_acc_records = [
            record for record in non_collapse_records
            if record["test_acc"] >= best_acc - 1.0
        ]
        if near_best_acc_records:
            balanced_record = max(
                near_best_acc_records,
                key=lambda item: item["separation_ratio"],
            )
        else:
            balanced_record = max(
                non_collapse_records,
                key=lambda item: (item["test_acc"], item["separation_ratio"]),
            )
    else:
        balanced_record = max(records, key=lambda item: item["separation_ratio"])

    balanced_record["phase"] = "平衡"
    balanced_lambda = balanced_record["lambda"]

    return {
        "balanced_lambda": balanced_lambda,
        "collapse_lambda": collapse_lambda,
    }


def _plot_lambda_curves(records: Sequence[Dict[str, float]]) -> None:
    lambdas = [record["lambda"] for record in records]
    xticks = np.arange(len(lambdas))
    ticklabels = [f"{value:g}" for value in lambdas]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].plot(xticks, [record["compactness"] for record in records], marker="o", label="Intra-class Radius")
    axes[0].plot(xticks, [record["global_spread"] for record in records], marker="s", label="Global Spread")
    axes[0].set_title("Compactness Trend")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(xticks, [record["mean_inter"] for record in records], marker="o", label="Mean Inter-class Distance")
    axes[1].plot(xticks, [record["min_inter"] for record in records], marker="s", label="Min Inter-class Distance")
    axes[1].set_title("Inter-class Separation Trend")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(xticks, [record["separation_ratio"] for record in records], marker="o", color="tab:green")
    axes[2].set_title("Separation Ratio")
    axes[2].grid(alpha=0.3)

    axes[3].plot(xticks, [record["test_acc"] for record in records], marker="o", color="tab:red")
    axes[3].set_title("Test Accuracy")
    axes[3].grid(alpha=0.3)

    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels(ticklabels)
        ax.set_xlabel("lambda")

    plt.suptitle("Center Loss lambda Sweep", fontsize=14)
    plt.tight_layout()
    plt.show()


def _plot_lambda_embeddings(records: Sequence[Dict[str, Any]]) -> None:
    phase_map = {
        "扩散": "Diffuse",
        "平衡": "Balanced",
        "过坍缩": "Over-collapse",
    }
    cols = min(3, len(records))
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, record in zip(axes, records):
        phase_label = phase_map.get(record["phase"], record["phase"])
        _scatter_embedding(ax, record["embedding"], record["labels"], f"lambda = {record['lambda']:g} ({phase_label})")

    for ax in axes[len(records):]:
        ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:10], legend_labels[:10], loc="upper right", bbox_to_anchor=(0.98, 0.98))
    plt.suptitle("2D Feature Space under Different lambda Values", fontsize=14)
    plt.tight_layout()
    plt.show()


def run_lambda_sweep(
    *,
    model_cls: Callable[..., nn.Module],
    center_loss_cls: Callable[..., nn.Module],
    train_dataset: Any,
    test_dataset: Any,
    device: torch.device,
    lambda_values: Sequence[float],
    epochs: Optional[int] = None,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
    batch_size: int = 256,
    seed: int = 123,
) -> Dict[str, Any]:
    defaults = _default_sizes(device)
    epochs = defaults["epochs"] if epochs is None else epochs
    train_samples = defaults["train_samples"] if train_samples is None else train_samples
    test_samples = defaults["test_samples"] if test_samples is None else test_samples

    train_indices = make_stratified_indices(train_dataset, train_samples, seed=seed)
    test_indices = make_stratified_indices(test_dataset, test_samples, seed=seed + 1)
    train_loader = make_loader(train_dataset, train_indices, batch_size=batch_size, shuffle=True)
    test_loader = make_loader(test_dataset, test_indices, batch_size=batch_size, shuffle=False)

    print(
        f"λ 扫描：使用 {len(train_indices)} 个训练样本、"
        f"{len(test_indices)} 个测试样本，固定 2D 特征维度。"
    )

    records: list[Dict[str, Any]] = []
    for offset, lambda_center in enumerate(sorted(float(value) for value in lambda_values)):
        print(f"\n===== 开始训练 λ = {lambda_center:g} =====")
        model, _, history = train_centerloss_model(
            model_cls,
            center_loss_cls,
            train_loader,
            test_loader,
            device,
            feature_dim=2,
            lambda_center=lambda_center,
            epochs=epochs,
            seed=seed + offset,
        )
        embedding, labels = collect_features(model, test_loader, device)
        record = {
            "lambda": lambda_center,
            "test_acc": history[-1]["test_acc"],
            **compute_cluster_metrics(embedding, labels),
            "embedding": embedding,
            "labels": labels,
        }
        records.append(record)

    summary = _label_lambda_phases(records)
    _plot_lambda_curves(records)
    _plot_lambda_embeddings(records)

    print("\nλ 扫描结果汇总：")
    _print_metric_table(records, name_key="lambda")

    balanced_lambda = summary["balanced_lambda"]
    collapse_lambda = summary["collapse_lambda"]
    if collapse_lambda is None:
        print(
            f"\n本次扫描中，λ≈{balanced_lambda:g} 附近达到较好的紧凑度/分离度平衡；"
            "在当前扫描范围内还没有出现非常明确的过坍缩。"
        )
    else:
        print(
            f"\n本次扫描中，λ≈{balanced_lambda:g} 附近开始进入较理想的聚类状态，"
            f"当 λ≈{collapse_lambda:g} 时开始出现“过坍缩”信号："
            "类内半径继续缩小，但全局扩张和最小类间距开始一起变小。"
        )

    return {
        "records": records,
        "summary": summary,
    }
