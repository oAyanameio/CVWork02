import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from torch.utils.data import Dataset
from models.yolov5 import YOLOv5
from typing import List, Dict
import os
from pycocotools.coco import COCO


def create_result_dirs(result_dir: str) -> None:
    """创建结果保存目录（自动创建不存在的文件夹）"""
    dirs = [
        f"{result_dir}/data_samples",
        f"{result_dir}/train_logs",
        f"{result_dir}/challenging_cases"
    ]
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def visualize_data_samples(dataset: Dataset, save_path: str, num_samples: int = 5) -> None:
    """可视化数据样本（含bounding box和类别名称）"""
    # 获取COCO类别名称映射（ID→名称）
    coco = dataset.coco
    cat_ids = coco.getCatIds()
    cat_name_map = {cat["id"]: cat["name"] for cat in coco.loadCats(cat_ids)}

    # 随机选择样本（固定种子确保可复现）
    np.random.seed(42)
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)

    # 绘制样本
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for idx, sample_idx in enumerate(sample_indices):
        img, bboxes, labels, img_id = dataset[sample_idx]
        # Tensor→numpy（[C,H,W]→[H,W,C]）
        img_np = img.permute(1, 2, 0).numpy()
        # 反归一化（如果需要，当前img已在dataset中归一化到[0,1]，可直接显示）

        # 绘制图像
        axes[idx].imshow(img_np)
        axes[idx].axis("off")
        axes[idx].set_title(f"Sample {img_id}")

        # 绘制bounding box（绿色边框）和类别名称
        for bbox, label in zip(bboxes, labels):
            if len(bbox) == 0:
                break
            x1, y1, x2, y2 = bbox.numpy()
            # 创建矩形框
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="green", facecolor="none"
            )
            axes[idx].add_patch(rect)
            # 标注类别名称
            cat_name = cat_name_map.get(label.item(), "Unknown")
            axes[idx].text(
                x1, y1 - 5, cat_name, fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/data_samples/data_samples.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Data samples saved to {save_path}/data_samples/data_samples.png")


def plot_train_logs(train_losses: List[float], val_maps: List[float], save_path: str) -> None:
    """绘制训练损失曲线和验证mAP曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 1. 训练损失曲线
    ax1.plot(range(1, len(train_losses) + 1), train_losses, color="red", label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss Curve")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 验证mAP曲线（每val_interval轮一个点）
    val_epochs = range(len(val_maps))  # 假设val_interval=5，需根据实际调整
    ax2.plot(val_epochs, val_maps, color="blue", marker="o", label="Val mAP@0.5")
    ax2.set_xlabel("Validation Epoch")
    ax2.set_ylabel("mAP@0.5")
    ax2.set_title("Validation mAP Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/train_logs/train_logs.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Train logs saved to {save_path}/train_logs/train_logs.png")


def visualize_challenging_cases(
        dataset: Dataset, model: YOLOv5, anchors: torch.Tensor,
        device: torch.device, save_path: str, num_cases: int = 3, conf_thres: float = 0.3
) -> None:
    """可视化挑战性案例（小目标、遮挡、低光照）"""
    from utils.evaluation import decode_predictions  # 导入预测解码函数

    model.eval()
    coco = dataset.coco
    cat_name_map = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

    # 1. 筛选挑战性案例（以小目标为例：面积<32×32）
    challenging_indices = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        # 判断是否含小目标
        has_small_obj = any(ann["bbox"][2] * ann["bbox"][3] < 32 * 32 for ann in anns)
        if has_small_obj:
            challenging_indices.append(idx)
            if len(challenging_indices) == num_cases:
                break

    # 2. 绘制真实标注与模型预测对比
    fig, axes = plt.subplots(num_cases, 2, figsize=(12, 15))  # 左：真实，右：预测
    with torch.no_grad():
        for i, idx in enumerate(challenging_indices):
            img, bboxes, labels, img_id = dataset[idx]
            img_np = img.permute(1, 2, 0).numpy()
            img_tensor = img.unsqueeze(0).to(device)

            # 绘制真实标注（左图）
            axes[i, 0].imshow(img_np)
            axes[i, 0].axis("off")
            axes[i, 0].set_title(f"Case {i+1}: Ground Truth (Small Object)")
            for bbox, label in zip(bboxes, labels):
                x1, y1, x2, y2 = bbox.numpy()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="green", facecolor="none")
                axes[i, 0].add_patch(rect)
                axes[i, 0].text(x1, y1-5, cat_name_map[label.item()], fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

            # 模型预测（右图）
            preds = model(img_tensor)
            decoded_preds = decode_predictions(preds, anchors, img_size=img.shape[1], conf_thres=conf_thres)[0]
            axes[i, 1].imshow(img_np)
            axes[i, 1].axis("off")
            axes[i, 1].set_title(f"Case {i+1}: Model Prediction")
            for pred in decoded_preds:
                x1, y1, x2, y2, conf, cls_id = pred.numpy()
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none")
                axes[i, 1].add_patch(rect)
                axes[i, 1].text(x1, y1-5, f"{cat_name_map[cls_id]} (conf:{conf:.2f})", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"{save_path}/challenging_cases/challenging_cases.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Challenging cases saved to {save_path}/challenging_cases/challenging_cases.png")