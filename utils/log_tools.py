import numpy as np
import torch
import os


def save_train_logs(logs, save_path):
    """
    保存训练日志（损失、mAP等）
    :param logs: 日志字典（key: epoch, value: {"train_loss": xx, "val_mAP": xx}）
    :param save_path: 保存路径（如"runs/train/train_logs.npy"）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, logs)
    print(f"✅ 训练日志已保存至：{save_path}")


def load_train_logs(log_path):
    """
    加载训练日志
    :param log_path: 日志文件路径
    :return: 日志字典
    """
    if not os.path.exists(log_path):
        raise Exception(f"❌ 日志文件未找到：{log_path}")
    return np.load(log_path, allow_pickle=True).item()


def save_model_weights(model, optimizer, epoch, best_val_map, save_path):
    """
    保存模型权重（含优化器状态、当前轮次、最佳mAP）
    :param model: 模型
    :param optimizer: 优化器
    :param epoch: 当前训练轮次
    :param best_val_map: 最佳验证mAP
    :param save_path: 保存路径（如"runs/train/best_yolov5_model.pt"）
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_map": best_val_map
    }, save_path)
    print(f"✅ 模型权重已保存至：{save_path}（当前轮次：{epoch}，最佳mAP：{best_val_map:.4f}）")


def load_model_weights(model, weight_path, device):
    """
    加载模型权重
    :param model: 待加载权重的模型
    :param weight_path: 权重文件路径
    :param device: 设备（cpu/cuda）
    :return: 加载权重后的模型、最佳mAP
    """
    if not os.path.exists(weight_path):
        raise Exception(f"❌ 模型权重文件未找到：{weight_path}")
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    best_val_map = checkpoint["best_val_map"]
    print(f"✅ 模型权重已加载（最佳mAP：{best_val_map:.4f}）")
    return model, best_val_map