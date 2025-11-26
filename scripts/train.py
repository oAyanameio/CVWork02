import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict
project_root = '/home/lbh/CVWork2'
if project_root not in sys.path:
    sys.path.append(project_root)
# 导入自定义模块
from data.dataset import COCODataset, collate_fn
from models.yolo5 import init_yolov5_model
from models.loss import YOLOLoss
from utils.evaluation import evaluate_coco
from utils.visualization import create_result_dirs, visualize_data_samples, plot_train_logs


def load_config(config_path: str = "../config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # 1. 加载配置与初始化
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 创建结果目录
    create_result_dirs(config["eval"]["result_dir"])
    # 创建权重保存目录
    if not os.path.exists(config["train"]["save_dir"]):
        os.makedirs(config["train"]["save_dir"])

    # 2. 加载数据集
    print("Loading dataset...")
    train_dataset = COCODataset(
        img_dir=config["data"]["train_img_dir"],
        ann_path=config["data"]["train_ann_path"],
        img_size=config["data"]["img_size"],
        augment=True  # 训练阶段启用数据增强
    )
    val_dataset = COCODataset(
        img_dir=config["data"]["val_img_dir"],
        ann_path=config["data"]["val_ann_path"],
        img_size=config["data"]["img_size"],
        augment=False  # 验证阶段禁用数据增强
    )

    # 可视化数据样本（验证数据加载正确性）
    visualize_data_samples(val_dataset, config["eval"]["result_dir"], num_samples=5)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 3. 初始化模型、损失函数、优化器
    print("Initializing model...")
    model = init_yolov5_model(
        num_classes=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
        pretrained_weight_path=config["model"]["pretrained_weight_path"]
    ).to(device)

    criterion = YOLOLoss(num_classes=config["model"]["num_classes"]).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"]
    )
    # 学习率调度器（余弦退火）
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"], eta_min=1e-6
    )

    # 4. 训练循环
    print("Start training...")
    best_val_map = 0.0  # 最佳验证mAP（用于早停与权重保存）
    train_losses = []   # 训练损失日志
    val_maps = []       # 验证mAP日志

    for epoch in range(config["train"]["epochs"]):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")

        # 训练一轮
        for batch in pbar:
            imgs, bboxes, labels, img_ids = batch
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            labels = labels.to(device)

            # 拼接真实标签（[B, N, 5]：x1,y1,x2,y2,cls_id）
            targets = torch.cat([bboxes, labels.unsqueeze(-1)], dim=-1)

            # 前向传播
            preds = model(imgs)
            # 转换锚框格式（适配损失函数）
            anchors = torch.tensor(config["anchors"], device=device).view(3, 3, 2)
            # 计算损失
            loss = criterion(preds, targets, anchors, img_size=config["data"]["img_size"])

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"Batch Loss": loss.item()})

        # 计算本轮平均损失
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # 学习率更新
        lr_scheduler.step()

        # 验证（每val_interval轮）
        if (epoch + 1) % config["train"]["val_interval"] == 0:
            print("Evaluating on validation set...")
            val_metrics = evaluate_coco(
                model=model,
                dataloader=val_loader,
                ann_path=config["data"]["val_ann_path"],
                anchors=torch.tensor(config["anchors"], device=device),
                device=device,
                conf_thres=config["eval"]["conf_thres"]
            )
            val_maps.append(val_metrics["mAP@0.5"])
            print(f"Val mAP@0.5: {val_metrics['mAP@0.5']:.4f} | Val Precision: {val_metrics['Precision']:.4f} | Val Recall: {val_metrics['Recall']:.4f}")

            # 保存最佳模型
            if val_metrics["mAP@0.5"] > best_val_map:
                best_val_map = val_metrics["mAP@0.5"]
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_map": best_val_map,
                    "config": config
                }, f"{config['train']['save_dir']}/best_yolov5_model.pt")
                print(f"Best model saved! Current best mAP@0.5: {best_val_map:.4f}")

    # 5. 训练结束：保存日志与可视化
    # 保存训练日志
    np.save(f"{config['eval']['result_dir']}/train_logs/train_losses.npy", np.array(train_losses))
    np.save(f"{config['eval']['result_dir']}/train_logs/val_maps.npy", np.array(val_maps))
    # 绘制训练日志曲线
    plot_train_logs(train_losses, val_maps, config["eval"]["result_dir"])
    print("Training finished! All results saved to", config["eval"]["result_dir"])


if __name__ == "__main__":
    main()