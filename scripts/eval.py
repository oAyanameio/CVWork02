import os
import yaml
import torch
from torch.utils.data import DataLoader
from typing import Dict

# 导入自定义模块
from data.dataset import COCODataset, collate_fn
from models.yolov5 import YOLOv5
from utils.evaluation import evaluate_coco
from utils.visualization import create_result_dirs, visualize_challenging_cases


def load_config(config_path: str = "../config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_trained_model(config: Dict, device: torch.device) -> YOLOv5:
    """加载训练好的模型权重"""
    weight_path = f"{config['train']['save_dir']}/best_yolov5_model.pt"
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Trained weight not found at {weight_path}")

    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location=device)
    # 初始化模型
    model = YOLOv5(num_classes=config["model"]["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Successfully loaded trained model from {weight_path}")
    print(f"Best val mAP@0.5 of this model: {checkpoint['best_val_map']:.4f}")
    return model


def main():
    # 1. 加载配置与初始化
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    # 创建结果目录
    create_result_dirs(config["eval"]["result_dir"])

    # 2. 加载数据集（验证集/测试集）
    print("Loading validation dataset...")
    val_dataset = COCODataset(
        img_dir=config["data"]["val_img_dir"],
        ann_path=config["data"]["val_ann_path"],
        img_size=config["data"]["img_size"],
        augment=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 3. 加载训练好的模型
    model = load_trained_model(config, device)
    model.eval()  # 切换到评估模式

    # 4. 执行COCO指标评估
    print("Starting COCO evaluation...")
    val_metrics = evaluate_coco(
        model=model,
        dataloader=val_loader,
        ann_path=config["data"]["val_ann_path"],
        anchors=torch.tensor(config["anchors"], device=device),
        device=device,
        conf_thres=config["eval"]["conf_thres"]
    )

    # 打印评估结果
    print("\n=== Final Evaluation Results ===")
    print(f"mAP@0.5:         {val_metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.5:0.95:    {val_metrics['mAP@0.5:0.95']:.4f}")
    print(f"Precision:       {val_metrics['Precision']:.4f}")
    print(f"Recall:          {val_metrics['Recall']:.4f}")

    # 5. 可视化挑战性案例（小目标、遮挡等）
    print("\nVisualizing challenging cases...")
    visualize_challenging_cases(
        dataset=val_dataset,
        model=model,
        anchors=torch.tensor(config["anchors"], device=device),
        device=device,
        save_path=config["eval"]["result_dir"],
        num_cases=3,
        conf_thres=config["eval"]["conf_thres"]
    )

    print("\nEvaluation finished! All results saved to", config["eval"]["result_dir"])


if __name__ == "__main__":
    main()