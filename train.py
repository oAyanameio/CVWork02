from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from ultralytics.utils.plotting import plot_results
import shutil
import torch

class Config:
    coco_yaml = "/home/lbh/.conda/envs/cvwork2/lib/python3.10/site-packages/ultralytics/data/datasets/coco128.yaml"
    data_path = "./datasets/coco128"
    model_type = "yolov8n.pt"
    trained_weights_path = "./runs/detect/train/weights/best.pt"
    epochs = 10
    batch_size = 4
    img_size = 640
    save_dir = "./assignment_results"
    vis_dir = f"{save_dir}/visualizations"
    model_dir = f"{save_dir}/trained_model"

def init_dirs():
    dirs = [Config.save_dir, Config.vis_dir, Config.model_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print("所有结果目录初始化完成")

def visualize_dataset_samples():
    from ultralytics.data import YOLODataset
    from ultralytics.data.utils import load_dataset

    data = load_dataset(Config.coco_yaml)
    dataset = YOLODataset(
        img_path=data["train"],
        data=data,
        img_size=Config.img_size,
        augment=False
    )

    class_names = data["names"]
    for i in range(5):
        img, targets, paths = dataset[i]
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype("uint8")

        h, w = img.shape[:2]
        for target in targets:
            cls_id, x1, y1, x2, y2 = target
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cls_name = class_names[int(cls_id)]
            cv2.putText(
                img, cls_name, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )

        save_path = f"{Config.vis_dir}/dataset_sample_{i+1}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"数据集可视化完成，结果保存在 {Config.vis_dir}")

def train_model():
    model = YOLO(Config.model_type)
    results = model.train(
        data=Config.coco_yaml,
        epochs=Config.epochs,
        batch=Config.batch_size,
        imgsz=Config.img_size,
        augment=True,
        device=0 if torch.cuda.is_available() else -1,
        project=Config.save_dir,
        name="train",
        save=True,
        val=True
    )

    plot_results(results=results, save_dir=Config.vis_dir)
    print(f" 模型训练完成！训练曲线保存在 {Config.vis_dir}")
    print(f" 最佳模型权重保存在 {Config.trained_weights_path}")

def plot_confusion_matrix(truths, preds, class_names, save_path):
    num_classes = len(class_names)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(truths, preds):
        if t < num_classes and p < num_classes:
            conf_matrix[t, p] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix (No pycocotools)", fontsize=14)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, conf_matrix[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"混淆矩阵绘制完成，保存在 {save_path}")

def evaluate_model():
    model = YOLO(Config.trained_weights_path)
    from ultralytics.data.utils import load_dataset
    data = load_dataset(Config.coco_yaml)
    class_names = list(data["names"].values())[:10]

    eval_results = model.val(
        data=Config.coco_yaml,
        split="test",
        imgsz=Config.img_size,
        device=0 if torch.cuda.is_available() else -1,
        verbose=True
    )

    print("\n" + "="*50)
    print(" 模型评估结果（COCO128 测试集）")
    print(f"mAP@0.5: {eval_results.box.map:.4f}")
    print(f"mAP@0.5:0.95: {eval_results.box.map50_95:.4f}")
    print(f"Precision: {eval_results.box.precision:.4f}")
    print(f"Recall: {eval_results.box.recall:.4f}")
    print("="*50 + "\n")

    try:
        truths = eval_results.box.target[:, 0].cpu().numpy().astype(int)
        preds = eval_results.box.pred[:, 5].cpu().numpy().astype(int)
        plot_confusion_matrix(truths, preds, class_names, f"{Config.vis_dir}/confusion_matrix.png")
    except Exception as e:
        print(f"⚠️  混淆矩阵绘制失败：{str(e)}")

    analyze_hard_cases(model, data)
    print(f" 模型评估完成！难例结果保存在 {Config.vis_dir}")

def analyze_hard_cases(model, data):
    test_img_paths = [
        os.path.join(data["test"], f"{i:012d}.jpg")
        for i in range(100000, 100010)
    ]

    for idx, img_path in enumerate(test_img_paths):
        if os.path.exists(img_path):
            results = model.predict(
                source=img_path,
                imgsz=Config.img_size,
                conf=0.25,
                iou=0.5
            )
            img = results[0].plot()
            save_path = f"{Config.vis_dir}/hard_case_{idx+1}.jpg"
            cv2.imwrite(save_path, img)

def organize_submission_materials():
    if os.path.exists(Config.trained_weights_path):
        shutil.copy(Config.trained_weights_path, Config.model_dir)
        args_path = os.path.join(os.path.dirname(Config.trained_weights_path), "args.yaml")
        if os.path.exists(args_path):
            shutil.copy(args_path, Config.model_dir)

    shutil.copy(__file__, Config.save_dir)

    print(f"提交材料整理完成！所有材料保存在 {Config.save_dir}")
    print(f"包含内容：")
    print(f"  - 训练好的模型权重：{Config.model_dir}")
    print(f"  - 可视化结果（数据集样本、训练曲线、混淆矩阵、难例）：{Config.vis_dir}")
    print(f"  - 源代码文件：{Config.save_dir}/{os.path.basename(__file__)}")

if __name__ == "__main__":
    init_dirs()
    visualize_dataset_samples()
    train_model()
    evaluate_model()
    organize_submission_materials()
    print("\n目标检测作业代码全部执行完成")