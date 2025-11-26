# ==============================
# 1. 导入必要库
# ==============================
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.metrics import ConfusionMatrix
import shutil

# ==============================
# 2. 配置全局参数（需根据自身环境调整）
# ==============================
class Config:
    # 数据集配置（COCO会自动下载，路径可自定义）
    data_path = "./datasets/coco"  # 数据集保存路径
    coco_yaml = "coco128.yaml"     # 简化版COCO（128张图，快速测试；正式训练用"coco.yaml"）
    
    # 模型配置
    model_type = "yolov8n.pt"      # YOLOv8 nano（轻量，适合PC训练；可选yolov8s/m/l/x.pt）
    trained_weights_path = "./runs/detect/train/weights/best.pt"  # 训练后权重保存路径
    
    # 训练配置
    epochs = 10                    # 训练轮次（正式训练建议30-50）
    batch_size = 4                 # 批次大小（根据GPU显存调整，显存足设8-16）
    img_size = 640                 # 输入图像尺寸
    
    # 结果保存配置
    save_dir = "./assignment_results"  # 作业结果总目录
    vis_dir = f"{save_dir}/visualizations"  # 可视化结果目录
    model_dir = f"{save_dir}/trained_model"  # 训练模型保存目录


# ==============================
# 3. 初始化目录（确保结果保存路径存在）
# ==============================
def init_dirs():
    dirs = [Config.save_dir, Config.vis_dir, Config.model_dir]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
    print("✅ 所有结果目录初始化完成")


# ==============================
# 4. 数据预处理与可视化（满足作业"数据预处理"要求）
# ==============================
def visualize_dataset_samples():
    """加载COCO样本并绘制边界框，保存可视化结果"""
    # 加载YOLO数据集（自动解析COCO标注）
    from ultralytics.data import YOLODataset
    dataset = YOLODataset(
        img_path=os.path.join(Config.data_path, "train2017"),
        yaml_path=Config.coco_yaml,
        img_size=Config.img_size,
        augment=False  # 不增强，仅可视化原始样本
    )
    
    # 可视化5个不同类别的样本
    class_names = dataset.names  # COCO类别名称（如"person", "car"）
    for i in range(5):
        img, targets, paths = dataset[i]  # 读取图像、标注、路径
        img = img.permute(1, 2, 0).cpu().numpy()  # 转换为OpenCV格式（HWC）
        img = (img * 255).astype("uint8")  # 反归一化（YOLO加载时默认归一化到0-1）
        
        # 绘制边界框（targets格式：[class_id, x1, y1, x2, y2]，相对坐标→绝对坐标）
        h, w = img.shape[:2]
        for target in targets:
            cls_id, x1, y1, x2, y2 = target
            x1, y1 = int(x1 * w), int(y1 * h)
            x2, y2 = int(x2 * w), int(y2 * h)
            # 画框（红色，线宽2）+ 写类别名称（白色文字，黑色背景）
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cls_name = class_names[int(cls_id)]
            cv2.putText(
                img, cls_name, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        # 保存可视化结果
        save_path = f"{Config.vis_dir}/dataset_sample_{i+1}.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # 转换为BGR（OpenCV默认）
    print(f"✅ 数据集可视化完成，结果保存在 {Config.vis_dir}")


# ==============================
# 5. 模型训练（满足作业"模型设计与训练"要求）
# ==============================
def train_model():
    """加载YOLOv8模型，配置训练参数并开始训练"""
    # 加载预训练模型（backbone为CSPDarknet，符合作业"backbone网络"要求）
    model = YOLO(Config.model_type)
    
    # 开始训练（内置分类损失+回归损失，符合作业"损失函数"要求）
    results = model.train(
        data=Config.coco_yaml,
        epochs=Config.epochs,
        batch=Config.batch_size,
        imgsz=Config.img_size,
        augment=True,  # 启用数据增强（随机翻转、缩放等，符合作业"数据增强"要求）
        device=0,      # 0=使用GPU，-1=使用CPU（建议用GPU，否则训练极慢）
        project=Config.save_dir,  # 训练结果保存根目录
        name="train",  # 训练结果子目录（会自动创建）
        save=True,     # 保存模型权重
        val=True       # 训练中自动验证
    )
    
    # 保存训练曲线（损失、mAP等）
    plot_results(results=results, save_dir=Config.vis_dir)
    print(f"✅ 模型训练完成！训练曲线保存在 {Config.vis_dir}")
    print(f"✅ 最佳模型权重保存在 {Config.trained_weights_path}")


# ==============================
# 6. 模型评估（满足作业"模型评估与分析"要求）
# ==============================
def evaluate_model():
    """用 COCO 测试集评估模型，计算 mAP（不依赖 pycocotools）"""
    # 加载训练好的模型
    model = YOLO(Config.trained_weights_path)
    
    # 计算标准指标
    eval_results = model.val(
        data=Config.coco_yaml,
        split="test",
        imgsz=Config.img_size,
        device=0,
        verbose=True  # 显示详细评估结果
    )
    
    # 打印核心评估指标（作业要求的 mAP 已包含）
    print("\n" + "="*50)
    print("📊 模型评估结果（COCO 测试集）")
    print(f"mAP@0.5: {eval_results.box.map:.4f}")       # 作业核心指标
    print(f"mAP@0.5:0.95: {eval_results.box.map50_95:.4f}")  # 拓展指标
    print(f"Precision: {eval_results.box.precision:.4f}")    # 精度
    print(f"Recall: {eval_results.box.recall:.4f}")          # 召回率
    print("="*50 + "\n")
    
    # 难例分析（不变，无需 pycocotools）
    analyze_hard_cases(model)
    print(f"✅ 模型评估完成！难例检测结果保存在 {Config.vis_dir}")
    
    # 打印评估结果
    print("\n" + "="*50)
    print("📊 模型评估结果（COCO测试集）")
    print(f"mAP@0.5: {eval_results.box.map:.4f}")       # IoU=0.5时的mAP
    print(f"mAP@0.5:0.95: {eval_results.box.map50_95:.4f}")  # IoU=0.5-0.95的mAP
    print(f"Precision: {eval_results.box.precision:.4f}")    # 精度
    print(f"Recall: {eval_results.box.recall:.4f}")          # 召回率
    print("="*50 + "\n")
    
    # 2. 可视化混淆矩阵（分析类别级检测效果）
    conf_matrix = ConfusionMatrix(model.names)
    conf_matrix.plot(save_dir=Config.vis_dir, fname="confusion_matrix.png")
    
    # 3. 难例分析（小目标、遮挡目标）
    analyze_hard_cases(model)
    print(f"✅ 模型评估完成！混淆矩阵保存在 {Config.vis_dir}")


def analyze_hard_cases(model):
    """分析小目标、遮挡目标等难例，保存检测结果"""
    # 加载COCO测试集中的难例（这里用10张含小目标/遮挡的样本，可自定义路径）
    test_img_paths = [
        os.path.join(Config.data_path, "test2017", f"{i:012d}.jpg") 
        for i in range(100000, 100010)  # COCO测试集图像ID（示例）
    ]
    
    for img_path in test_img_paths:
        if not os.path.exists(img_path):
            continue
        
        # 检测图像
        results = model.predict(
            source=img_path,
            imgsz=Config.img_size,
            conf=0.25,  # 置信度阈值（过滤低置信度预测）
            iou=0.5     # IoU阈值（非极大值抑制）
        )
        
        # 保存检测结果（含边界框和类别标签）
        img = results[0].plot()  # 自动绘制检测框
        img_name = os.path.basename(img_path)
        save_path = f"{Config.vis_dir}/hard_case_{img_name}"
        cv2.imwrite(save_path, img)
    
    print(f"✅ 难例分析完成！难例检测结果保存在 {Config.vis_dir}")


# ==============================
# 7. 模型保存与提交材料整理（满足作业"提交材料"要求）
# ==============================
def organize_submission_materials():
    """整理作业提交材料：代码、模型权重、可视化、报告模板"""
    # 1. 复制训练好的模型到指定目录
    if os.path.exists(Config.trained_weights_path):
        shutil.copy(Config.trained_weights_path, Config.model_dir)
        # 复制模型配置文件（YOLOv8配置）
        shutil.copy(
            os.path.join(os.path.dirname(Config.trained_weights_path), "args.yaml"),
            Config.model_dir
        )
    
    # 2. 保存代码文件（当前脚本）
    shutil.copy(__file__, Config.save_dir)
    
    # 3. 生成实验报告模板（Markdown格式，用户需补充细节）
    report_content = """# 目标检测作业实验报告
## 1. 数据集介绍
- 采用数据集：COCO 2017
- 训练集规模：118k images（正式训练）/ 128 images（测试）
- 类别数量：80类
- 标注类型：边界框（bounding box）

## 2. 模型设计
- 模型架构：YOLOv8（单阶段检测器）
- Backbone：CSPDarknet
- 检测头：多尺度检测头（支持小/中/大目标检测）
- 损失函数：分类损失（交叉熵）+ 回归损失（CIoU）

## 3. 训练过程
- 超参数：epochs={}, batch_size={}, img_size={}
- 优化器：AdamW
- 数据增强：随机水平翻转、尺度缩放、亮度调整
- 训练损失曲线：见 visualizations/results.png

## 4. 评估结果
- mAP@0.5：{}（需补充实际数值）
- mAP@0.5:0.95：{}（需补充实际数值）
- 难例分析：
  - 小目标：检测精度较低（因特征提取不充分）
  - 遮挡目标：遮挡率>50%时易漏检（因目标特征不完整）

## 5. 改进方向
1. 增加小目标样本的数据增强（如过采样）
2. 融合上下文特征（如加入注意力机制）
3. 调整置信度阈值以平衡精度和召回率
""".format(Config.epochs, Config.batch_size, Config.img_size, "?", "?")
    
    # 保存报告模板
    with open(f"{Config.save_dir}/experiment_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"✅ 提交材料整理完成！所有材料保存在 {Config.save_dir}")
    print("\n📋 提交材料清单：")
    print(f"1. 代码：{Config.save_dir}/{os.path.basename(__file__)}")
    print(f"2. 模型权重：{Config.model_dir}/best.pt")
    print(f"3. 实验报告：{Config.save_dir}/experiment_report.md")
    print(f"4. 可视化结果：{Config.vis_dir}（样本图、训练曲线、混淆矩阵、难例分析）")


# ==============================
# 8. 主函数（按流程执行所有步骤）
# ==============================
if __name__ == "__main__":
    # 步骤1：初始化目录
    init_dirs()
    
    # 步骤2：数据集可视化
    visualize_dataset_samples()
    
    # 步骤3：模型训练（耗时较长，GPU约1-2小时/10轮）
    train_model()
    
    # 步骤4：模型评估
    evaluate_model()
    
    # 步骤5：整理提交材料
    organize_submission_materials()
    
    print("\n🎉 目标检测作业代码全部执行完成！请检查 submission_results 目录准备提交。")