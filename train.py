import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
from pycocotools.coco import COCO
import os
import zipfile
import urllib.request
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# --- 0. 配置参数 (Configuration) ---
COCO_DATA_ROOT = './coco_data' 
NUM_CLASSES = 91 # COCO 80 categories + 1 background + others
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 4
LEARNING_RATE = 0.005
NUM_EPOCHS = 5 # 建议在正式提交前训练更久 (例如 20+ epochs)
SAVE_PATH = 'faster_rcnn_coco_assignment_weights.pth' 


# --- 1. 数据下载与设置 (Data Preprocessing Requirement 1) ---

class DownloadProgressBar(tqdm):
    """带进度条的下载工具"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    """带进度条的下载函数"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def setup_coco_dataset():
    """下载并解压 COCO 2017 训练集图片和标注文件"""
    print("--- 正在设置 COCO 数据集 (用于训练) ---")
    os.makedirs(COCO_DATA_ROOT, exist_ok=True)
    
    files_to_download = {
        'train2017.zip': 'http://images.cocodataset.org/zips/train2017.zip',
        'annotations_trainval2017.zip': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }

    for filename, url in files_to_download.items():
        zip_path = os.path.join(COCO_DATA_ROOT, filename)
        
        # 简化检查，只要目标目录存在且非空就跳过
        if filename == 'annotations_trainval2017.zip':
            target_dir = os.path.join(COCO_DATA_ROOT, 'annotations')
        else:
            target_dir = os.path.join(COCO_DATA_ROOT, filename.split('.')[0])
        
        if os.path.exists(target_dir) and os.listdir(target_dir):
            print(f"✅ {filename.split('.')[0]} 已存在。跳过下载和解压。")
            continue
                
        # 下载和解压逻辑 (与之前代码相同)
        if not os.path.exists(zip_path):
            print(f"⬇️ 正在下载 {filename} (请耐心等待)...")
            download_url(url, zip_path)
            print(f"✅ {filename} 下载完成。")
        
        print(f"🔨 正在解压 {filename}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(COCO_DATA_ROOT)
        print(f"✅ {filename} 解压完成。")
    
    IMG_DIR_TRAIN = os.path.join(COCO_DATA_ROOT, 'train2017')
    ANN_FILE_TRAIN = os.path.join(COCO_DATA_ROOT, 'annotations', 'instances_train2017.json')
    
    print("--- COCO 数据集设置完成 ---")
    return IMG_DIR_TRAIN, ANN_FILE_TRAIN


# --- 2. 数据集类和加载器 (Data Preprocessing) ---

class CocoDetection_Custom(CocoDetection):
    """
    重写 CocoDetection，将 COCO 格式标注转换为 PyTorch 目标检测模型所需的格式。
    """
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)
        # 获取 COCO API 实例
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        coco_target = self.coco.loadAnns(ann_ids)
        image = self._load_image(img_id)

        boxes = []
        labels = []
        
        for annotation in coco_target:
            if annotation.get('iscrowd', 0) == 1: # 忽略 iscrowd 目标
                continue
            x, y, w, h = annotation['bbox']
            boxes.append([x, y, x + w, y + h]) # 转换为 [x_min, y_min, x_max, y_max]
            labels.append(annotation['category_id']) 
        
        # 转换为 Tensor 格式
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([img_id])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # 数据增强/转换
        if self.transform is not None:
             image = self.transform(image)

        return image, target

def collate_fn(batch):
    """用于 DataLoader 的自定义 collate function"""
    return tuple(zip(*batch))

# --- 3. 模型设计 (Model Design Requirement 2) ---

def get_faster_rcnn_model(num_classes):
    """
    使用预训练的 ResNet50-FPN 作为主干网络的 Faster R-CNN 模型。
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 替换分类器头部以适应新的类别数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# --- 4. 训练和优化函数 (Training Requirement 3) ---

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    
    print(f"\n--- Epoch {epoch} Start ---")
    
    # 使用 tqdm 包装 data_loader 以显示进度
    data_iterator = tqdm(data_loader, desc=f"Epoch {epoch} Training")
    
    for i, (images, targets) in enumerate(data_iterator):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 模型在训练模式下计算损失 (分类损失 + 回归损失)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        
        data_iterator.set_postfix({'Loss': f'{running_loss/(i+1):.4f}'})

    print(f"Epoch {epoch} Final Loss: {running_loss/len(data_loader):.4f}")


# --- 5. 可视化和评估函数 (Evaluation and Analysis Requirement 4 & Visualization) ---

def visualize_sample(dataset, index, num_samples=1, is_ground_truth=True):
    """
    可视化样本图片的 Ground Truth 标注或模型预测。
    (Requirement 1 & 4)
    """
    if num_samples > 3: num_samples = 3 # 限制数量
    
    for i in range(num_samples):
        # 使用不带增强的原始图片来保证可视化正确
        image, target = dataset[index + i] 
        
        # 将图片从 Tensor 转换为 PIL Image 或 Numpy Array
        img = (image * 255).byte().permute(1, 2, 0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        
        title = "Ground Truth Annotation"
        if not is_ground_truth:
             title = "Model Prediction"

        print(f"Image {index + i} {title} (ID: {target['image_id'].item()})")
        
        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            # 绘制矩形框
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f'Class ID: {label}', color='white', backgroundcolor='red')

        plt.title(title)
        plt.show()

def evaluate_coco(model, data_loader, device):
    """
    使用 pycocotools 进行 COCO mAP 标准评估 (Requirement 4)。
    注意: 这是一个简化骨架。在实际提交时，你需要使用 PyTorch 官方
    检测示例 (references/detection) 中的 engine.py 和 coco_eval.py
    脚本，它们包含了完整的 COCO 评估逻辑。
    """
    print("--- 正在进行 COCO 标准评估 ---")
    model.eval()
    
    # COCO 评估需要一个特殊的格式，这里只展示骨架和所需库
    
    # 1. 初始化 COCO 对象 (用于Ground Truth)
    coco_gt = data_loader.dataset.coco
    coco_dt = [] # 存储模型预测结果

    # 2. 遍历数据并收集预测结果
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Collecting Predictions"):
            images = list(image.to(device) for image in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                img_id = targets[i]["image_id"].item()
                
                # 转换预测结果为 COCO 格式
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.05: # 仅保留高置信度预测
                        # 转换 [x_min, y_min, x_max, y_max] 到 COCO [x, y, w, h]
                        x, y, xmax, ymax = box
                        w = xmax - x
                        h = ymax - y
                        
                        coco_dt.append({
                            "image_id": img_id,
                            "bbox": [x, y, w, h],
                            "score": score,
                            "category_id": int(label)
                        })
    
    # 3. 运行 COCO 评估
    if not coco_dt:
        print("没有检测到有效目标，评估失败。")
        return

    import json
    # 保存预测结果到临时 JSON 文件
    with open("results.json", "w") as f:
        json.dump(coco_dt, f)
        
    coco_results = coco_gt.loadRes("results.json")
    
    from pycocotools.cocoeval import COCOeval
    coco_eval = COCOeval(coco_gt, coco_results, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    print(f"\n✅ Mean Average Precision (mAP) for all categories: {coco_eval.stats[0]:.3f}")
    
    # 清理临时文件
    os.remove("results.json")

# --- 6. 主程序 (Main Execution) ---

def run_training_pipeline():
    # 1. 自动下载并获取路径
    IMG_DIR_TRAIN, ANN_FILE_TRAIN = setup_coco_dataset()
    
    # 2. 数据加载
    transform = Compose([
        ToTensor(), 
        RandomHorizontalFlip(0.5) # 数据增强 [cite: 8]
    ])

    dataset = CocoDetection_Custom(root=IMG_DIR_TRAIN, annFile=ANN_FILE_TRAIN, transform=transform)

    # 划分训练集和验证集
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    print(f"总数据集大小: {len(dataset)}, 训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 3. 模型初始化
    model = get_faster_rcnn_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # 4. 可视化 Ground Truth 样本 (Requirement 1 & 4)
    # 取训练集中的前三个样本进行可视化
    print("\n--- 可视化 Ground Truth 样本 (Requirement 1) ---")
    visualize_sample(train_dataset, 0, num_samples=3)

    # 5. 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print(f"\nStart training on {len(train_dataset)} images on device: {DEVICE}")

    # 6. 训练
    for epoch in range(NUM_EPOCHS):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        
        # 每个 epoch 结束时进行评估 (可选，但推荐)
        if epoch % 1 == 0:
            evaluate_coco(model, val_loader, DEVICE)
        
    # 7. 保存模型权重 (Submission Material 2)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nTraining finished. Model weights saved to {SAVE_PATH}")


if __name__ == '__main__':
    run_training_pipeline()