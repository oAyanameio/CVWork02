# Assignment2: 完整目标检测单文件（Faster R-CNN, VOC）—— 符合作业文档要求（除报告）

"""
此脚本实现并满足课程作业要求的所有代码部分（除实验报告外）：
1. 数据预处理（含适用于检测的增强）并提供按类别的可视化样例（带 GT 框）
2. 模型设计：Faster R-CNN（ResNet50-FPN），包含 backbone、RPN、ROI heads（分类+bbox回归）
3. 训练与优化：输出分类/回归/RPN 等子损失，保存 checkpoint，并绘制训练损失曲线
4. 评估与分析：实现 VOC-style mAP@0.5，保存 per-class AP；可视化困难样例（小目标、遮挡）
5. 自动下载 VOC 数据集；训练/评估/可视化的命令行模式；保存配置与可视化结果

使用方法：
    python assignment2_full.py --mode train
    python assignment2_full.py --mode eval
    python assignment2_full.py --mode visualize_samples
    python assignment2_full.py --mode analyze_hard

请将此文件命名为 assignment2_full.py 并放在你的项目根目录，确保有网络以便自动下载 VOC（如果缺失）。
"""

import os
import sys
import json
import time
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 服务器模式下使用
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms.functional as F
from torchvision.datasets import VOCDetection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from tqdm import tqdm
from data.dataset import COCODataset
# ---------------------------
# 配置（可修改）
# ---------------------------
CFG = {
    "dataset_root": "./VOCdevkit",
    "year": "2012",
    "train_set": "train",
    "val_set": "val",
    "num_classes": 21,  # VOC 20 + background
    "batch_size": 4,
    "num_workers": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "num_epochs": 12,
    "save_dir": "./outputs",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "min_size": 600,
    "visualize_score_thresh": 0.5,
    "small_object_area": 32*32,  # 小目标面积阈值（像素）
    "occlusion_iou_thresh": 0.5
}

os.makedirs(CFG["save_dir"], exist_ok=True)

# ---------------------------
# 类别映射
# ---------------------------
VOC_CLASSES = [
    "background",
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
CLASS_TO_IDX = {c:i for i,c in enumerate(VOC_CLASSES)}

# ---------------------------
# 数据增强（适用于目标检测）
# ---------------------------
class DetectionAugment:
    """包含适用于目标检测的增强：
    - 随机水平翻转（保持 bbox）
    - 随机缩放（短边 resize）
    - 随机裁剪（保留目标中心在裁剪区内）
    - 亮度/对比度扰动
    """
    def __init__(self, min_size=600, hflip_prob=0.5, color_jitter=0.2, crop_prob=0.3):
        self.min_size = min_size
        self.hflip_prob = hflip_prob
        self.color_jitter = color_jitter
        self.crop_prob = crop_prob

    def __call__(self, image: Image.Image, target: dict):
        # 随机水平翻转
        if random.random() < self.hflip_prob:
            image = F.hflip(image)
            w, _ = image.size
            if target["boxes"].numel() > 0:
                target["boxes"][:, [0,2]] = w - target["boxes"][:, [2,0]]

        # 随机短边缩放
        image, target = self.resize(image, target, short_side=self.min_size)

        # 随机裁剪
        if random.random() < self.crop_prob:
            image, target = self.random_crop(image, target)

        # 颜色扰动
        if random.random() < 0.8:
            b = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            c = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            image = F.adjust_brightness(image, b)
            image = F.adjust_contrast(image, c)

        # 转 Tensor
        image = F.to_tensor(image)
        return image, target

    def resize(self, image, target, short_side=600):
        w, h = image.size
        scale = short_side / min(h, w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        image = F.resize(image, (new_h, new_w))
        if target["boxes"].numel() > 0:
            target["boxes"] = target["boxes"] * scale
        return image, target

    def random_crop(self, image, target, min_ratio=0.6):
        w, h = image.size
        cw = int(w * random.uniform(min_ratio, 1.0))
        ch = int(h * random.uniform(min_ratio, 1.0))
        if cw == w and ch == h:
            return image, target
        left = random.randint(0, w - cw)
        top = random.randint(0, h - ch)
        right = left + cw
        bottom = top + ch
        image = image.crop((left, top, right, bottom))
        if target["boxes"].numel() == 0:
            return image, target
        boxes = target["boxes"].clone()
        cx = 0.5 * (boxes[:,0] + boxes[:,2])
        cy = 0.5 * (boxes[:,1] + boxes[:,3])
        mask = (cx >= left) & (cx <= right) & (cy >= top) & (cy <= bottom)
        boxes = boxes[mask]
        labels = target["labels"][mask]
        if boxes.numel() > 0:
            boxes[:, [0,2]] = boxes[:, [0,2]] - left
            boxes[:, [1,3]] = boxes[:, [1,3]] - top
        else:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        target["boxes"] = boxes
        target["labels"] = labels
        return image, target

# ---------------------------
# VOC wrapper（自动下载）
# ---------------------------

def parse_voc_ann(ann):
    objs = ann["annotation"].get("object", [])
    if isinstance(objs, dict):
        objs = [objs]
    boxes = []
    labels = []
    iscrowd = []
    for o in objs:
        b = o["bndbox"]
        xmin = float(b["xmin"]) ; ymin = float(b["ymin"]) ; xmax = float(b["xmax"]) ; ymax = float(b["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX.get(o["name"], 0))
        iscrowd.append(int(o.get("difficult", "0")))
    boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
    iscrowd = torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64)
    return boxes, labels, iscrowd

class VOCWrapperAuto(Dataset):
    def __init__(self, root, year='2012', image_set='train', transforms=None):
        # torchvision 会自动下载 dataset 到 root/VOC{year}
        self.base = VOCDetection(root, year=year, image_set=image_set, download=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, ann = self.base[idx]
        boxes, labels, iscrowd = parse_voc_ann(ann)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "iscrowd": iscrowd}
        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# ---------------------------
# 模型建造：Faster R-CNN（ResNet50-FPN）
# ---------------------------

def build_fasterrcnn(num_classes=21, pretrained_backbone=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=pretrained_backbone)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ---------------------------
# 评估：mAP@0.5（VOC 风格，简化实现）
# ---------------------------

def compute_ap(rec, prec):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size-1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx+1] - mrec[idx]) * mpre[idx+1])
    return ap


def evaluate_map50(preds, gts, num_classes=21):
    APs = []
    for cls in range(1, num_classes):
        detections = []
        gt_per_image = {}
        npos = 0
        for i, (p, g) in enumerate(zip(preds, gts)):
            gt_mask = (g['labels'] == cls)
            gt_boxes = g['boxes'][gt_mask]
            gt_per_image[i] = {'boxes': gt_boxes.copy(), 'detected': np.zeros(len(gt_boxes), dtype=bool)}
            npos += len(gt_boxes)
            pred_mask = (p['labels'] == cls)
            for box, score in zip(p['boxes'][pred_mask], p['scores'][pred_mask]):
                detections.append({'image_id': i, 'box': box, 'score': float(score)})
        if len(detections) == 0:
            APs.append(0.0)
            continue
        detections = sorted(detections, key=lambda x: -x['score'])
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))
        for idx, det in enumerate(detections):
            imgid = det['image_id']
            bb = det['box'].reshape(1,4)
            gt_entry = gt_per_image[imgid]
            gt_boxes = gt_entry['boxes']
            if gt_boxes.shape[0] == 0:
                FP[idx] = 1
                continue
            ious = box_iou(torch.tensor(bb), torch.tensor(gt_boxes)).numpy().flatten()
            max_iou_idx = np.argmax(ious)
            if ious[max_iou_idx] >= 0.5 and not gt_entry['detected'][max_iou_idx]:
                TP[idx] = 1
                gt_entry['detected'][max_iou_idx] = True
            else:
                FP[idx] = 1
        fp = np.cumsum(FP)
        tp = np.cumsum(TP)
        rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = compute_ap(rec, prec) if npos > 0 else 0.0
        APs.append(ap)
    mAP = float(np.mean(APs))
    return mAP, APs

# ---------------------------
# 可视化工具
# ---------------------------

def save_figure(img_tensor, gt_boxes=None, pred_boxes=None, pred_scores=None, pred_labels=None, out_path=None):
    img = img_tensor.permute(1,2,0).numpy()
    fig, ax = plt.subplots(1, figsize=(12,8))
    ax.imshow(img)
    if gt_boxes is not None:
        for b in gt_boxes:
            xmin, ymin, xmax, ymax = b
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
    if pred_boxes is not None:
        for b, s, l in zip(pred_boxes, pred_scores, pred_labels):
            xmin, ymin, xmax, ymax = b
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin-5, f"{VOC_CLASSES[int(l)]}:{s:.2f}", color='red', fontsize=10, backgroundcolor='white')
    ax.axis('off')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ---------------------------
# 按类别可视化样例（保存）
# ---------------------------

def visualize_samples_per_class(dataset, out_dir, samples_per_class=3):
    os.makedirs(out_dir, exist_ok=True)
    examples = defaultdict(list)
    for idx in range(len(dataset)):
        img, t = dataset[idx]
        labels = t['labels'].numpy()
        boxes = t['boxes'].numpy()
        for cls in np.unique(labels):
            examples[int(cls)].append((idx, img, boxes[labels==cls], labels[labels==cls]))
    for cls in range(1, len(VOC_CLASSES)):
        arr = examples.get(cls, [])
        for i, (idx, img, b, l) in enumerate(arr[:samples_per_class]):
            out = os.path.join(out_dir, f"class_{cls}_{VOC_CLASSES[cls]}_img{idx}_sample{i}.png")
            save_figure(img, gt_boxes=b, out_path=out)
    print(f"Saved class samples to {out_dir}")

# ---------------------------
# 训练主流程（包含详细子损失记录）
# ---------------------------

def train(cfg):
    device = torch.device(cfg['device'])
    print('Device:', device)
    transforms = DetectionAugment(min_size=cfg['min_size'])
    dataset = VOCWrapperAuto(cfg['dataset_root'], year=cfg['year'], image_set=cfg['train_set'], transforms=transforms)
    dataset_val = VOCWrapperAuto(cfg['dataset_root'], year=cfg['year'], image_set=cfg['val_set'], transforms=DetectionAugment(min_size=cfg['min_size'], hflip_prob=0.0, crop_prob=0.0))

    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'], collate_fn=collate_fn)
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=cfg['num_workers'], collate_fn=collate_fn)

    model = build_fasterrcnn(num_classes=cfg['num_classes'])
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    train_losses = []
    classifier_losses = []
    boxreg_losses = []
    objectness_losses = []
    rpn_box_losses = []
    val_maps = []

    best_map = 0.0

    for epoch in tqdm(range(cfg['num_epochs']), desc='Epochs'):
        model.train()
        epoch_losses = []
        # batch-level progress
        for images, targets in tqdm(loader, desc=f'Epoch {epoch} Training'):
            images = [img.to(device) for img in images]
            targs = []
            for t in targets:
                d = {k: v.to(device) for k,v in t.items()}
                targs.append(d)
            loss_dict = model(images, targs)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # 记录子损失
            train_losses.append(float(losses.item()))
            epoch_losses.append(float(losses.item()))
            classifier_losses.append(float(loss_dict.get('loss_classifier', torch.tensor(0.0)).item()))
            boxreg_losses.append(float(loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()))
            objectness_losses.append(float(loss_dict.get('loss_objectness', torch.tensor(0.0)).item()))
            rpn_box_losses.append(float(loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item()))

        scheduler.step()
        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print(f"Epoch {epoch} average loss: {avg_loss:.4f}")

        # 保存 checkpoint
        ckpt = os.path.join(cfg['save_dir'], f'fasterrcnn_epoch_{epoch}.pth')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'cfg': cfg}, ckpt)
        print('Saved checkpoint:', ckpt)

        # 在验证集上做快速评估
        print('Running quick evaluation on val set...')
        preds, gts = predict_on_loader(model, loader_val, device, score_thresh=0.05)
        mAP, APs = evaluate_map50(preds, gts, num_classes=cfg['num_classes'])
        val_maps.append(mAP)
        print(f'Validation mAP@0.5: {mAP:.4f}')

        # 保存最优模型
        if mAP > best_map:
            best_map = mAP
            best_path = os.path.join(cfg['save_dir'], 'fasterrcnn_best.pth')
            torch.save(model.state_dict(), best_path)
            print('Saved new best model:', best_path)

        # 每个 epoch 保存训练曲线图
        plot_training_curves(train_losses, classifier_losses, boxreg_losses, objectness_losses, rpn_box_losses, cfg['save_dir'])

    # 训练结束
    final_path = os.path.join(cfg['save_dir'], 'fasterrcnn_final.pth')
    torch.save(model.state_dict(), final_path)
    print('Training finished. Final model saved to', final_path)

    # 保存训练日志与配置
    log = {'train_losses': train_losses, 'val_maps': val_maps, 'cfg': cfg, 'timestamp': datetime.now().isoformat()}
    with open(os.path.join(cfg['save_dir'], 'training_log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print('Saved training_log.json')

    return model

# ---------------------------
# 预测辅助（在 loader 上推理，返回 preds, gts）
# ---------------------------

def predict_on_loader(model, data_loader, device, score_thresh=0.05, max_images=None):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for i, (images, targets) in enumerate(tqdm(data_loader, desc='Predicting')):
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, t in zip(outputs, targets):
                boxes = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()
                labels = out['labels'].cpu().numpy()
                sel = scores >= score_thresh
                preds.append({'boxes': boxes[sel], 'scores': scores[sel], 'labels': labels[sel]})
                gts.append({'boxes': t['boxes'].numpy(), 'labels': t['labels'].numpy()})
            if max_images and len(preds) >= max_images:
                break
    return preds, gts

# ---------------------------
# 绘制训练曲线
# ---------------------------

def plot_training_curves(train_losses, cls_losses, box_losses, obj_losses, rpn_box_losses, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 平滑（简单移动平均）
    def smooth(x, w=50):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w)/w, mode='valid')
    plt.figure()
    plt.plot(smooth(train_losses))
    plt.title('Total Loss (smoothed)')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.savefig(os.path.join(out_dir, 'loss_total.png'))
    plt.close()

    plt.figure()
    plt.plot(smooth(cls_losses))
    plt.title('Classifier Loss (smoothed)')
    plt.savefig(os.path.join(out_dir, 'loss_classifier.png'))
    plt.close()

    plt.figure()
    plt.plot(smooth(box_losses))
    plt.title('Box Regression Loss (smoothed)')
    plt.savefig(os.path.join(out_dir, 'loss_boxreg.png'))
    plt.close()

    plt.figure()
    plt.plot(smooth(obj_losses))
    plt.title('Objectness Loss (smoothed)')
    plt.savefig(os.path.join(out_dir, 'loss_objectness.png'))
    plt.close()

    plt.figure()
    plt.plot(smooth(rpn_box_losses))
    plt.title('RPN Box Loss (smoothed)')
    plt.savefig(os.path.join(out_dir, 'loss_rpn_box.png'))
    plt.close()

    print('Saved training curve images to', out_dir)

# ---------------------------
# 评估模式入口（保存 per-class AP 与若干可视化）
# ---------------------------

def eval_mode(cfg, model_path=None, save_visuals=True, vis_n=50):
    device = torch.device(cfg['device'])
    dataset_val = VOCWrapperAuto(cfg['dataset_root'], year=cfg['year'], image_set=cfg['val_set'], transforms=DetectionAugment(min_size=cfg['min_size'], hflip_prob=0.0, crop_prob=0.0))
    loader_val = DataLoader(dataset_val, batch_size=2, shuffle=False, num_workers=cfg['num_workers'], collate_fn=collate_fn)

    model = build_fasterrcnn(num_classes=cfg['num_classes'])
    if model_path is None:
        model_path = os.path.join(cfg['save_dir'], 'fasterrcnn_final.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    preds, gts = predict_on_loader(model, loader_val, device, score_thresh=0.05, max_images=None)
    mAP, APs = evaluate_map50(preds, gts, num_classes=cfg['num_classes'])
    print(f'Final mAP@0.5 = {mAP:.4f}')

    # 保存 per-class AP
    ap_out = {VOC_CLASSES[i+1]: float(APs[i]) for i in range(len(APs))}
    with open(os.path.join(cfg['save_dir'], 'per_class_AP.json'), 'w') as f:
        json.dump({'mAP@0.5': mAP, 'per_class_AP': ap_out}, f, indent=2)
    print('Saved per_class_AP.json')

    # 保存若干可视化
    if save_visuals:
        os.makedirs(os.path.join(cfg['save_dir'], 'eval_visuals'), exist_ok=True)
        model.eval()
        cnt = 0
        with torch.no_grad():
            for idx in range(len(dataset_val)):
                img, t = dataset_val[idx]
                out = model([img.to(device)])[0]
                boxes = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()
                labels = out['labels'].cpu().numpy()
                sel = scores >= cfg['visualize_score_thresh']
                save_figure(img, gt_boxes=t['boxes'].numpy(), pred_boxes=boxes[sel], pred_scores=scores[sel], pred_labels=labels[sel], out_path=os.path.join(cfg['save_dir'], 'eval_visuals', f'val_{idx}.png'))
                cnt += 1
                if cnt >= vis_n:
                    break
        print('Saved evaluation visuals to', os.path.join(cfg['save_dir'], 'eval_visuals'))

    return mAP, APs

# ---------------------------
# 难例分析：小目标与遮挡
# ---------------------------

def analyze_hard_cases(cfg, model_path=None, save_n=100):
    device = torch.device(cfg['device'])
    ds = VOCWrapperAuto(cfg['dataset_root'], year=cfg['year'], image_set=cfg['val_set'], transforms=DetectionAugment(min_size=cfg['min_size'], hflip_prob=0.0, crop_prob=0.0))
    model = build_fasterrcnn(num_classes=cfg['num_classes'])
    if model_path is None:
        model_path = os.path.join(cfg['save_dir'], 'fasterrcnn_final.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    os.makedirs(os.path.join(cfg['save_dir'], 'hard_cases'), exist_ok=True)
    cnt = 0
    model.eval()
    with torch.no_grad():
        for idx in range(len(ds)):
            img, t = ds[idx]
            gt_boxes = t['boxes'].numpy()
            gt_labels = t['labels'].numpy()
            # 小目标检测：任何 gt box 面积小于阈值
            areas = (gt_boxes[:,2]-gt_boxes[:,0]) * (gt_boxes[:,3]-gt_boxes[:,1]) if gt_boxes.size>0 else np.array([])
            small_mask = areas < cfg['small_object_area']
            # 遮挡检测：gt box 之间有较大 IOU
            occ_mask = np.zeros(len(gt_boxes), dtype=bool)
            if len(gt_boxes) > 1:
                iou_matrix = box_iou(torch.tensor(gt_boxes), torch.tensor(gt_boxes)).numpy()
                for i in range(len(gt_boxes)):
                    # ignore self
                    max_iou = np.max(np.delete(iou_matrix[i], i)) if len(gt_boxes)>1 else 0
                    if max_iou >= cfg['occlusion_iou_thresh']:
                        occ_mask[i] = True
            hard = small_mask | occ_mask
            if hard.any():
                out = model([img.to(device)])[0]
                boxes = out['boxes'].cpu().numpy()
                scores = out['scores'].cpu().numpy()
                labels = out['labels'].cpu().numpy()
                sel = scores >= cfg['visualize_score_thresh']
                save_figure(img, gt_boxes=gt_boxes, pred_boxes=boxes[sel], pred_scores=scores[sel], pred_labels=labels[sel], out_path=os.path.join(cfg['save_dir'], 'hard_cases', f'idx_{idx}.png'))
                cnt += 1
                if cnt >= save_n:
                    break
    print('Saved hard case visuals to', os.path.join(cfg['save_dir'], 'hard_cases'))

# ---------------------------
# 生成按类别的 GT 可视化与样例（供报告使用）
# ---------------------------

def generate_class_visuals(cfg, samples_per_class=5):
    ds = COCODataset(
        img_dir=cfg['data']['val_img_dir'],  # COCO验证集图像路径
        ann_path=cfg['data']['val_ann_path'],  # COCO验证集标注路径
        img_size=cfg['data']['img_size'],
        augment=False  # 可视化无需增强
    )
    out_dir = os.path.join(cfg['save_dir'], 'class_samples')
    os.makedirs(out_dir, exist_ok=True)
    examples = defaultdict(list)
    for idx in range(len(ds)):
        img, t = ds[idx]
        labels = t['labels'].numpy()
        boxes = t['boxes'].numpy()
        for cls in np.unique(labels):
            examples[int(cls)].append((idx, img, boxes[labels==cls], labels[labels==cls]))
    for cls in range(1, len(VOC_CLASSES)):
        arr = examples.get(cls, [])
        for i, (idx, img, b, l) in enumerate(arr[:samples_per_class]):
            out = os.path.join(out_dir, f'class_{cls}_{VOC_CLASSES[cls]}_img{idx}_sample{i}.png')
            save_figure(img, gt_boxes=b, out_path=out)
    print('Saved class sample visuals to', out_dir)

# ---------------------------
# CLI 入口
# ---------------------------

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train','eval','visualize_samples','analyze_hard','full_pipeline'], help='运行模式')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=CFG['save_dir'])
    parser.add_argument('--samples_per_class', type=int, default=3)
    parser.add_argument('--vis_n', type=int, default=50)
    parser.add_argument('--save_n', type=int, default=100)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    CFG['save_dir'] = args.save_dir
    os.makedirs(CFG['save_dir'], exist_ok=True)
    # save config
    with open(os.path.join(CFG['save_dir'], 'config_used.json'), 'w') as f:
        json.dump(CFG, f, indent=2)

    if args.mode == 'train':
        print('Starting training with config:')
        print(json.dumps(CFG, indent=2))
        model = train(CFG)
    elif args.mode == 'eval':
        print('Running evaluation...')
        eval_mode(CFG, model_path=args.model_path, save_visuals=True, vis_n=args.vis_n)
    elif args.mode == 'visualize_samples':
        print('Generating class-wise GT visualizations...')
        generate_class_visuals(CFG, samples_per_class=args.samples_per_class)
    elif args.mode == 'analyze_hard':
        print('Analyzing hard cases (small/occluded) ...')
        analyze_hard_cases(CFG, model_path=args.model_path, save_n=args.save_n)
    elif args.mode == 'full_pipeline':
        print('Running full pipeline: visualize_samples → train → eval → analyze_hard')
        generate_class_visuals(CFG, samples_per_class=args.samples_per_class)
        model = train(CFG)
        eval_mode(CFG, model_path=args.model_path, save_visuals=True, vis_n=args.vis_n)
        analyze_hard_cases(CFG, model_path=args.model_path, save_n=args.save_n)
        print('Analyzing hard cases (small/occluded) ...')
        analyze_hard_cases(CFG, model_path=args.model_path, save_n=args.save_n)
    elif args.mode == 'full_pipeline':
        print('Running full pipeline: visualize_samples → train → eval → analyze_hard')
        generate_class_visuals(CFG, samples_per_class=args.samples_per_class)
        model = train(CFG)
        eval_mode(CFG, model_path=args.model_path, save_visuals=True, vis_n=args.vis_n)
        analyze_hard_cases(CFG, model_path=args.model_path, save_n=args.save_n)
        print('Analyzing hard cases (small/occluded) ...')
        analyze_hard_cases(CFG, model_path=args.model_path, save_n=args.save_n)
    else:
        print('Unknown mode. Available: train, eval, visualize_samples, analyze_hard, full_pipeline')
