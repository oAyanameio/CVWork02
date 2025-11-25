import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader


def decode_predictions(
        preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        anchors: torch.Tensor, img_size: int = 640, conf_thres: float = 0.3, iou_thres: float = 0.45
) -> List[torch.Tensor]:
    """解码模型输出，应用NMS（非极大值抑制）筛选有效预测结果"""
    device = preds[0].device
    batch_preds = []  # 存储批次内每个样本的预测结果

    # 遍历3个尺度的预测
    for scale_idx, (pred, anchor) in enumerate(zip(preds, anchors)):
        batch_size, num_anchors, grid_h, grid_w, _ = pred.shape
        stride = img_size // grid_w  # 下采样步长

        # 1. 解码预测框
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = torch.exp(pred[..., 2:4])
        pred_conf = torch.sigmoid(pred[..., 4:5])  # 激活置信度（0~1）
        pred_cls = torch.sigmoid(pred[..., 5:])    # 激活类别概率（0~1）
        pred_cls_id = torch.argmax(pred_cls, dim=-1, keepdim=True)  # 预测类别ID

        # 生成网格坐标
        grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w, 1)
        grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)

        # 计算绝对坐标（x1,y1,x2,y2）
        pred_xy = (pred_xy + grid) * stride
        pred_wh = pred_wh * anchor.view(1, num_anchors, 1, 1, 2)
        pred_x1 = pred_xy[..., 0:1] - pred_wh[..., 0:1] / 2
        pred_y1 = pred_xy[..., 1:2] - pred_wh[..., 1:2] / 2
        pred_x2 = pred_xy[..., 0:1] + pred_wh[..., 0:1] / 2
        pred_y2 = pred_xy[..., 1:2] + pred_wh[..., 1:2] / 2

        # 拼接预测结果（x1,y1,x2,y2,conf,cls_id）
        pred_box = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2, pred_conf, pred_cls_id.float()], dim=-1)
        pred_box = pred_box.view(batch_size, -1, 6)  # [B, 3*H*W, 6]

        # 2. 置信度筛选（过滤低置信预测）
        pred_box = pred_box[pred_box[..., 4] > conf_thres]

        # 3. NMS（非极大值抑制，去除重复预测）
        if len(pred_box) == 0:
            batch_preds.append(torch.empty((0, 6), device=device))
            continue

        # 按类别分组NMS（避免不同类别间的抑制）
        cls_ids = pred_box[..., 5].unique()
        keep_boxes = []
        for cls_id in cls_ids:
            cls_boxes = pred_box[pred_box[..., 5] == cls_id]
            # 计算IoU
            x1, y1, x2, y2, conf, _ = cls_boxes.chunk(6, dim=-1)
            areas = (x2 - x1) * (y2 - y1)
            # 按置信度降序排序
            order = conf.squeeze(1).argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0]
                keep.append(i)
                if order.numel() == 1:
                    break
                # 计算当前框与其他框的IoU
                inter_x1 = torch.max(x1[i], x1[order[1:]])
                inter_y1 = torch.max(y1[i], y1[order[1:]])
                inter_x2 = torch.min(x2[i], x2[order[1:]])
                inter_y2 = torch.min(y2[i], y2[order[1:]])
                inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)
                iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)
                # 保留IoU<iou_thres的框
                order = order[1:][iou < iou_thres]
            keep_boxes.append(cls_boxes[keep])

        # 拼接当前样本的所有有效预测
        if keep_boxes:
            batch_preds.append(torch.cat(keep_boxes, dim=0))
        else:
            batch_preds.append(torch.empty((0, 6), device=device))

    return batch_preds


def evaluate_coco(
        model: torch.nn.Module, dataloader: DataLoader, ann_path: str,
        anchors: torch.Tensor, device: torch.device, conf_thres: float = 0.3
) -> Dict[str, float]:
    """用COCO评估指标评估模型性能（mAP@0.5、mAP@0.5:0.95等）"""
    model.eval()
    coco_gt = COCO(ann_path)  # COCO真实标注
    coco_results = []          # 存储模型预测结果（COCO格式）

    with torch.no_grad():
        for batch in dataloader:
            imgs, bboxes, labels, img_ids = batch
            imgs = imgs.to(device)

            # 模型预测与解码
            preds = model(imgs)
            decoded_preds = decode_predictions(preds, anchors, img_size=imgs.shape[2], conf_thres=conf_thres)

            # 转换为COCO评估格式（bbox需为[x,y,w,h]）
            for img_id, pred in zip(img_ids, decoded_preds):
                if len(pred) == 0:
                    continue
                x1, y1, x2, y2, conf, cls_id = pred.chunk(6, dim=-1)
                w = x2 - x1
                h = y2 - y1

                # 转换为整数（COCO标注要求）
                x1 = x1.squeeze(1).cpu().numpy().astype(int)
                y1 = y1.squeeze(1).cpu().numpy().astype(int)
                w = w.squeeze(1).cpu().numpy().astype(int)
                h = h.squeeze(1).cpu().numpy().astype(int)
                conf = conf.squeeze(1).cpu().numpy()
                cls_id = cls_id.squeeze(1).cpu().numpy().astype(int)

                # 加入结果列表
                for x, y, ww, hh, c, cls in zip(x1, y1, w, h, conf, cls_id):
                    coco_results.append({
                        "image_id": img_id,
                        "bbox": [x, y, ww, hh],
                        "score": c,
                        "category_id": cls
                    })

    # 执行COCO评估
    coco_dt = coco_gt.loadRes(coco_results)  # 加载预测结果
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回核心评估指标
    return {
        "mAP@0.5": coco_eval.stats[0],        # IoU=0.5时的mAP
        "mAP@0.5:0.95": coco_eval.stats[1],   # IoU=0.5~0.95的平均mAP
        "Precision": coco_eval.stats[10],     # 精确率
        "Recall": coco_eval.stats[11]         # 召回率
    }