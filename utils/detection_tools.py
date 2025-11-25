import torch
import torch.nn.functional as F
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.loss import bbox_ciou


def decode_preds(preds, anchors, img_size=640, conf_thres=0.3, iou_thres=0.45):
    """
    解码模型输出，筛选有效预测框（置信度过滤 + NMS）
    :param preds: 模型输出（3个尺度，each [B, 3, H, W, 85]）
    :param anchors: 锚框（3个尺度，each [3, 2]）
    :param img_size: 输入图像尺寸
    :param conf_thres: 置信度阈值（过滤低置信度预测）
    :param iou_thres: NMS IoU阈值（去除重复预测）
    :return: 解码后的预测结果（list of [N, 6]，格式[x1,y1,x2,y2,conf,cls_id]）
    """
    device = preds[0].device
    batch_results = []

    # 遍历3个尺度的预测
    for scale_idx, (pred, anchor) in enumerate(zip(preds, anchors)):
        batch_size, num_anchors, H, W, _ = pred.shape

        # 解析预测结果
        pred_xy = torch.sigmoid(pred[..., 0:2])
        pred_wh = torch.exp(pred[..., 2:4])
        pred_conf = torch.sigmoid(pred[..., 4:5])  # 置信度激活（0~1）
        pred_cls = torch.sigmoid(pred[..., 5:])    # 类别概率激活（0~1）
        pred_cls_id = torch.argmax(pred_cls, dim=-1, keepdim=True)  # 预测类别ID

        # 生成网格坐标
        grid_x = torch.arange(W, device=device).repeat(H, 1).view(1, 1, H, W, 1)
        grid_y = torch.arange(H, device=device).repeat(W, 1).t().view(1, 1, H, W, 1)
        grid = torch.cat([grid_x, grid_y], dim=-1)

        # 计算预测框绝对坐标（还原到图像尺寸）
        pred_xy = (pred_xy + grid) * (img_size / W)
        pred_wh = pred_wh * anchor.view(1, num_anchors, 1, 1, 2)
        pred_x1 = pred_xy[..., 0:1] - pred_wh[..., 0:1] / 2
        pred_y1 = pred_xy[..., 1:2] - pred_wh[..., 1:2] / 2
        pred_x2 = pred_xy[..., 0:1] + pred_wh[..., 0:1] / 2
        pred_y2 = pred_xy[..., 1:2] + pred_wh[..., 1:2] / 2

        # 拼接预测结果（[B, 3*H*W, 6]）
        pred_box = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2, pred_conf, pred_cls_id.float()], dim=-1)
        pred_box = pred_box.view(batch_size, -1, 6)

        # 按批次处理每个图像
        for b in range(batch_size):
            # 1. 置信度过滤（保留conf>conf_thres的预测框）
            box = pred_box[b]
            box = box[box[..., 4] > conf_thres]
            if len(box) == 0:
                if scale_idx == 0:  # 第一个尺度初始化结果列表
                    batch_results.append([])
                continue

            # 2. NMS（非极大值抑制，按类别分组）
            cls_ids = box[..., 5].unique()  # 当前图像的预测类别
            keep_boxes = []
            for cls_id in cls_ids:
                # 筛选当前类别的预测框
                cls_boxes = box[box[..., 5] == cls_id]
                x1, y1, x2, y2, conf, _ = cls_boxes.chunk(6, dim=-1)

                # 计算IoU
                areas = (x2 - x1) * (y2 - y1)
                # 按置信度降序排序
                order = conf.squeeze(1).argsort(descending=True)
                keep = []

                while order.numel() > 0:
                    i = order[0]
                    keep.append(i)
                    if order.numel() == 1:
                        break

                    # 计算当前框与剩余框的IoU
                    inter_x1 = torch.max(x1[i], x1[order[1:]])
                    inter_y1 = torch.max(y1[i], y1[order[1:]])
                    inter_x2 = torch.min(x2[i], x2[order[1:]])
                    inter_y2 = torch.min(y2[i], y2[order[1:]])
                    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
                    iou = inter_area / (areas[i] + areas[order[1:]] - inter_area + 1e-6)

                    # 保留IoU<iou_thres的框
                    order = order[1:][iou < iou_thres]

                # 保存当前类别的保留框
                keep_boxes.append(cls_boxes[keep])

            # 合并所有类别的保留框
            if keep_boxes:
                if scale_idx == 0:
                    batch_results.append(torch.cat(keep_boxes, dim=0))
                else:
                    batch_results[b] = torch.cat([batch_results[b], torch.cat(keep_boxes, dim=0)], dim=0)

    # 处理空结果（确保每个图像都有结果，即使为空）
    for b in range(batch_size):
        if len(batch_results[b]) == 0:
            batch_results[b] = torch.empty((0, 6), device=device)

    return batch_results


def evaluate_coco_metrics(preds, img_ids, ann_path, conf_thres=0.3, iou_thres=0.45):
    """
    计算COCO评估指标（mAP@0.5、mAP@0.5:0.95等）
    :param preds: 解码后的预测结果（list of [N, 6]）
    :param img_ids: 图像ID列表（与preds一一对应）
    :param ann_path: COCO标注文件路径
    :param conf_thres: 置信度阈值
    :param iou_thres: NMS IoU阈值
    :return: 评估指标字典（mAP@0.5、mAP@0.5:0.95、Precision、Recall）
    """
    # 初始化COCO评估对象
    coco_gt = COCO(ann_path)
    coco_results = []

    # 转换预测结果为COCO格式（[x,y,w,h]）
    for pred, img_id in zip(preds, img_ids):
        if len(pred) == 0:
            continue
        # 拆分预测结果
        x1, y1, x2, y2, conf, cls_id = pred.chunk(6, dim=-1)
        w = x2 - x1
        h = y2 - y1

        # 转换为numpy数组（COCO评估需CPU数据）
        x1 = x1.squeeze(1).cpu().numpy().astype(int)
        y1 = y1.squeeze(1).cpu().numpy().astype(int)
        w = w.squeeze(1).cpu().numpy().astype(int)
        h = h.squeeze(1).cpu().numpy().astype(int)
        conf = conf.squeeze(1).cpu().numpy()
        cls_id = cls_id.squeeze(1).cpu().numpy().astype(int)

        # 添加到COCO结果列表
        for x, y, ww, hh, c, cls in zip(x1, y1, w, h, conf, cls_id):
            coco_results.append({
                "image_id": img_id,
                "bbox": [x, y, ww, hh],
                "score": c,
                "category_id": cls
            })

    # 执行COCO评估
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # 返回关键指标（参考COCOeval.summarize()输出顺序）
    return {
        "mAP@0.5": round(coco_eval.stats[0], 4),
        "mAP@0.5:0.95": round(coco_eval.stats[1], 4),
        "Precision": round(coco_eval.stats[10], 4),  # 所有类别Precision均值
        "Recall": round(coco_eval.stats[11], 4)      # 所有类别Recall均值
    }