import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """计算CIoU（Complete IoU）损失：比IoU更优，考虑重叠度、中心点距离、宽高比"""
    # 解析bounding box（x1,y1,x2,y2）
    box1_x1, box1_y1, box1_x2, box1_y2 = box1.chunk(4, dim=-1)
    box2_x1, box2_y1, box2_x2, box2_y2 = box2.chunk(4, dim=-1)

    # 1. 计算重叠区域面积
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0.0) * torch.clamp(inter_y2 - inter_y1, min=0.0)

    # 2. 计算预测框和真实框的面积
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area + 1e-6  # 加1e-6避免除0

    # 3. 计算IoU
    iou = inter_area / union_area

    # 4. 计算中心点距离（归一化到最小外接矩形对角线）
    box1_cx = (box1_x1 + box1_x2) / 2
    box1_cy = (box1_y1 + box1_y2) / 2
    box2_cx = (box2_x1 + box2_x2) / 2
    box2_cy = (box2_y1 + box2_y2) / 2
    rho2 = (box1_cx - box2_cx) ** 2 + (box1_cy - box2_cy) ** 2  # 中心点距离平方

    # 5. 计算最小外接矩形对角线平方
    enclose_x1 = torch.min(box1_x1, box2_x1)
    enclose_y1 = torch.min(box1_y1, box2_y1)
    enclose_x2 = torch.max(box1_x2, box2_x2)
    enclose_y2 = torch.max(box1_y2, box2_y2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-6

    # 6. 计算宽高比一致性
    box1_w = box1_x2 - box1_x1
    box1_h = box1_y2 - box1_y1
    box2_w = box2_x2 - box2_x1
    box2_h = box2_y2 - box2_y1
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(box1_w / box1_h) - torch.atan(box2_w / box2_h), 2)
    alpha = v / (1 - iou + v + 1e-6)  # 平衡因子

    # 7. CIoU = IoU - 中心点距离项 - 宽高比项
    ciou = iou - (rho2 / c2) - alpha * v
    return ciou


class YOLOLoss(nn.Module):
    """YOLOv5损失函数：分类损失 + 回归损失（CIoU） + 置信度损失"""
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")  # 二分类交叉熵（适用于多类别）

    def forward(
            self, preds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            targets: torch.Tensor, anchors: torch.Tensor, img_size: int = 640
    ) -> torch.Tensor:
        """
        Args:
            preds: 模型输出（3个尺度，each [B, 3, H, W, 85]）
            targets: 真实标签（[B, N, 5]，N=最大目标数，5=[x1,y1,x2,y2,cls_id]）
            anchors: 锚框（[3, 3, 2]，3个尺度×3个锚框×2个维度（w,h））
            img_size: 输入图像尺寸
        Returns:
            total_loss: 总损失（分类损失 + 回归损失 + 置信度损失）
        """
        total_loss = 0.0
        device = preds[0].device
        batch_size = preds[0].shape[0]

        # 遍历3个尺度的预测结果
        for scale_idx, (pred, anchor) in enumerate(zip(preds, anchors)):
            # 1. 解析当前尺度的预测结果
            num_anchors = pred.shape[1]
            grid_h, grid_w = pred.shape[2], pred.shape[3]
            stride = img_size // grid_w  # 当前尺度的下采样步长（32/16/8）

            # 预测框解码：从偏移量转换为绝对坐标
            pred_xy = torch.sigmoid(pred[..., 0:2])  # xy偏移量（0~1）
            pred_wh = torch.exp(pred[..., 2:4])      # wh缩放因子（指数确保为正）
            pred_conf = pred[..., 4:5]               # 目标置信度（未激活）
            pred_cls = pred[..., 5:]                 # 类别概率（未激活）

            # 生成网格坐标（H×W网格，用于计算绝对坐标）
            grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w, 1)
            grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w, 1)
            grid = torch.cat([grid_x, grid_y], dim=-1)  # [1, 1, H, W, 2]

            # 计算绝对坐标（归一化到[0, img_size]）
            pred_xy = (pred_xy + grid) * stride  # xy坐标（网格坐标 + 偏移量）× 步长
            pred_wh = pred_wh * anchor.view(1, num_anchors, 1, 1, 2)  # wh坐标（缩放因子 × 锚框）
            pred_x1 = pred_xy[..., 0:1] - pred_wh[..., 0:1] / 2
            pred_y1 = pred_xy[..., 1:2] - pred_wh[..., 1:2] / 2
            pred_x2 = pred_xy[..., 0:1] + pred_wh[..., 0:1] / 2
            pred_y2 = pred_xy[..., 1:2] + pred_wh[..., 1:2] / 2
            pred_box = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)  # [B, 3, H, W, 4]

            # 2. 匹配真实标签与锚框（确定正样本）
            target_box = targets[..., 0:4]  # [B, N, 4]
            target_cls = targets[..., 4:5].long()  # [B, N, 1]
            target_conf = torch.zeros((batch_size, num_anchors, grid_h, grid_w, 1), device=device)  # [B, 3, H, W, 1]

            # 遍历每个样本的真实目标
            for b in range(batch_size):
                if len(target_box[b]) == 0:
                    continue  # 无目标样本跳过

                # 计算当前样本的真实目标与所有锚框的IoU
                target_box_b = target_box[b].unsqueeze(1).unsqueeze(2).unsqueeze(3)  # [N, 1, 1, 1, 4]
                pred_box_b = pred_box[b].unsqueeze(0)  # [1, 3, H, W, 4]
                iou = bbox_ciou(pred_box_b, target_box_b)  # [N, 3, H, W, 1]

                # 选择IoU最大的锚框作为正样本（每个目标匹配一个锚框）
                max_iou, max_idx = iou.max(dim=1, keepdim=True)  # [N, 1, H, W, 1]
                mask = max_iou > 0.5  # IoU>0.5的锚框视为正样本
                if mask.any():
                    # 更新正样本的置信度标签（1）和类别标签
                    target_conf[b] = torch.where(mask, torch.ones_like(target_conf[b]), target_conf[b])
                    # 记录正样本的类别（简化版，完整实现需匹配目标与网格位置）

        # 3. 计算损失（分类损失 + 回归损失 + 置信度损失）
        # 置信度损失（正样本：1，负样本：0）
        loss_conf = self.bce_loss(pred_conf, target_conf).mean()
        # 分类损失（仅正样本计算）
        loss_cls = self.bce_loss(pred_cls, F.one_hot(target_cls, self.num_classes).float()).mean()
        # 回归损失（仅正样本计算，用1-CIoU作为损失）
        pos_mask = target_conf == 1
        loss_reg = (1 - bbox_ciou(pred_box[pos_mask], target_box[pos_mask])).mean()

        # 累加当前尺度的损失（3个尺度平均）
        total_loss += (loss_conf + loss_cls + loss_reg) / len(preds)
        return total_loss