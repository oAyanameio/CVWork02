import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(box1, box2):
    """
    计算CIoU损失（Complete IoU）：同时考虑重叠度、中心点距离、宽高比
    :param box1: 预测框（[B, N, 4]，格式[x1,y1,x2,y2]）
    :param box2: 真实框（[B, N, 4]，格式[x1,y1,x2,y2]）
    :return: CIoU值（[B, N, 1]）
    """
    # 拆分坐标
    box1_x1, box1_y1, box1_x2, box1_y2 = box1.chunk(4, dim=-1)
    box2_x1, box2_y1, box2_x2, box2_y2 = box2.chunk(4, dim=-1)

    # 计算重叠区域面积
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # 计算预测框和真实框的面积
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
    union_area = box1_area + box2_area - inter_area + 1e-6  # 加1e-6避免除零

    # 1. 计算IoU
    iou = inter_area / union_area

    # 2. 计算中心点距离（归一化到最小外接矩形对角线）
    box1_cx = (box1_x1 + box1_x2) / 2
    box1_cy = (box1_y1 + box1_y2) / 2
    box2_cx = (box2_x1 + box2_x2) / 2
    box2_cy = (box2_y1 + box2_y2) / 2
    rho2 = (box1_cx - box2_cx) ** 2 + (box1_cy - box2_cy) ** 2  # 中心点距离平方

    # 计算最小外接矩形对角线平方
    enclose_x1 = torch.min(box1_x1, box2_x1)
    enclose_y1 = torch.min(box1_y1, box2_y1)
    enclose_x2 = torch.max(box1_x2, box2_x2)
    enclose_y2 = torch.max(box1_y2, box2_y2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-6

    # 3. 计算宽高比一致性
    box1_w = box1_x2 - box1_x1
    box1_h = box1_y2 - box1_y1
    box2_w = box2_x2 - box2_x1
    box2_h = box2_y2 - box2_y1
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(box1_w / box1_h) - torch.atan(box2_w / box2_h), 2)
    alpha = v / (1 - iou + v + 1e-6)  # 平衡因子

    # 4. CIoU = IoU - 中心点距离项 - 宽高比项
    ciou = iou - (rho2 / c2) - alpha * v
    return ciou


class YOLOLoss(nn.Module):
    """YOLOv5损失函数：分类损失 + 回归损失（CIoU） + 置信度损失"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")  # 二分类交叉熵（用于分类和置信度）

    def forward(self, preds, targets, anchors, img_size=640):
        """
        :param preds: 模型输出（3个尺度，each [B, 3, H, W, 85]）
        :param targets: 真实标签（[B, max_num_boxes, 5]，格式[x1,y1,x2,y2,cls_id]）
        :param anchors: 锚框（3个尺度，each [3, 2]，格式[w,h]）
        :param img_size: 输入图像尺寸
        :return: 总损失（分类损失 + 回归损失 + 置信度损失）
        """
        total_loss = 0.0
        device = preds[0].device
        batch_size = preds[0].shape[0]

        # 遍历3个尺度的预测结果
        for scale_idx, (pred, anchor) in enumerate(zip(preds, anchors)):
            H, W = pred.shape[2], pred.shape[3]  # 当前尺度的特征图尺寸
            num_anchors = pred.shape[1]          # 每个尺度的锚框数（3）

            # 1. 解析预测结果
            pred_xy = torch.sigmoid(pred[..., 0:2])  # xy偏移量（0~1，相对于网格）
            pred_wh = torch.exp(pred[..., 2:4])      # wh缩放因子（指数确保为正）
            pred_conf = pred[..., 4:5]               # 目标置信度（未激活）
            pred_cls = pred[..., 5:]                 # 类别概率（未激活）

            # 2. 生成网格坐标（[1, 1, H, W, 2]）
            grid_x = torch.arange(W, device=device).repeat(H, 1).view(1, 1, H, W, 1)
            grid_y = torch.arange(H, device=device).repeat(W, 1).t().view(1, 1, H, W, 1)
            grid = torch.cat([grid_x, grid_y], dim=-1)

            # 3. 计算预测框的绝对坐标（归一化到图像尺寸）
            pred_xy = (pred_xy + grid) * (img_size / W)  # 网格坐标 → 图像坐标
            pred_wh = pred_wh * anchor.view(1, num_anchors, 1, 1, 2)  # 锚框缩放 → 实际宽高
            pred_x1 = pred_xy[..., 0:1] - pred_wh[..., 0:1] / 2
            pred_y1 = pred_xy[..., 1:2] - pred_wh[..., 1:2] / 2
            pred_x2 = pred_xy[..., 0:1] + pred_wh[..., 0:1] / 2
            pred_y2 = pred_xy[..., 1:2] + pred_wh[..., 1:2] / 2
            pred_box = torch.cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)  # [B, 3, H, W, 4]

            # 4. 预处理真实标签（匹配当前尺度的目标）
            targets_box = targets[..., 0:4]  # [B, max_num_boxes, 4]
            targets_cls = targets[..., 4:5].long()  # [B, max_num_boxes, 1]
            targets_conf = torch.zeros((batch_size, num_anchors, H, W, 1), device=device)  # 置信度标签

            # 5. 锚框匹配（IoU>0.5为正样本，否则为负样本）
            # 展开预测框和真实框，计算IoU（[B, 3*H*W, max_num_boxes]）
            pred_box_flat = pred_box.view(batch_size, -1, 4)
            targets_box_flat = targets_box.unsqueeze(1).repeat(1, pred_box_flat.shape[1], 1, 1)
            iou = bbox_ciou(pred_box_flat.unsqueeze(2), targets_box_flat)  # [B, 3*H*W, max_num_boxes, 1]
            max_iou, max_idx = iou.max(dim=2)  # [B, 3*H*W, 1] → 每个预测框匹配的最优真实框

            # 标记正样本（IoU>0.5）
            pos_mask = max_iou > 0.5
            if pos_mask.any():
                # 为正样本分配真实框类别和置信度
                targets_conf_flat = targets_conf.view(batch_size, -1, 1)
                targets_conf_flat[pos_mask] = 1.0  # 正样本置信度=1
                # 展开目标类别，分配给正样本
                targets_cls_flat = targets_cls.unsqueeze(1).repeat(1, pred_box_flat.shape[1], 1)
                targets_cls_flat = targets_cls_flat.gather(2, max_idx)  # [B, 3*H*W, 1]

            # 6. 计算损失
            # 6.1 置信度损失（正样本：1，负样本：0）
            loss_conf = self.bce_loss(pred_conf.view(batch_size, -1, 1), targets_conf.view(batch_size, -1, 1))
            loss_conf = loss_conf.mean()

            # 6.2 分类损失（仅正样本计算）
            loss_cls = torch.tensor(0.0, device=device)
            if pos_mask.any():
                pred_cls_flat = pred_cls.view(batch_size, -1, self.num_classes)
                targets_cls_onehot = F.one_hot(targets_cls_flat[pos_mask].squeeze(-1), self.num_classes).float()
                loss_cls = self.bce_loss(pred_cls_flat[pos_mask], targets_cls_onehot).mean()

            # 6.3 回归损失（CIoU损失，仅正样本计算）
            loss_reg = torch.tensor(0.0, device=device)
            if pos_mask.any():
                pred_box_pos = pred_box_flat[pos_mask]
                targets_box_pos = targets_box_flat[pos_mask].gather(1, max_idx[pos_mask].unsqueeze(-1).repeat(1, 1, 4)).squeeze(1)
                loss_reg = (1 - bbox_ciou(pred_box_pos.unsqueeze(1), targets_box_pos.unsqueeze(1))).mean()

            # 累加当前尺度损失（权重平衡）
            total_loss += loss_conf + loss_cls + loss_reg

        # 平均3个尺度的损失
        return total_loss / len(preds)