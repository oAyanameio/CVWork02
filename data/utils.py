import cv2
import numpy as np
import torch


def resize_img(img, target_size):
    """
    保持宽高比Resize图像，用灰条（RGB：(114,114,114)）填充空白区域
    :param img: 输入图像（RGB格式，[H,W,C]）
    :param target_size: 目标尺寸（正方形，如640）
    :return: resize后图像、缩放比例、填充量（左右填充，上下填充）
    """
    h, w = img.shape[:2]
    # 计算缩放比例（取宽高缩放比例的最小值，避免图像超出目标尺寸）
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize图像
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算填充量（左右填充、上下填充，确保最终尺寸为target_size×target_size）
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    pad = (pad_w, pad_h)  # (左右填充, 上下填充)

    # 填充灰条
    img = cv2.copyMakeBorder(
        img, pad_h, target_size - new_h - pad_h, pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return img, (scale, scale), pad


def collate_fn(batch):
    """
    DataLoader的collate_fn：处理变长的Bounding Box和Labels（用0填充至最大长度）
    :param batch: 批量数据（list of (img, bboxes, labels, img_id)）
    :return: 堆叠后的img、填充后的bboxes、填充后的labels、img_id列表
    """
    imgs = []
    bboxes_list = []
    labels_list = []
    img_ids = []

    # 分离批量数据
    for img, bboxes, labels, img_id in batch:
        imgs.append(img)
        bboxes_list.append(bboxes)
        labels_list.append(labels)
        img_ids.append(img_id)

    # 堆叠图像（图像尺寸一致，可直接stack）
    imgs = torch.stack(imgs, dim=0)

    # 计算最大目标数（用于填充）
    max_num_boxes = max(len(bboxes) for bboxes in bboxes_list) if bboxes_list else 0

    # 填充Bounding Box（[B, max_num_boxes, 4]）
    bboxes_padded = []
    for bboxes in bboxes_list:
        pad_num = max_num_boxes - len(bboxes)
        if pad_num > 0:
            pad_boxes = torch.zeros((pad_num, 4), device=bboxes.device)
            bboxes_padded.append(torch.cat([bboxes, pad_boxes], dim=0))
        else:
            bboxes_padded.append(bboxes)
    bboxes_padded = torch.stack(bboxes_padded, dim=0)

    # 填充Labels（[B, max_num_boxes]）
    labels_padded = []
    for labels in labels_list:
        pad_num = max_num_boxes - len(labels)
        if pad_num > 0:
            pad_labels = torch.zeros(pad_num, dtype=torch.long, device=labels.device)
            labels_padded.append(torch.cat([labels, pad_labels], dim=0))
        else:
            labels_padded.append(labels)
    labels_padded = torch.stack(labels_padded, dim=0)

    return imgs, bboxes_padded, labels_padded, img_ids