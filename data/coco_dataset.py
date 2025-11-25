import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from data.utils import resize_img  # 引用数据工具函数


class COCODataset(Dataset):
    def __init__(self, data_path, split="train", img_size=640, augment=True):
        """
        COCO数据集加载类
        :param data_path: COCO数据集根路径
        :param split: 数据集划分（train/val）
        :param img_size: 输入图像尺寸
        :param augment: 是否启用数据增强
        """
        self.data_path = data_path
        self.split = split
        self.img_dir = f"{data_path}/{split}2017"
        self.ann_path = f"{data_path}/annotations/instances_{split}2017.json"
        self.img_size = img_size
        self.augment = augment

        # 加载COCO标注
        self.coco = COCO(self.ann_path)
        self.img_ids = self._filter_valid_imgs()  # 筛选含目标的图像
        self.cat_id_to_name = {cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())}

    def _filter_valid_imgs(self):
        """筛选含目标的图像（排除无标注的空图像）"""
        img_ids = []
        for cat_id in self.coco.getCatIds():
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        return list(set(img_ids))  # 去重

    def _augment_data(self, img, bboxes):
        """数据增强：水平翻转、随机裁剪、亮度/对比度微调"""
        h, w = img.shape[:2]

        # 1. 随机水平翻转（概率0.5）
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]  # 同步翻转x坐标

        # 2. 随机裁剪（概率0.3，仅保留完全在裁剪区域的目标）
        if np.random.random() < 0.3 and self.augment:
            crop_h = np.random.randint(int(0.3 * h), h)
            crop_w = np.random.randint(int(0.3 * w), w)
            crop_y = np.random.randint(0, h - crop_h)
            crop_x = np.random.randint(0, w - crop_w)

            # 裁剪图像
            img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            # 调整Bounding Box
            bboxes[:, [0, 2]] -= crop_x
            bboxes[:, [1, 3]] -= crop_y
            # 过滤超出裁剪区域的目标
            mask = (bboxes[:, 0] >= 0) & (bboxes[:, 1] >= 0) & (bboxes[:, 2] <= crop_w) & (bboxes[:, 3] <= crop_h)
            bboxes = bboxes[mask]

        # 3. 亮度/对比度微调（概率0.4）
        if np.random.random() < 0.4 and self.augment:
            alpha = np.random.uniform(0.8, 1.2)  # 对比度增益
            beta = np.random.uniform(-20, 20)    # 亮度偏移
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img, bboxes

    def __getitem__(self, idx):
        # 1. 加载图像
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR→RGB（PyTorch默认格式）

        # 2. 加载Bounding Box与类别（COCO标注：[x,y,w,h]→[x1,y1,x2,y2]）
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])  # COCO类别ID（0~79）
        bboxes = np.array(bboxes, dtype=np.float32) if bboxes else np.empty((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.empty(0, dtype=np.int64)

        # 3. 数据增强
        if self.augment:
            img, bboxes = self._augment_data(img, bboxes)

        # 4. 图像Resize（保持宽高比，灰条填充）
        img, ratio, pad = resize_img(img, self.img_size)
        # 调整Bounding Box坐标（适配Resize后的图像）
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * ratio[0] + pad[0]
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * ratio[1] + pad[1]

        # 5. 格式转换（Tensor）
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # [H,W,C]→[C,H,W]，归一化到[0,1]
        bboxes = torch.from_numpy(bboxes)
        labels = torch.from_numpy(labels)

        return img, bboxes, labels, img_id  # 返回img_id用于评估

    def __len__(self):
        return len(self.img_ids)