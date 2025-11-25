import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO  # COCO数据集解析
from typing import Tuple, List


class COCODataset(Dataset):
    """COCO数据集类（支持数据增强与bounding box同步调整）"""
    def __init__(self, img_dir: str, ann_path: str, img_size: int = 640, augment: bool = True):
        self.img_dir = img_dir
        self.ann_path = ann_path
        self.img_size = img_size
        self.augment = augment
        self.coco = self._load_coco_annotations()  # 加载COCO标注
        self.img_ids = self._filter_valid_imgs()   # 过滤无标注的图像

    def _load_coco_annotations(self) -> COCO:
        """加载COCO标注文件，返回COCO对象"""
        coco = COCO(self.ann_path)
        return coco

    def _filter_valid_imgs(self) -> List[int]:
        """过滤含目标的图像（排除空图像）"""
        img_ids = []
        for cat_id in self.coco.getCatIds():
            img_ids.extend(self.coco.getImgIds(catIds=[cat_id]))
        return list(set(img_ids))  # 去重

    def _augment_data(self, img: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强：保证bounding box与图像同步变换"""
        h, w = img.shape[:2]

        # 1. 随机水平翻转（概率0.5）
        if np.random.random() < 0.5:
            img = cv2.flip(img, 1)
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]  # 翻转x坐标

        # 2. 随机裁剪（保留完全包含的目标）
        if np.random.random() < 0.3:
            crop_h = np.random.randint(int(0.3 * h), h)
            crop_w = np.random.randint(int(0.3 * w), w)
            crop_y = np.random.randint(0, h - crop_h)
            crop_x = np.random.randint(0, w - crop_w)

            # 裁剪图像
            img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            # 调整bounding box
            bboxes[:, [0, 2]] -= crop_x
            bboxes[:, [1, 3]] -= crop_y
            # 过滤超出裁剪区域的目标
            mask = (bboxes[:, 0] >= 0) & (bboxes[:, 1] >= 0) & (bboxes[:, 2] <= crop_w) & (bboxes[:, 3] <= crop_h)
            bboxes = bboxes[mask]

        # 3. 亮度/对比度微调
        if np.random.random() < 0.4:
            alpha = np.random.uniform(0.8, 1.2)  # 对比度增益
            beta = np.random.uniform(-20, 20)    # 亮度偏移
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        return img, bboxes

    def _resize_with_pad(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """保持宽高比resize，用灰条填充空白（避免目标变形）"""
        h, w = img.shape[:2]
        scale = min(self.img_size / w, self.img_size / h)  # 缩放比例
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize图像
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 计算填充量（上下左右）
        pad_w = (self.img_size - new_w) // 2
        pad_h = (self.img_size - new_h) // 2
        # 填充灰条（RGB：(114,114,114)，YOLO默认填充色）
        img = cv2.copyMakeBorder(
            img, pad_h, self.img_size - new_h - pad_h, pad_w, self.img_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return img, scale, (pad_w, pad_h)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """返回：图像Tensor、bounding box Tensor、类别Tensor、图像ID"""
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.img_dir}/{img_info['file_name']}"

        # 1. 加载图像（BGR→RGB）
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. 加载bounding box与类别（COCO标注：[x,y,w,h]→转换为[x1,y1,x2,y2]）
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes.append([x1, y1, x2, y2])
            labels.append(ann["category_id"])  # COCO类别ID（0~79）

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # 3. 数据增强（训练阶段启用）
        if self.augment:
            img, bboxes = self._augment_data(img, bboxes)

        # 4. Resize与填充
        img, scale, (pad_w, pad_h) = self._resize_with_pad(img)
        # 调整bounding box坐标（适配resize后的图像）
        if len(bboxes) > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * scale + pad_w
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * scale + pad_h

        # 5. 格式转换（numpy→Tensor，[H,W,C]→[C,H,W]，像素归一化到[0,1]）
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        bboxes = torch.from_numpy(bboxes) if len(bboxes) > 0 else torch.empty((0, 4))
        labels = torch.from_numpy(labels) if len(labels) > 0 else torch.empty(0, dtype=torch.int64)

        return img, bboxes, labels, img_id

    def __len__(self) -> int:
        return len(self.img_ids)


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """DataLoader的collate函数：处理变长的bounding box和类别（用0填充）"""
    imgs = []
    bboxes_list = []
    labels_list = []
    img_ids = []

    # 收集批次数据
    for img, bboxes, labels, img_id in batch:
        imgs.append(img)
        bboxes_list.append(bboxes)
        labels_list.append(labels)
        img_ids.append(img_id)

    # 计算批次内最大目标数（用于填充）
    max_num_boxes = max(len(bboxes) for bboxes in bboxes_list) if bboxes_list else 0

    # 填充bounding box和类别
    bboxes_padded = []
    labels_padded = []
    for bboxes, labels in zip(bboxes_list, labels_list):
        pad_num = max_num_boxes - len(bboxes)
        # 填充0（无目标的位置）
        bboxes_padded.append(torch.cat([bboxes, torch.zeros(pad_num, 4, device=bboxes.device)]))
        labels_padded.append(torch.cat([labels, torch.zeros(pad_num, dtype=torch.int64, device=labels.device)]))

    # 拼接为批次Tensor
    return torch.stack(imgs), torch.stack(bboxes_padded), torch.stack(labels_padded), img_ids


