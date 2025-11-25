import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# 基础组件：卷积+BN+SiLU（YOLOv5默认激活函数）
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()  # 替代ReLU，在目标检测中性能更优
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# CSP模块（跨阶段部分连接，Backbone核心，减少计算量同时保留特征）
class CSPBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_conv: int = 1):
        super().__init__()
        # 通道拆分（分为两部分，一部分直接传递，一部分经过卷积）
        self.split_conv1 = ConvBlock(in_ch, out_ch // 2, kernel=1, padding=0)
        self.split_conv2 = ConvBlock(in_ch, out_ch // 2, kernel=1, padding=0)
        # 多轮卷积（提取特征）
        self.conv_series = nn.Sequential(*[ConvBlock(out_ch // 2, out_ch // 2) for _ in range(num_conv)])
        # 通道拼接后压缩
        self.concat_conv = ConvBlock(out_ch, out_ch, kernel=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.split_conv1(x)  # 直接传递分支
        x2 = self.split_conv2(x)  # 卷积特征分支
        x2 = self.conv_series(x2)
        x = torch.cat([x1, x2], dim=1)  # 通道拼接
        return self.concat_conv(x)


# Backbone：CSPDarknet53（提取多尺度特征）
class CSPDarknet(nn.Module):
    def __init__(self, in_ch: int = 3, out_chs: List[int] = [64, 128, 256, 512, 1024]):
        super().__init__()
        self.stem = ConvBlock(in_ch, out_chs[0], kernel=6, stride=2, padding=2)  # 初始下采样（2x）
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 后续下采样（2x/次）

        # 5个CSP模块（对应5次下采样，最终输出3个尺度特征）
        self.csp1 = CSPBlock(out_chs[0], out_chs[1], num_conv=1)
        self.csp2 = CSPBlock(out_chs[1], out_chs[2], num_conv=2)
        self.csp3 = CSPBlock(out_chs[2], out_chs[3], num_conv=8)
        self.csp4 = CSPBlock(out_chs[3], out_chs[4], num_conv=8)
        self.csp5 = CSPBlock(out_chs[4], out_chs[4], num_conv=4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输出3个尺度特征：x0(32x下采样)、x1(16x下采样)、x2(8x下采样)"""
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.csp1(x)
        x = self.maxpool(x)
        x2 = self.csp2(x)  # 8x尺度（小目标检测）
        x = self.maxpool(x2)
        x1 = self.csp3(x)  # 16x尺度（中目标检测）
        x = self.maxpool(x1)
        x0 = self.csp4(x)  # 32x尺度（大目标检测）
        x0 = self.csp5(x0)
        return x0, x1, x2


# 特征融合网络：PANet（路径聚合网络，增强多尺度特征传递）
class PANet(nn.Module):
    def __init__(self, in_chs: List[int] = [1024, 512, 256], out_ch: int = 256):
        super().__init__()
        # 上采样路径（32x→16x→8x，融合高层与低层特征）
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_up1 = ConvBlock(in_chs[0], out_ch, kernel=1, padding=0)
        self.csp_up1 = CSPBlock(out_ch + in_chs[1], out_ch, num_conv=1)
        self.conv_up2 = ConvBlock(out_ch, out_ch // 2, kernel=1, padding=0)
        self.csp_up2 = CSPBlock(out_ch // 2 + in_chs[2], out_ch // 2, num_conv=1)

        # 下采样路径（8x→16x→32x，增强特征传播）
        self.conv_down1 = ConvBlock(out_ch // 2, out_ch, kernel=3, stride=2, padding=1)
        self.csp_down1 = CSPBlock(out_ch + out_ch, out_ch, num_conv=1)
        self.conv_down2 = ConvBlock(out_ch, in_chs[0], kernel=3, stride=2, padding=1)
        self.csp_down2 = CSPBlock(in_chs[0] + in_chs[0], in_chs[0], num_conv=1)

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入Backbone的3个尺度特征，输出融合后的3个尺度特征"""
        # 上采样1：32x→16x
        x0_up = self.conv_up1(x0)
        x0_up = self.up_sample(x0_up)
        x = torch.cat([x0_up, x1], dim=1)
        x1_fuse = self.csp_up1(x)

        # 上采样2：16x→8x
        x1_up = self.conv_up2(x1_fuse)
        x1_up = self.up_sample(x1_up)
        x = torch.cat([x1_up, x2], dim=1)
        x2_fuse = self.csp_up2(x)

        # 下采样1：8x→16x
        x2_down = self.conv_down1(x2_fuse)
        x = torch.cat([x2_down, x1_fuse], dim=1)
        x1_fuse = self.csp_down1(x)

        # 下采样2：16x→32x
        x1_down = self.conv_down2(x1_fuse)
        x = torch.cat([x1_down, x0], dim=1)
        x0_fuse = self.csp_down2(x)

        return x0_fuse, x1_fuse, x2_fuse


# 检测头：输出类别概率、bounding box坐标、目标置信度
class DetectionHead(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 80, num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 输出通道数：num_anchors × (num_classes + 4坐标 + 1置信度)
        self.out_ch = num_anchors * (num_classes + 5)
        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch * 2, kernel=3, padding=1),
            nn.Conv2d(in_ch * 2, self.out_ch, kernel=1, padding=0)  # 无BN/激活，直接输出原始预测
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出形状：[batch_size, num_anchors, H, W, num_classes+5]"""
        batch_size = x.shape[0]
        x = self.conv(x)
        # 调整维度顺序（适配后续解码）
        x = x.view(batch_size, self.num_anchors, self.num_classes + 5, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 3, 4, 2)  # [B, anchors, H, W, C]
        return x


# YOLOv5整体模型
class YOLOv5(nn.Module):
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.backbone = CSPDarknet(out_chs=[64, 128, 256, 512, 1024])
        self.neck = PANet(in_chs=[1024, 512, 256])
        # 3个检测头（对应3个尺度）
        self.head0 = DetectionHead(in_ch=1024, num_classes=num_classes)  # 32x尺度
        self.head1 = DetectionHead(in_ch=512, num_classes=num_classes)   # 16x尺度
        self.head2 = DetectionHead(in_ch=256, num_classes=num_classes)   # 8x尺度

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输出3个尺度的原始预测结果"""
        # Backbone提取特征
        x0, x1, x2 = self.backbone(x)
        # PANet融合特征
        x0_fuse, x1_fuse, x2_fuse = self.neck(x0, x1, x2)
        # 检测头输出
        out0 = self.head0(x0_fuse)
        out1 = self.head1(x1_fuse)
        out2 = self.head2(x2_fuse)
        return out0, out1, out2


def init_yolov5_model(num_classes: int = 80, pretrained: bool = True, pretrained_weight_path: str = "yolov5s.pt") -> YOLOv5:
    """初始化YOLOv5模型，支持加载预训练权重"""
    model = YOLOv5(num_classes=num_classes)
    if pretrained:
        try:
            # 加载官方预训练权重（需先下载：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt）
            pretrained_weights = torch.load(pretrained_weight_path, map_location="cpu")["model"]
            # 过滤不匹配的权重（如类别数不同时的检测头权重）
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_weights.state_dict().items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"Successfully loaded pretrained weights from {pretrained_weight_path}")
        except Exception as e:
            raise ValueError(f"Failed to load pretrained weights: {str(e)}")
    return model