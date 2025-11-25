import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils import load_pretrained_weights  # 引用模型工具函数


# 基础组件：Conv + BN + SiLU
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU()  # YOLOv5默认激活函数，性能优于ReLU
        )

    def forward(self, x):
        return self.conv(x)


# CSP模块（Backbone核心，跨阶段部分连接）
class CSPBlock(nn.Module):
    def __init__(self, in_ch, out_ch, num_conv=1):
        super().__init__()
        # 通道拆分（分为两部分，一部分直接传递，一部分经过卷积）
        self.split_conv1 = ConvBlock(in_ch, out_ch // 2, kernel=1, padding=0)
        self.split_conv2 = ConvBlock(in_ch, out_ch // 2, kernel=1, padding=0)
        # 多轮卷积（增强特征提取）
        self.conv_series = nn.Sequential(*[ConvBlock(out_ch // 2, out_ch // 2) for _ in range(num_conv)])
        # 通道拼接后卷积（融合特征）
        self.concat_conv = ConvBlock(out_ch, out_ch, kernel=1, padding=0)

    def forward(self, x):
        x1 = self.split_conv1(x)  # 直接传递分支
        x2 = self.split_conv2(x)  # 卷积分支
        x2 = self.conv_series(x2)
        x = torch.cat([x1, x2], dim=1)  # 通道拼接（dim=1为通道维度）
        return self.concat_conv(x)


# Backbone：CSPDarknet53（提取多尺度特征）
class CSPDarknet(nn.Module):
    def __init__(self, in_ch=3, out_chs=[64, 128, 256, 512, 1024]):
        super().__init__()
        # 初始卷积（下采样2倍）
        self.stem = ConvBlock(in_ch, out_chs[0], kernel=6, stride=2, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样2倍

        # CSP模块（5个，对应5次下采样，最终输出3个尺度特征）
        self.csp1 = CSPBlock(out_chs[0], out_chs[1], num_conv=1)
        self.csp2 = CSPBlock(out_chs[1], out_chs[2], num_conv=2)
        self.csp3 = CSPBlock(out_chs[2], out_chs[3], num_conv=8)
        self.csp4 = CSPBlock(out_chs[3], out_chs[4], num_conv=8)
        self.csp5 = CSPBlock(out_chs[4], out_chs[4], num_conv=4)

    def forward(self, x):
        # 初始下采样（2倍）
        x = self.stem(x)
        x = self.maxpool(x)

        # 中间CSP模块（下采样2倍×2）
        x = self.csp1(x)
        x = self.maxpool(x)
        x2 = self.csp2(x)  # 输出尺度：1/8x（小目标检测）

        x = self.maxpool(x2)
        x1 = self.csp3(x)  # 输出尺度：1/16x（中目标检测）

        x = self.maxpool(x1)
        x0 = self.csp4(x)  # 输出尺度：1/32x（大目标检测）
        x0 = self.csp5(x0)

        return x0, x1, x2  # 返回3个尺度特征（32x, 16x, 8x）


# 特征融合网络：PANet（路径聚合网络）
class PANet(nn.Module):
    def __init__(self, in_chs=[1024, 512, 256], out_ch=256):
        super().__init__()
        # 上采样路径（32x→16x→8x，融合高层与低层特征）
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")  # 上采样2倍
        self.conv_up1 = ConvBlock(in_chs[0], out_ch, kernel=1, padding=0)
        self.csp_up1 = CSPBlock(out_ch + in_chs[1], out_ch, num_conv=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_up2 = ConvBlock(out_ch, out_ch // 2, kernel=1, padding=0)
        self.csp_up2 = CSPBlock(out_ch // 2 + in_chs[2], out_ch // 2, num_conv=1)

        # 下采样路径（8x→16x→32x，增强特征传播）
        self.down1 = ConvBlock(out_ch // 2, out_ch, kernel=3, stride=2, padding=1)  # 下采样2倍
        self.csp_down1 = CSPBlock(out_ch + out_ch, out_ch, num_conv=1)

        self.down2 = ConvBlock(out_ch, in_chs[0], kernel=3, stride=2, padding=1)
        self.csp_down2 = CSPBlock(in_chs[0] + in_chs[0], in_chs[0], num_conv=1)

    def forward(self, x0, x1, x2):
        # 上采样1：32x→16x（融合x0与x1）
        x0_up = self.conv_up1(x0)
        x0_up = self.up1(x0_up)
        x = torch.cat([x0_up, x1], dim=1)
        x1_fuse = self.csp_up1(x)

        # 上采样2：16x→8x（融合x1_fuse与x2）
        x1_up = self.conv_up2(x1_fuse)
        x1_up = self.up2(x1_up)
        x = torch.cat([x1_up, x2], dim=1)
        x2_fuse = self.csp_up2(x)

        # 下采样1：8x→16x（融合x2_fuse与x1_fuse）
        x2_down = self.down1(x2_fuse)
        x = torch.cat([x2_down, x1_fuse], dim=1)
        x1_fuse = self.csp_down1(x)

        # 下采样2：16x→32x（融合x1_fuse与x0）
        x1_down = self.down2(x1_fuse)
        x = torch.cat([x1_down, x0], dim=1)
        x0_fuse = self.csp_down2(x)

        return x0_fuse, x1_fuse, x2_fuse  # 融合后的3个尺度特征


# 检测头：输出类别概率、Bounding Box坐标、目标置信度
class DetectionHead(nn.Module):
    def __init__(self, in_ch, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 输出通道数：num_anchors × (num_classes + 4坐标 + 1置信度)
        self.out_ch = num_anchors * (num_classes + 5)
        # 卷积层（无BN和激活，直接输出原始预测）
        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch * 2),
            nn.Conv2d(in_ch * 2, self.out_ch, kernel_size=1, padding=0)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        # 调整输出形状：[B, out_ch, H, W] → [B, num_anchors, H, W, num_classes+5]
        x = x.view(batch_size, self.num_anchors, self.num_classes + 5, x.shape[2], x.shape[3])
        x = x.permute(0, 1, 3, 4, 2)  # 维度重排：[B, 锚框数, 高, 宽, 特征数]
        return x


# YOLOv5整体模型
class YOLOv5(nn.Module):
    def __init__(self, num_classes=80, pretrained=True):
        super().__init__()
        # 核心组件
        self.backbone = CSPDarknet()
        self.neck = PANet()
        self.head0 = DetectionHead(in_ch=1024, num_classes=num_classes)  # 32x尺度
        self.head1 = DetectionHead(in_ch=512, num_classes=num_classes)   # 16x尺度
        self.head2 = DetectionHead(in_ch=256, num_classes=num_classes)   # 8x尺度

        # 加载预训练权重（加速收敛）
        if pretrained:
            self.load_state_dict(load_pretrained_weights(), strict=False)
            print("✅ 预训练权重（YOLOv5s）加载成功！")

    def forward(self, x):
        # Backbone提取特征
        x0, x1, x2 = self.backbone(x)
        # PANet融合特征
        x0_fuse, x1_fuse, x2_fuse = self.neck(x0, x1, x2)
        # 检测头输出预测结果
        out0 = self.head0(x0_fuse)  # [B, 3, 20, 20, 85]（640/32=20）
        out1 = self.head1(x1_fuse)  # [B, 3, 40, 40, 85]（640/16=40）
        out2 = self.head2(x2_fuse)  # [B, 3, 80, 80, 85]（640/8=80）
        return out0, out1, out2s