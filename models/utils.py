import torch


def load_pretrained_weights(weight_path="yolov5s.pt"):
    """
    加载YOLOv5s预训练权重（适配自定义模型结构，过滤不匹配权重）
    :param weight_path: 预训练权重路径（需提前下载：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt）
    :return: 适配后的权重字典
    """
    try:
        # 加载官方权重（含模型权重、超参数等）
        pretrained_dict = torch.load(weight_path, map_location="cpu")["model"].state_dict()
        # 过滤自定义模型中不存在的权重（如官方权重中的辅助模块）
        custom_model_keys = [
            "backbone.stem", "backbone.maxpool", "backbone.csp",
            "neck.up", "neck.conv_up", "neck.csp_up", "neck.down", "neck.csp_down",
            "head0.conv", "head1.conv", "head2.conv"
        ]
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            # 保留与自定义模型结构匹配的权重
            if any(key in k for key in custom_model_keys) or "csp" in k or "conv" in k:
                filtered_dict[k] = v
        return filtered_dict
    except FileNotFoundError:
        raise Exception(f"❌ 预训练权重文件未找到，请下载并放置到 {weight_path}，下载链接：https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt")