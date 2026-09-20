# 导入 PyTorch，用于创建 Adam 优化器。
import torch


def build_optimizer(params, lr=1e-3, weight_decay=1e-4):
    """
    统一创建 Adam 优化器。

    形参：
        params         模型参数可迭代对象，一般传 model.parameters()。
        lr             学习率，默认 1e-3，控制每次参数更新步长。
        weight_decay   L2 正则系数，默认 1e-4，惩罚过大的权重以缓解过拟合。
    返回：
        torch.optim.Adam 优化器实例。
    """
    # 创建 Adam 优化器，Trainer 原来就是这行：torch.optim.Adam(参数, lr, weight_decay)。
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
