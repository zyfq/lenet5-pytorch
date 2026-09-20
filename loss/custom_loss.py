# 导入 PyTorch，用于张量计算和自动求导。
import torch
# 导入神经网络基类，自定义损失必须继承 nn.Module。
from torch import nn


# 定义自定义交叉熵损失函数，满足作业要求中的“自定义损失函数”。
class CustomCrossEntropyLoss(nn.Module):
    """
    自定义多分类交叉熵损失。

    计算过程：logits -> log_softmax -> 按真实类别取对数概率 -> 取负 -> 对 batch 求平均。
    """

    # 初始化损失对象。
    def __init__(self):
        # 调用 PyTorch 损失基类初始化方法。
        super().__init__()

    # 计算多分类交叉熵，nn.Module 约定入口名叫 forward。
    def forward(self, logits, targets):
        """
        计算一个批次的交叉熵损失。

        形参：
            logits  模型原始输出，形状 (batch_size, num_classes)，未经过 softmax。
            targets  真实类别索引，形状 (batch_size,)，long 类型。
        返回：
            标量张量，当前批次所有样本的平均交叉熵损失。
        """
        # 使用 log_softmax 提高数值稳定性，dim=1 表示类别维度。
        log_probabilities = torch.log_softmax(logits, dim=1)
        # 根据每个样本的真实类别取出对应的对数概率。
        selected_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        # 取负号并对 batch 求平均，得到标量损失。
        return -selected_log_probabilities.mean()
