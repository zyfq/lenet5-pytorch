# 导入 PyTorch，用于关闭预测阶段的梯度计算。
import torch
# 导入 PyTorch 神经网络模块，用于定义组合模型。
from torch import nn

# 导入 LeNet-5 的卷积骨干和全连接分类头。
from .backbone import BackBone
from .head import Head


# 定义完整的 LeNet-5，负责组合 BackBone 和 Head。
class LeNet5(nn.Module):
    """
    经典 LeNet-5 完整模型，由卷积骨干 BackBone 和分类头 Head 组成。

    数据流：
        (N,1,32,32)
        -> BackBone: (N,120)
        -> Head: (N,num_classes)

    类属性：
        backbone  卷积、池化和特征提取模块。
        head      全连接分类模块。
    """

    # 初始化完整 LeNet-5 模型。
    def __init__(self, num_classes, image_size=32):
        """
        组合 BackBone 和 Head。

        形参：
            num_classes  分类类别数，决定 Head 最后输出的 logits 数量。
            image_size   输入图片边长，经典无填充 LeNet-5 要求为 32；保留该参数用于校验。
        """
        # 调用父类构造函数，注册两个子模块。
        super().__init__()
        # 经典 LeNet-5 的无填充卷积结构要求输入为 32*32。
        if image_size != 32:
            raise ValueError("经典 LeNet-5 的无填充结构要求 image_size=32")
        # 保存类别数量，使训练器不依赖 Head 的内部层名称。
        self.num_classes = num_classes
        # 创建卷积、池化和特征提取模块。
        self.backbone = BackBone(in_channels=1)
        # 创建全连接分类模块，输入维度由 BackBone 的 output_size 提供。
        self.head = Head(num_classes=num_classes, input_features=self.backbone.output_size)

    # 定义完整模型的标准前向传播接口。
    def forward(self, images):
        """
        对一个批次的灰度图片执行一次前向传播。

        形参：
            images  形状 (batch_size, 1, 32, 32) 的灰度图片张量，像素通常已归一化到 0~1。
        返回：
            形状 (batch_size, num_classes) 的 logits，未经过 softmax。
        """
        # 先用 BackBone 提取卷积特征。
        features = self.backbone(images)
        # 再用 Head 将特征转换为每个类别的原始得分。
        return self.head(features)

    # 定义预测接口，供推理时直接得到类别和置信度。
    @torch.no_grad()
    def predict(self, images):
        """
        使用当前模型预测类别，不记录梯度。

        形参：
            images  形状 (batch_size, 1, 32, 32) 的灰度图片张量。
        返回：
            (labels, confidence) 二元组：预测类别索引和对应最大概率。
        """
        # 前向计算得到未归一化的类别得分。
        logits = self(images)
        # 将 logits 转为类别概率。
        probabilities = torch.softmax(logits, dim=1)
        # 取每个样本概率最大的类别和值。
        confidence, labels = torch.max(probabilities, dim=1)
        # 返回预测标签和对应置信度。
        return labels, confidence
