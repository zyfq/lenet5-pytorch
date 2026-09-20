# 导入 PyTorch，用于张量计算。
import torch
# 导入 PyTorch 神经网络模块，用于定义全连接层、激活函数和 Dropout。
from torch import nn


# 定义原来的手写字符识别全连接网络，保留它以便与 LeNet-5 做对比实验。
class HandwrittenClassifier(nn.Module):
    """
    手写字符识别多层感知机（全连接网络）。

    结构：Flatten -> Linear(1024->256) -> ReLU -> Dropout(0.3)
          -> Linear(256->128) -> ReLU -> Dropout(0.2)
          -> Linear(128->num_classes)

    类属性：
        num_classes  分类类别数，供训练器创建统计数组和混淆矩阵。
        network      nn.Sequential 容器，按顺序包含所有网络层。
    """

    # 初始化全连接网络结构。
    def __init__(self, num_classes, image_size=32):
        """
        搭建原来的全连接网络。

        形参：
            num_classes  分类类别数，决定输出层神经元数量。
            image_size   输入图片边长，默认 32，输入特征数为 image_size*image_size。
        """
        # 调用父类构造函数，注册网络参数和子模块。
        super().__init__()
        # 保存类别数量，使训练器不依赖某个具体模型的内部层名称。
        self.num_classes = num_classes
        # 计算展平后的输入特征数量，单通道图片为 1*H*W。
        input_features = image_size * image_size
        # 定义多层感知机，Linear 参数是输入特征数和输出特征数。
        self.network = nn.Sequential(
            # 将 C,H,W 展平为一维向量，Flatten 不改变 batch 维度。
            nn.Flatten(),
            # 第一个全连接层，把所有像素映射为 256 个特征。
            nn.Linear(input_features, 256),
            # ReLU 把负数置零并引入非线性。
            nn.ReLU(),
            # Dropout 随机丢弃 30% 神经元，用于缓解过拟合。
            nn.Dropout(0.3),
            # 第二个全连接层，把 256 个特征映射为 128 个特征。
            nn.Linear(256, 128),
            # 第二个 ReLU 激活函数。
            nn.ReLU(),
            # 第二个 Dropout 层随机丢弃 20% 神经元。
            nn.Dropout(0.2),
            # 输出层把 128 个特征映射到 num_classes 个类别得分。
            nn.Linear(128, num_classes),
        )

    # 定义前向传播，输入形状为 batch_size,1,H,W。
    def forward(self, images):
        """
        使用原全连接网络执行前向传播。

        形参：
            images  形状 (batch_size, 1, H, W) 的图片张量，像素值范围为 0~1。
        返回：
            形状 (batch_size, num_classes) 的 logits，未经过 softmax。
        """
        # 顺序通过所有层，返回每个类别的原始得分。
        return self.network(images)

    # 定义统一预测接口，返回类别索引和最大概率。
    @torch.no_grad()
    def predict(self, images):
        """
        在不记录梯度的情况下预测类别和置信度。

        形参：
            images  形状 (batch_size, 1, H, W) 的图片张量。
        返回：
            (labels, confidence) 二元组：labels 是预测类别，confidence 是对应最大概率。
        """
        # 把 logits 转成类别概率。
        probabilities = torch.softmax(self(images), dim=1)
        # 取每个样本概率最大的类别和值。
        confidence, labels = torch.max(probabilities, dim=1)
        # 返回预测标签和置信度。
        return labels, confidence
