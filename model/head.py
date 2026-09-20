# 导入 PyTorch 神经网络模块，用于搭建 LeNet-5 的全连接分类层。
from torch import nn


# 定义 LeNet-5 的全连接分类部分，独立封装成 Head 类。
class Head(nn.Module):
    """
    LeNet-5 分类头：F6 -> OUTPUT。

    输入是 BackBone 的 C5 输出，每个样本包含 120 个特征；
    先映射到 84 个隐藏特征，再映射到 num_classes 个类别得分。

    类属性：
        classifier  nn.Sequential 容器，按顺序包含全连接层和激活函数。
    """

    # 初始化全连接分类头。
    def __init__(self, num_classes, input_features=120):
        """
        搭建 F6 和输出层。

        形参：
            num_classes    分类类别数，决定最后输出层神经元数量。
            input_features BackBone 输出特征数，经典 LeNet-5 的 C5 输出为 120。
        """
        # 调用父类构造函数，注册网络参数和子模块。
        super().__init__()
        # 按任务图搭建 120 -> 84 -> num_classes 的全连接分类部分。
        self.classifier = nn.Sequential(
            # F6 全连接层：120 个骨干特征映射到 84 个特征。
            nn.Linear(input_features, 84),
            # F6 激活：ReLU，引入非线性并保持与 BackBone 一致。
            nn.ReLU(),
            # 输出层：84 个特征映射到 num_classes 个类别 logits。
            nn.Linear(84, num_classes),
        )

    # 定义分类头前向传播。
    def forward(self, features):
        """
        将 BackBone 输出特征转换为类别 logits。

        形参：
            features  形状 (batch_size, 120) 的骨干特征张量。
        返回：
            形状 (batch_size, num_classes) 的 logits，未经过 softmax。
        """
        # 顺序通过 F6 和输出层，返回每个类别的原始得分。
        return self.classifier(features)
