# 导入 PyTorch 神经网络模块，用于搭建卷积、池化和激活层。
from torch import nn


# 定义 LeNet-5 卷积部分，独立封装成 BackBone 类。
class BackBone(nn.Module):
    """
    经典 LeNet-5 卷积骨干：C1 -> S2 -> C3 -> S4 -> C5。

    输入输出形状：
        (N,1,32,32) -> C1:6@28x28 -> S2:6@14x14
        -> C3:16@10x10 -> S4:16@5x5 -> C5:120@1x1 -> (N,120)。

    类属性：
        features     nn.Sequential 容器，按顺序保存卷积、Tanh、池化和展平层。
        output_size  输出特征维度，固定为 120，供 Head 创建全连接层。
    """

    # 初始化经典 LeNet-5 卷积骨干。
    def __init__(self, in_channels=1):
        """
        搭建 C1、S2、C3、S4 和 C5。

        形参：
            in_channels  输入图片通道数，灰度图片默认是 1。
        """
        # 调用父类构造函数，注册网络参数和子模块。
        super().__init__()
        # 保存输出特征维度，Head 的输入层需要该值。
        self.output_size = 120
        # 严格按照经典 LeNet-5 使用 5*5 卷积、平均池化和 Tanh 激活。
        self.features = nn.Sequential(
            # C1：1 通道映射为 6 张 28*28 特征图。
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),
            # 经典 LeNet-5 使用 Tanh 类激活函数。
            nn.Tanh(),
            # S2：2*2 平均池化，把 28*28 缩小为 14*14。
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C3：6 通道映射为 16 张 10*10 特征图。
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # C3 后使用 Tanh 激活。
            nn.Tanh(),
            # S4：2*2 平均池化，把 10*10 缩小为 5*5。
            nn.AvgPool2d(kernel_size=2, stride=2),
            # C5：用 5*5 卷积把 16@5x5 转换为 120@1x1。
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            # C5 后使用 Tanh 激活。
            nn.Tanh(),
            # 把 (N,120,1,1) 展平成 (N,120)，交给 Head 分类。
            nn.Flatten(start_dim=1),
        )

    # 定义 BackBone 前向传播。
    def forward(self, images):
        """
        从输入图片中提取 120 维特征。

        形参：
            images  形状 (batch_size, 1, 32, 32) 的灰度图片张量。
        返回：
            形状 (batch_size, 120) 的特征张量。
        """
        # 顺序通过 C1、S2、C3、S4、C5 和展平层。
        return self.features(images)
