# 导入随机数工具，用于固定训练过程中的随机状态。
import random
# 导入日期时间工具，用于生成唯一的实验运行目录。
from datetime import datetime
# 导入操作系统路径工具，用于拼接数据目录和 runs 输出目录。
from pathlib import Path

# 导入 NumPy，用于固定数据处理相关的随机状态。
import numpy as np
# 导入 PyTorch，用于固定模型参数初始化并选择训练设备。
import torch
# 导入数据加载器，用于按批次加载训练集和测试集。
from torch.utils.data import DataLoader

# 导入原全连接模型、LeNet-5 和训练器，数据集从 dataset_opt 包导入，main 只负责组装和调用。
from dataset_opt import HandwrittenDataset
from model import HandwrittenClassifier, LeNet5, Trainer


# 定义程序入口函数，集中管理数据集、模型、训练和结果保存。
def main():
    """
    主流程：固定随机种子 -> 建数据集 -> 建 DataLoader -> 选模型 -> 建 Trainer -> 训练并保存结果。

    原数据仍在老地方 D:/zyf/data/handwritten，不会被移动或修改。
    每次运行的结果保存到 runs/时间戳/，包括权重、CSV、JSON、TensorBoard 和图片。

    本次默认使用经典 LeNet-5；如需对比原全连接模型，只改 use_lenet5=False。
    """
    # 设置本地手写数据集根目录，下面必须包含 train 和 test 两个目录。
    dataset_root = Path(r"D:\zyf\data\handwritten")
    # 设置训练集目录，兼容 train/类别名/图片 和 train/类别_编号.png 两种结构。
    train_root = dataset_root / "train"
    # 设置测试集目录，兼容 test/类别名/图片 和 test/类别_编号.png 两种结构。
    test_root = dataset_root / "test"
    # 设置统一图片尺寸，经典 LeNet-5 的无填充结构要求输入为 32*32。
    image_size = 32
    # 设置训练批大小，每次前向和反向处理 64 张图片。
    batch_size = 64
    # 设置训练轮数。
    epochs = 50
    # 选择模型：True 使用经典 LeNet-5，False 使用原来的全连接模型。
    use_lenet5 = True
    # 为本次实验命名，名称会写入图片标题和文件名。
    experiment_name = "LeNet5" if use_lenet5 else "FullyConnected"

    # 固定 Python 随机种子，保证 Python 层面的随机操作可复现。
    random.seed(42)
    # 固定 NumPy 随机种子，保证数组随机操作可复现。
    np.random.seed(42)
    # 固定 PyTorch 随机种子，保证参数初始化、DataLoader shuffle 和 Dropout 可复现。
    torch.manual_seed(42)
    # 根据 CUDA 是否可用选择训练设备。
    """
    torch.device 是一个“地址标签”，用来指明张量和模型放在 CPU 还是 GPU 上。
    torch.device("cpu")        表示使用 CPU。
    torch.device("cuda")       表示使用第 0 块 GPU。
    torch.device("cuda:1")     表示使用第 1 块 GPU。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建带时间戳的实验目录，避免多次运行覆盖之前的结果。
    run_dir = Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    # 确保实验根目录存在。
    run_dir.mkdir(parents=True, exist_ok=True)

    # 创建训练数据集，并由训练集建立类别名称到整数标签的映射。
    train_dataset = HandwrittenDataset(train_root, image_size=image_size)
    # 创建测试数据集，复用训练集映射，保证同一类别对应同一标签。
    test_dataset = HandwrittenDataset(
        test_root,
        image_size=image_size,
        class_to_index=train_dataset.class_to_index,
    )
    # 创建训练数据加载器，shuffle=True 表示每个 epoch 打乱训练样本。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 创建测试数据加载器，测试阶段不需要打乱样本。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 根据配置创建原全连接模型或经典 LeNet-5 模型。
    if use_lenet5:
        # LeNet-5 由 BackBone 卷积池化模块和 Head 全连接模块组成。
        model = LeNet5(num_classes=train_dataset.num_classes, image_size=image_size)
    else:
        # 保留原全连接模型，便于和 LeNet-5 做对照实验。
        model = HandwrittenClassifier(train_dataset.num_classes, image_size=image_size)
    # 创建训练器，并把本次实验目录传给它保存所有结果。
    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        device,
        learning_rate=1e-3,
        run_dir=run_dir,
    )

    # 输出本次实验的类别、数据量、设备和结果目录，便于核对配置。
    print(
        f"experiment={experiment_name} | classes={train_dataset.index_to_class} | "
        f"train={len(train_dataset)} | test={len(test_dataset)} | "
        f"device={device} | out={run_dir}"
    )
    # 开始训练；best.pt 按测试集 accuracy 最高保存，last.pt 保存最后一轮。
    trainer.fit(
        epochs,
        image_dir=run_dir / "images",
        class_names=[train_dataset.index_to_class[index] for index in range(train_dataset.num_classes)],
        experiment_name=experiment_name,
    )


# 只有直接运行本文件时才执行训练入口。
if __name__ == "__main__":
    # 调用主函数启动手写字符识别训练。
    main()
