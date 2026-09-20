# 导入操作系统路径工具，用于指定 last.pt 和训练数据目录。
from pathlib import Path

# 导入 PyTorch，用于选择设备。
import torch
# 导入数据加载器，用于按批次加载训练集和测试集。
from torch.utils.data import DataLoader

# 导入数据集类，实际实现位于 dataset_opt 包。
from dataset_opt import HandwrittenDataset
# 导入 LeNet-5 和训练器，复用 Trainer 的 fit() 继续训练。
from model import LeNet5, Trainer


# 在 runs 目录下查找最新的 weights/last.pt，找不到返回 None。
def _find_latest_last_pt(runs_dir):
    """
    自动定位最新一次训练的 last.pt，避免手写时间戳路径。

    形参：
        runs_dir  runs 目录路径，里面每个子目录是一次训练的时间戳。
    返回：
        最新 last.pt 的 Path；没有任何 last.pt 时返回 None。
    """
    # runs 目录不存在时直接返回 None。
    if not runs_dir.is_dir():
        return None
    # 收集所有 runs/时间戳/weights/last.pt 文件。
    candidates = sorted(runs_dir.glob("*/weights/last.pt"), key=lambda path: path.stat().st_mtime)
    # 没有任何 last.pt 时返回 None。
    if not candidates:
        return None
    # 返回修改时间最新的 last.pt。
    return candidates[-1]


# 从 last.pt 恢复后继续训练指定轮数。
def main():
    """
    主流程：建模型 -> 读 last.pt（含优化器） -> 继续 fit() -> 按 test_accuracy 更新 best.pt。

    继续训练的模型结构、类别数、图片尺寸必须和首次训练完全一致。
    """
    # 指定要恢复的权重文件，一般用 last.pt 继续训练。
    # 默认自动查找 runs 下最新的 weights/last.pt；找不到时再用下面的硬编码路径。
    checkpoint_path = _find_latest_last_pt(Path(__file__).resolve().parent / "runs")
    if checkpoint_path is None:
        checkpoint_path = Path(r"D:\zyf\PythonProject1\runs\20260920_153440\weights\last.pt")
    # 指定训练集和测试集目录，和首次训练保持一致。
    train_root = Path(r"D:\zyf\data\handwritten\train")
    # 指定测试集目录。
    test_root = Path(r"D:\zyf\data\handwritten\test")
    # 本次继续训练的轮数，例如再训练 10 轮。
    extra_epochs = 10
    # 统一图片尺寸，必须和首次训练一致。
    image_size = 32
    # 训练批大小，和首次训练保持一致即可。
    batch_size = 64
    # 继续训练时的学习率，一般和首次训练保持一致。
    learning_rate = 1e-3
    # 按 CUDA 是否可用选择设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先建训练集，类别映射由训练集目录自动建立。
    train_dataset = HandwrittenDataset(train_root, image_size=image_size)
    # 再建测试集，复用训练集映射，保证标签一致。
    test_dataset = HandwrittenDataset(
        test_root,
        image_size=image_size,
        class_to_index=train_dataset.class_to_index,
    )
    # 创建训练加载器，每个 epoch 打乱样本。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 创建测试加载器，测试阶段不需要打乱。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 先建一个和首次训练结构完全相同的空模型。
    model = LeNet5(num_classes=train_dataset.num_classes, image_size=image_size)
    # 创建训练器，它的 optimizer 会在 load_checkpoint 时被恢复。
    trainer = Trainer(model, train_loader, test_loader, device, learning_rate=learning_rate)
    # 读取 last.pt，同时恢复模型权重和 Adam 优化器状态。
    checkpoint = Trainer.load_checkpoint(trainer.model, checkpoint_path, optimizer=trainer.optimizer, device=device)
    # 输出恢复信息，确认从哪一轮继续。
    print(f"已从 epoch={checkpoint['epoch']} 恢复，当时 test_acc={checkpoint['test_accuracy']:.2%}，继续训练 {extra_epochs} 轮")

    # 继续训练，best.pt 会在测试准确率更高时自动覆盖，last.pt 每轮覆盖。
    trainer.fit(
        extra_epochs,
        class_names=[train_dataset.index_to_class[index] for index in range(train_dataset.num_classes)],
        experiment_name="LeNet5_resume",
    )


# 只有直接运行本文件时才执行继续训练入口。
if __name__ == "__main__":
    # 调用主函数开始恢复并继续训练。
    main()
