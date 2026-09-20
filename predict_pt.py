# 导入操作系统路径工具，用于指定权重文件和图片路径。
from pathlib import Path

# 导入 OpenCV，用于读取单张待预测图片。
import cv2
# 导入 NumPy，用于执行像素类型转换。
import numpy as np
# 导入 PyTorch，用于加载权重和执行推理。
import torch
# 导入数据加载器，用于批量评估测试集。
from torch.utils.data import DataLoader

# 导入数据集类，实际实现位于 dataset_opt 包。
from dataset_opt import HandwrittenDataset
# 导入 LeNet-5 和训练器，复用 Trainer 的 evaluate() 做测试集评估。
from model import LeNet5, Trainer


# 在 runs 目录下查找最新的 weights/best.pt，找不到返回 None。
def _find_latest_best_pt(runs_dir):
    """
    自动定位最新一次训练的 best.pt，避免手写时间戳路径。

    形参：
        runs_dir  runs 目录路径，里面每个子目录是一次训练的时间戳。
    返回：
        最新 best.pt 的 Path；没有任何 best.pt 时返回 None。
    """
    # runs 目录不存在时直接返回 None。
    if not runs_dir.is_dir():
        return None
    # 收集所有 runs/时间戳/weights/best.pt 文件。
    candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda path: path.stat().st_mtime)
    # 没有任何 best.pt 时返回 None。
    if not candidates:
        return None
    # 返回修改时间最新的 best.pt。
    return candidates[-1]


# 用 best.pt 在测试集上重新评估，并预测单张图片。
def main():
    """
    主流程：建空模型 -> 读 best.pt -> 测试集复评 -> 单张图片预测。

    读取的权重结构必须和训练时一致：LeNet-5、10 类、32*32 输入。
    """
    # 指定要读取的权重文件，训练后改成自己 runs 目录下的 best.pt。
    # 默认自动查找 runs 下最新的 weights/best.pt；找不到时再用下面的硬编码路径。
    checkpoint_path = _find_latest_best_pt(Path(__file__).resolve().parent / "runs")
    if checkpoint_path is None:
        checkpoint_path = Path(r"D:\zyf\PythonProject1\runs\20260920_153440\weights\best.pt")
    # 指定测试集目录，用于重新评估该 pt 的真实效果。
    test_root = Path(r"D:\zyf\data\handwritten\test")
    # 指定单张待预测图片，改成自己想测的那张。
    single_image_path = Path(r"D:\zyf\data\handwritten\test\0_0.png")
    # 类别数是 0~9 共 10 类，必须和训练时一致。
    num_classes = 10
    # 图片尺寸必须和训练时一致，LeNet-5 要求 32。
    image_size = 32
    # 评估批量大小，只影响速度不影响结果。
    batch_size = 64
    # 按 CUDA 是否可用选择设备。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 先建一个和训练时结构完全相同的空模型。
    model = LeNet5(num_classes=num_classes, image_size=image_size)
    # 创建一个训练器占位，只借用它的 evaluate()，不会执行训练。
    test_dataset = HandwrittenDataset(test_root, image_size=image_size)
    # 创建测试数据加载器，测试阶段不需要打乱样本。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 训练器需要同时传入训练和测试加载器，这里用测试加载器占位即可。
    evaluator = Trainer(model, test_loader, test_loader, device)
    # 读取 pt 文件并恢复模型权重，不传 optimizer 表示只做推理评估。
    checkpoint = Trainer.load_checkpoint(evaluator.model, checkpoint_path, optimizer=None, device=device)
    # 输出权重保存时的轮次和准确率，确认读的是哪个 best。
    print(f"权重来自 epoch={checkpoint['epoch']}，保存时 test_acc={checkpoint['test_accuracy']:.2%}")

    # 调用已有的评估逻辑，得到 loss、总体准确率、每类准确率和混淆矩阵。
    test_loss, test_acc, class_accuracy, confusion_matrix = evaluator.evaluate()
    # 输出复评的总体结果。
    print(f"复评结果：test_loss={test_loss:.4f} | test_acc={test_acc:.2%}")
    # 输出每类准确率和 TP/FP/TN/FN/P/R 表格，和训练结尾的格式一致。
    evaluator._print_class_accuracy(class_accuracy, getattr(evaluator.test_loader.dataset, "samples", []))
    # 用最终混淆矩阵输出每个类别的统计，class_names 按索引顺序排列。
    evaluator._print_tp_fp_tn_fn(
        confusion_matrix,
        getattr(evaluator.test_loader.dataset, "samples", []),
        [test_dataset.index_to_class[index] for index in range(test_dataset.num_classes)],
    )

    # 单张图片预测：以灰度模式读取，手写字符不需要颜色通道。
    image = cv2.imread(str(single_image_path), cv2.IMREAD_GRAYSCALE)
    # 检查图片是否读取成功。
    if image is None:
        # 读取失败时直接报错，提示检查图片路径。
        raise FileNotFoundError(f"单张图片不存在或无法读取: {single_image_path}")
    # 缩放到训练时的统一尺寸，参数顺序是宽、高。
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    # 转 float32 并归一化到 0~1，再补 batch 和通道维度，形成 (1,1,H,W)。
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    # 切换到评估模式，关闭 Dropout 等训练行为。
    evaluator.model.eval()
    # 不记录梯度执行一次前向推理。
    with torch.no_grad():
        # 计算每个类别的 logits。
        logits = evaluator.model(tensor)
        # 转成概率分布。
        probs = torch.softmax(logits, dim=1)
        # 取概率最大的类别和置信度。
        conf, pred = torch.max(probs, dim=1)
    # 输出单张图片的预测类别和置信度。
    print(f"图片 {single_image_path.name} 预测为 {int(pred.item())}，置信度 {float(conf.item()):.2%}")


# 只有直接运行本文件时才执行推理入口。
if __name__ == "__main__":
    # 调用主函数开始读取 pt 并评估预测。
    main()
