# 导入路径工具，用于自动创建图片输出目录。
from pathlib import Path

# 导入 NumPy，用于处理混淆矩阵和计算平滑曲线。
import numpy as np
# 导入 Matplotlib 的 pyplot 接口，用于绘制训练指标和混淆矩阵。
import matplotlib.pyplot as plt


def _setup_chinese_font():
    """设置中文字体和负号显示；本函数没有形参和返回值。"""
    # Windows 自带黑体，保证中文标题和图例正常显示。
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 解决中文字体下负号显示为方块的问题。
    plt.rcParams["axes.unicode_minus"] = False


def _prepare_save_path(save_path):
    """
    创建图片父目录并返回 Path 对象。

    形参：
        save_path  图片文件路径，可以是字符串或 Path。
    返回：
        已确保父目录存在的 Path 对象。
    """
    # 把字符串路径统一转换为 Path。
    save_path = Path(save_path)
    # 自动创建不存在的父目录。
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # 返回处理完成的路径。
    return save_path


def _moving_average(values, window=5):
    """
    使用简单移动平均平滑曲线。

    形参：
        values  原始数值列表。
        window  平滑窗口大小，默认 5 个 epoch。
    返回：
        与原列表等长的平滑数值列表。
    """
    # 数据量太少时缩小窗口，至少保留一个数。
    window = min(window, len(values))
    # 空列表直接返回空列表。
    if window == 0:
        return []
    # 对每个位置取最近 window 个数的平均值。
    return [float(np.mean(values[max(0, index - window + 1):index + 1])) for index in range(len(values))]


def plot_training_curves(history, train_dir, test_dir, best_epoch, experiment_name):
    """
    分别保存训练集、测试集和四项指标总览图，并额外保存平滑趋势图。

    形参：
        history          指标字典，包含四条 loss/accuracy 曲线。
        train_dir        训练指标图片目录。
        test_dir         测试指标图片目录。
        best_epoch       根据测试 accuracy 保存 best.pt 的 epoch。
        experiment_name  实验名称，用于标题和文件名。
    """
    # 设置中文字体。
    _setup_chinese_font()
    # 创建从 1 开始的 epoch 横坐标。
    epochs = range(1, len(history["train_loss"]) + 1)
    # 统一的最佳权重说明文字。
    best_note = f"best.pt 保存于 epoch {best_epoch}，依据 test accuracy"

    # 绘制训练 loss 与训练 accuracy。
    figure, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(epochs, history["train_loss"], label="train loss", color="#d95f02")
    axes[0].set_title("训练集损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train accuracy", color="#1b9e77")
    axes[1].set_title("训练集准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    figure.suptitle(f"{experiment_name} 训练指标\n{best_note}")
    figure.tight_layout()
    save_path = _prepare_save_path(Path(train_dir) / f"{experiment_name}_train_metrics.png")
    figure.savefig(save_path, dpi=200)
    plt.close(figure)

    # 绘制测试 loss 与测试 accuracy，并用竖线标出最佳 epoch。
    figure, axes = plt.subplots(2, 1, figsize=(9, 7))
    axes[0].plot(epochs, history["test_loss"], label="test loss", color="#7570b3")
    axes[0].set_title("测试集损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["test_accuracy"], label="test accuracy", color="#e7298a")
    axes[1].axvline(best_epoch, color="black", linestyle="--", label=f"best epoch={best_epoch}")
    axes[1].set_title("测试集准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    figure.suptitle(f"{experiment_name} 测试指标\n{best_note}")
    figure.tight_layout()
    save_path = _prepare_save_path(Path(test_dir) / f"{experiment_name}_test_metrics.png")
    figure.savefig(save_path, dpi=200)
    plt.close(figure)

    # 绘制四项指标总览图，两个 loss 在上、两个 accuracy 在下。
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["test_loss"], label="test loss")
    axes[0].set_title("Train/Test Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_accuracy"], label="train accuracy")
    axes[1].plot(epochs, history["test_accuracy"], label="test accuracy")
    axes[1].axvline(best_epoch, color="black", linestyle="--", label=f"best epoch={best_epoch}")
    axes[1].set_title("Train/Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    figure.suptitle(f"{experiment_name} 四项指标总览\n{best_note}")
    figure.tight_layout()
    save_path = _prepare_save_path(Path(train_dir).parent / f"{experiment_name}_all_metrics.png")
    figure.savefig(save_path, dpi=200)
    plt.close(figure)

    # 无论波动大小都保存一张五轮移动平均图，便于直接观察趋势。
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(epochs, _moving_average(history["train_loss"]), label="smoothed train loss")
    axes[0].plot(epochs, _moving_average(history["test_loss"]), label="smoothed test loss")
    axes[0].set_title("平滑后的 Loss 趋势（窗口最多 5 个 epoch）")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[1].plot(epochs, _moving_average(history["train_accuracy"]), label="smoothed train accuracy")
    axes[1].plot(epochs, _moving_average(history["test_accuracy"]), label="smoothed test accuracy")
    axes[1].axvline(best_epoch, color="black", linestyle="--", label=f"best epoch={best_epoch}")
    axes[1].set_title("平滑后的 Accuracy 趋势（窗口最多 5 个 epoch）")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    figure.suptitle(f"{experiment_name} 平滑指标\n{best_note}")
    figure.tight_layout()
    save_path = _prepare_save_path(Path(train_dir).parent / f"{experiment_name}_smoothed_metrics.png")
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_learning_rate(learning_rates, save_path):
    """
    绘制学习率随 epoch 的变化曲线。

    形参：
        learning_rates  每个 epoch 使用的学习率列表。
        save_path       学习率曲线保存路径。
    """
    # 设置中文字体。
    _setup_chinese_font()
    # 生成 epoch 横坐标。
    epochs = range(1, len(learning_rates) + 1)
    # 创建学习率曲线画布。
    figure, axes = plt.subplots(figsize=(9, 5))
    axes.plot(epochs, learning_rates, label="learning rate", color="#66a61e")
    axes.set_title("学习率随 Epoch 的变化")
    axes.set_xlabel("Epoch")
    axes.set_ylabel("Learning Rate")
    axes.legend()
    figure.tight_layout()
    # 保存并关闭图像。
    save_path = _prepare_save_path(save_path)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_confusion_matrix(confusion_matrix, save_path, class_names=None,
                          best_epoch=None, best_accuracy=None, experiment_name="experiment"):
    """
    绘制训练结束后最终模型在测试集上的混淆矩阵。

    形参：
        confusion_matrix  二维列表，行是真实类别，列是预测类别。
        save_path         混淆矩阵图片保存路径。
        class_names       {类别索引: 类别名称} 字典；None 时显示整数索引。
        best_epoch        best.pt 对应的 epoch，用于图标题说明。
        best_accuracy     best.pt 对应的测试 accuracy，用于图标题说明。
        experiment_name   实验名称，用于图标题。
    """
    # 设置中文字体。
    _setup_chinese_font()
    # 把混淆矩阵转换为 NumPy 整数数组。
    matrix = np.array(confusion_matrix, dtype=int)
    # 空矩阵不绘制。
    if matrix.size == 0:
        return
    # 根据映射生成按索引顺序排列的类别名称。
    if class_names is None:
        labels = [str(index) for index in range(matrix.shape[0])]
    else:
        labels = [str(class_names.get(index, index)) for index in range(matrix.shape[0])]
    # 创建混淆矩阵热力图。
    figure, axes = plt.subplots(figsize=(7, 6))
    image = axes.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axes)
    # 设置坐标轴和类别刻度。
    axes.set_xlabel("预测类别")
    axes.set_ylabel("真实类别")
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels)
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels)
    # 在标题中写明 best.pt 的保存依据和对应 epoch。
    accuracy_text = f"{best_accuracy:.2%}" if best_accuracy is not None else "未知"
    axes.set_title(
        f"{experiment_name} 测试集混淆矩阵\n"
        f"best.pt: epoch={best_epoch}, test accuracy={accuracy_text}"
    )
    # 在每个格子中标出样本数量。
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
    figure.tight_layout()
    # 保存并关闭图像。
    save_path = _prepare_save_path(save_path)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_tp_fp_tn_fn_table(confusion_matrix, save_path, class_names=None, experiment_name="experiment"):
    """
    绘制每个类别的 TP、FP、TN、FN、P、R 表格图片，一行一个类别。

    多分类下每个类别都按 One-vs-Rest 统计：
    TP 是该类被正确预测的数量，FN 是该类被错分成其他类的数量，
    FP 是其他类被错分成该类的数量，TN 是剩余全部正确拒绝的数量。
    P 是精确率，公式为 TP / (TP + FP)，分母为零时记为 0；
    R 是召回率，公式为 TP / (TP + FN)，分母为零时记为 0。

    形参：
        confusion_matrix  二维列表，行是真实类别，列是预测类别。
        save_path         表格图片保存路径。
        class_names       {类别索引: 类别名称} 字典；None 时显示整数索引。
        experiment_name   实验名称，用于图标题。
    """
    # 设置中文字体，保证中文标题正常显示。
    _setup_chinese_font()
    # 把混淆矩阵转换为 NumPy 整数数组，方便按行列求和。
    matrix = np.array(confusion_matrix, dtype=int)
    # 空矩阵不绘制。
    if matrix.size == 0:
        return
    # 获取类别数量，混淆矩阵是 num_classes * num_classes 的方阵。
    num_classes = matrix.shape[0]
    # 生成按索引顺序排列的类别名称。
    if class_names is None:
        labels = [str(index) for index in range(num_classes)]
    else:
        labels = [str(class_names.get(index, index)) for index in range(num_classes)]
    # 计算全部测试样本数量，TN 需要用总数减去 TP、FP、FN。
    total = int(matrix.sum())
    # 初始化表格每一行的数据。
    rows = []
    # 初始化 P 和 R 的累计值，用于文末计算 macro 平均。
    precision_sum = 0.0
    recall_sum = 0.0
    # 按 One-vs-Rest 逐个类别统计 TP、FP、FN、TN、P、R。
    for class_index in range(num_classes):
        # TP 是真实该类且预测该类的数量，即混淆矩阵对角线。
        true_positive = int(matrix[class_index, class_index])
        # FN 是真实该类但预测成其他类的数量，即该行总数减去 TP。
        false_negative = int(matrix[class_index, :].sum() - true_positive)
        # FP 是真实其他类但预测成该类的数量，即该列总数减去 TP。
        false_positive = int(matrix[:, class_index].sum() - true_positive)
        # TN 是既不是该类也没有预测成该类的数量。
        true_negative = int(total - true_positive - false_positive - false_negative)
        # P 是精确率，表示预测成该类的样本中有多少真的属于该类。
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        # R 是召回率，表示真实该类的样本中有多少被成功找出来。
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        # 累加当前类别的 P 和 R，用于后续 macro 平均。
        precision_sum += precision
        recall_sum += recall
        # 保存当前类别的一行表格数据，P 和 R 保留四位小数。
        rows.append([labels[class_index], true_positive, false_positive, true_negative, false_negative, f"{precision:.4f}", f"{recall:.4f}"])
    # 计算 macro 平均 P 和 macro 平均 R，每个类别权重相同。
    macro_precision = precision_sum / num_classes
    macro_recall = recall_sum / num_classes
    # 追加 macro 平均行，便于一眼看到整体水平。
    rows.append(["macro平均", "-", "-", "-", "-", f"{macro_precision:.4f}", f"{macro_recall:.4f}"])
    # 动态计算画布高度，保证类别越多表格越高。
    figure, axes = plt.subplots(figsize=(10, 1.6 + 0.45 * (num_classes + 1)))
    # 隐藏坐标轴，只保留表格。
    axes.axis("off")
    # 设置表格标题，写明 P 和 R 的公式。
    axes.set_title(f"{experiment_name} 每个类别的 TP / FP / TN / FN / P / R\nP=TP/(TP+FP)，R=TP/(TP+FN)", pad=16)
    # 创建表格：列名 + 每个类别一行 + macro 平均行。
    table = axes.table(
        cellText=rows,
        colLabels=["类别", "TP", "FP", "TN", "FN", "P", "R"],
        loc="center",
    )
    # 放大表格字体和行高，保证数字清晰可读。
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # 自动调整布局，避免标题和表格重叠。
    figure.tight_layout()
    # 保存并关闭图像。
    save_path = _prepare_save_path(save_path)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)


def plot_class_accuracy(class_accuracy, samples, save_path):
    """
    保留原有类别准确率柱状图函数，兼容旧代码。

    形参：
        class_accuracy  {类别索引: 识别成功率} 字典。
        samples         测试集样本列表，用于统计各类别样本数。
        save_path       柱状图保存路径。
    """
    # 设置中文字体。
    _setup_chinese_font()
    # 统计每个类别样本数。
    class_counts = {}
    for _, label in samples:
        class_counts[label] = class_counts.get(label, 0) + 1
    # 准备排序后的类别和准确率。
    labels = sorted(class_accuracy)
    accuracies = [class_accuracy[label] for label in labels]
    # 创建柱状图。
    figure, axes = plt.subplots(figsize=(8, 5))
    bars = axes.bar([str(label) for label in labels], accuracies, color="#4f8ef7")
    # 在每个柱顶标注准确率和样本数。
    for bar, label, accuracy in zip(bars, labels, accuracies):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                  f"{accuracy:.2%}\n(n={class_counts.get(label, 0)})",
                  ha="center", va="bottom", fontsize=8)
    axes.set_title("每个类别识别成功率")
    axes.set_xlabel("类别")
    axes.set_ylabel("识别成功率")
    axes.set_ylim(0, 1.1)
    figure.tight_layout()
    # 保存并关闭图像。
    save_path = _prepare_save_path(save_path)
    figure.savefig(save_path, dpi=200)
    plt.close(figure)
