# 导入 CSV 工具，用于逐行保存每个 epoch 的原始指标。
import csv
# 导入 JSON 工具，用于额外保存结构化训练历史和实验摘要。
import json
# 导入路径工具，用于组织 run 目录下的权重、日志和图片。
from pathlib import Path

# 导入 PyTorch，用于张量计算、自动求导和权重保存。
import torch

# 导入自定义交叉熵损失函数。
from loss import CustomCrossEntropyLoss
# 导入优化器创建函数。
from optimizer import build_optimizer
# 导入曲线图、混淆矩阵和 TP/FP/TN/FN 表格绘制函数。
from picturedraw import plot_training_curves, plot_confusion_matrix, plot_tp_fp_tn_fn_table


# 定义训练器类，负责训练、测试、指标记录和权重保存。
class Trainer:
    """
    训练器：封装训练一个 epoch、测试集评估和完整实验流程。

    类属性：
        model         待训练的神经网络。
        train_loader  训练数据加载器。
        test_loader   测试数据加载器。
        device        运行设备，例如 cpu 或 cuda。
        criterion     自定义交叉熵损失对象。
        optimizer     Adam 优化器。
        run_dir       当前实验结果根目录。
        writer        TensorBoard SummaryWriter；依赖缺失时为 None。
    """

    # 初始化模型、优化器、损失函数、设备和实验目录。
    def __init__(self, model, train_loader, test_loader, device,
                 learning_rate=1e-3, run_dir=None):
        """
        保存训练组件并准备实验输出目录。

        形参：
            model           待训练模型，可以是 HandwrittenClassifier 或 LeNet5。
            train_loader    训练集 DataLoader，按 batch 提供图片与标签。
            test_loader     测试集 DataLoader，用于每个 epoch 后评估。
            device          torch.device 对象，指定 CPU 或 GPU。
            learning_rate   Adam 初始学习率，默认 1e-3。
            run_dir         当前实验根目录；传 None 时只训练而不保存文件。
        """
        # 保存模型并移动到指定设备。
        self.model = model.to(device)
        # 保存训练和测试数据加载器。
        self.train_loader = train_loader
        self.test_loader = test_loader
        # 保存运行设备。
        self.device = device
        # 创建自定义交叉熵损失对象。
        self.criterion = CustomCrossEntropyLoss()
        # 创建 Adam 优化器，并使用 optimizer 模块中的统一配置。
        self.optimizer = build_optimizer(self.model.parameters(), lr=learning_rate)
        # 保存实验目录；没有传入目录时保持 None。
        self.run_dir = Path(run_dir) if run_dir is not None else None
        # TensorBoard 写入器默认不存在。
        self.writer = None
        # 如果指定了实验目录，则创建所有固定子目录。
        if self.run_dir is not None:
            # 创建图片根目录以及 train、test 两个图片子目录。
            (self.run_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
            (self.run_dir / "images" / "test").mkdir(parents=True, exist_ok=True)
            # 创建权重保存目录。
            (self.run_dir / "weights").mkdir(parents=True, exist_ok=True)
            # 尝试启用 TensorBoard；没有安装依赖时不阻塞训练。
            try:
                # 延迟导入可选依赖，避免程序启动时直接报错。
                from torch.utils.tensorboard import SummaryWriter
                # 将 TensorBoard 事件文件保存到当前实验目录。
                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except ModuleNotFoundError:
                # 明确提示缺少依赖，同时保留 CSV、JSON 和图片记录。
                print("提示：当前环境未安装 tensorboard；训练继续，CSV/JSON/图片仍会保存。")

    # 训练一个 epoch，并统计训练损失与训练准确率。
    def train_one_epoch(self):
        """
        遍历一遍训练集，每个 batch 执行一次前向、一次反向和一次权重更新。

        返回：
            (average_loss, accuracy) 二元组，分别是训练集平均损失与总体准确率。
        """
        # 切换到训练模式，启用 Dropout 等训练行为。
        self.model.train()
        # 初始化累计损失、正确样本数和总样本数。
        total_loss = 0.0
        correct = 0
        total = 0
        # 遍历训练数据的所有 batch。
        for images, targets in self.train_loader:
            # 把图片和标签移动到相同运行设备。
            images = images.to(self.device)
            targets = targets.to(self.device)
            # 清空上一个 batch 的梯度。
            self.optimizer.zero_grad()
            # 调用模型 forward，执行一次前向传播。
            logits = self.model(images)
            # 使用自定义交叉熵计算当前 batch 损失。
            loss = self.criterion(logits, targets)
            # 根据前向计算图执行一次反向传播。
            loss.backward()
            # Adam 根据梯度更新一次模型参数。
            self.optimizer.step()
            # 按 batch 样本数累计损失。
            total_loss += loss.item() * images.size(0)
            # 取最大 logit 所在类别作为预测类别。
            predictions = logits.argmax(dim=1)
            # 累加预测正确的样本数。
            correct += (predictions == targets).sum().item()
            # 累加训练样本总数。
            total += targets.size(0)
        # 防止空训练集导致除零。
        if total == 0:
            return 0.0, 0.0
        # 返回整个训练集的平均损失和准确率。
        return total_loss / total, correct / total

    # 在测试集上计算损失、准确率、每类准确率和混淆矩阵。
    @torch.no_grad()
    def evaluate(self):
        """
        在测试集上评估模型，不计算梯度、不反向传播、不更新权重。

        返回：
            (test_loss, accuracy, class_accuracy, confusion_matrix) 四元组：
            test_loss        测试集平均损失。
            accuracy         测试集总体准确率。
            class_accuracy   {类别索引: 识别成功率} 字典。
            confusion_matrix 二维列表，行是真实类别，列是预测类别。
        """
        # 切换到评估模式，关闭 Dropout 等训练行为。
        self.model.eval()
        # 从模型公共属性中读取类别数量，兼容全连接模型与 LeNet-5。
        num_classes = self.model.num_classes
        # 初始化累计测试损失、正确样本数和总样本数。
        total_loss = 0.0
        correct = 0
        total = 0
        # 初始化每类正确数和每类总数。
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        # 初始化混淆矩阵，行是真实类别，列是预测类别。
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        # 遍历全部测试 batch。
        for images, targets in self.test_loader:
            # 把图片和标签移动到运行设备。
            images = images.to(self.device)
            targets = targets.to(self.device)
            # 执行测试阶段前向传播。
            logits = self.model(images)
            # 计算当前 batch 的测试损失。
            loss = self.criterion(logits, targets)
            # 取最大 logit 所在类别作为预测结果。
            predictions = logits.argmax(dim=1)
            # 按 batch 样本数累计损失。
            total_loss += loss.item() * images.size(0)
            # 累加预测正确的数量和样本总数。
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            # 转为 CPU Python 列表，便于逐样本统计混淆矩阵。
            targets_cpu = targets.cpu().tolist()
            predictions_cpu = predictions.cpu().tolist()
            # 逐个样本更新每类统计和混淆矩阵。
            for true_label, predicted_label in zip(targets_cpu, predictions_cpu):
                confusion_matrix[true_label][predicted_label] += 1
                class_total[true_label] += 1
                if true_label == predicted_label:
                    class_correct[true_label] += 1
        # 防止空测试集发生除零。
        if total == 0:
            return 0.0, 0.0, {}, confusion_matrix
        # 计算每个类别的识别成功率，没有样本的类别记为 0。
        class_accuracy = {
            index: (class_correct[index] / class_total[index] if class_total[index] else 0.0)
            for index in range(num_classes)
        }
        # 返回测试集平均损失、总体准确率、每类准确率和混淆矩阵。
        return total_loss / total, correct / total, class_accuracy, confusion_matrix

    # 训练指定轮数，并保存指标、权重、图表和日志。
    def fit(self, epochs, image_dir=None, class_names=None, experiment_name="lenet5"):
        """
        执行完整训练流程。

        形参：
            epochs           训练总轮数。
            image_dir        图片根目录，内部再分 train 和 test；None 时使用 run_dir/images。
            class_names      按索引排列的类别名称列表，用于混淆矩阵坐标。
            experiment_name  实验名称，用于图片文件名与标题。
        返回：
            history 字典，包含每个 epoch 的五项指标列表。
        """
        # 确定图片保存目录，并创建 train 和 test 两个子目录。
        if image_dir is None and self.run_dir is not None:
            image_dir = self.run_dir / "images"
        image_dir = Path(image_dir) if image_dir is not None else None
        if image_dir is not None:
            (image_dir / "train").mkdir(parents=True, exist_ok=True)
            (image_dir / "test").mkdir(parents=True, exist_ok=True)
        # 初始化每轮指标历史。
        history = {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "test_accuracy": [],
            "learning_rate": [],
        }
        # 最佳模型按测试集 accuracy 保存。
        best_accuracy = -1.0
        best_epoch = 0
        # 逐轮训练并评估。
        for epoch in range(1, epochs + 1):
            # 读取本轮实际使用的学习率。
            learning_rate = self.optimizer.param_groups[0]["lr"]
            # 训练一轮，得到训练集 loss 和 accuracy。
            train_loss, train_accuracy = self.train_one_epoch()
            # 测试一轮，得到测试集 loss、accuracy、每类准确率和混淆矩阵。
            test_loss, test_accuracy, class_accuracy, confusion_matrix = self.evaluate()
            # 保存本轮所有关键指标。
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_accuracy"].append(train_accuracy)
            history["test_accuracy"].append(test_accuracy)
            history["learning_rate"].append(learning_rate)
            # 每轮立即追加 CSV，训练中途停止时也保留已完成轮次。
            self._append_csv(epoch, train_loss, test_loss, train_accuracy, test_accuracy, learning_rate)
            # 每轮把相同指标写入 TensorBoard。
            self._write_tensorboard(epoch, train_loss, test_loss, train_accuracy, test_accuracy, learning_rate)
            # 当前测试准确率更高时覆盖保存 best.pt。
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = epoch
                self._save_checkpoint("best.pt", epoch, test_accuracy, history)
            # 每轮覆盖 last.pt，使它始终代表最近完成的 epoch。
            self._save_checkpoint("last.pt", epoch, test_accuracy, history)
            # 每轮输出一次关键训练状态。
            print(
                f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | "
                f"test_loss={test_loss:.4f} | train_acc={train_accuracy:.2%} | "
                f"test_acc={test_accuracy:.2%} | lr={learning_rate:.6g}"
            )
            # 每十轮或最后一轮打印每个类别的测试准确率和 TP/FP/TN/FN 表格。
            if epoch % 10 == 0 or epoch == epochs:
                # HandwrittenDataset 提供 samples；通用 Dataset 没有时传空列表，准确率仍正常打印。
                test_samples = getattr(self.test_loader.dataset, "samples", [])
                self._print_class_accuracy(class_accuracy, test_samples)
                # 用本轮混淆矩阵输出每个类别的 TP、FP、TN、FN，便于对照混淆矩阵。
                self._print_tp_fp_tn_fn(confusion_matrix, test_samples, class_names)
        # 训练结束后刷新并关闭 TensorBoard 日志。
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        # 保存完整 JSON 历史和实验摘要。
        self._save_json(history, best_epoch, best_accuracy, experiment_name)
        # 在最终模型上再评估一次，生成最终测试集混淆矩阵和 TP/FP/TN/FN 统计。
        test_loss, test_accuracy, class_accuracy, final_confusion_matrix = self.evaluate()
        # 控制台输出最终模型的 TP/FP/TN/FN 表格，方便直接查看评估结果。
        print(f"最终模型测试结果：test_loss={test_loss:.4f} | test_acc={test_accuracy:.2%}")
        self._print_class_accuracy(class_accuracy, getattr(self.test_loader.dataset, "samples", []))
        self._print_tp_fp_tn_fn(final_confusion_matrix, getattr(self.test_loader.dataset, "samples", []), class_names)
        # 保存全部曲线和混淆矩阵。
        if image_dir is not None:
            # 绘图函数分别接收训练图片目录和测试图片目录。
            plot_training_curves(
                history,
                image_dir / "train",
                image_dir / "test",
                best_epoch,
                experiment_name,
            )
            # 把类别名称列表转换为“类别索引 -> 类别名称”字典，供混淆矩阵生成刻度。
            class_name_mapping = (
                {index: name for index, name in enumerate(class_names)}
                if class_names is not None else None
            )
            # 绘制最终模型在测试集上的混淆矩阵，并在标题中标明最佳轮次和准确率。
            plot_confusion_matrix(
                final_confusion_matrix,
                image_dir / "test" / f"{experiment_name}_confusion_matrix.png",
                class_names=class_name_mapping,
                best_epoch=best_epoch,
                best_accuracy=best_accuracy,
                experiment_name=experiment_name,
            )
            # 用同一个最终混淆矩阵绘制每个类别的 TP、FP、TN、FN 表格。
            plot_tp_fp_tn_fn_table(
                final_confusion_matrix,
                image_dir / "test" / f"{experiment_name}_tp_fp_tn_fn.png",
                class_names=class_name_mapping,
                experiment_name=experiment_name,
            )
        # 返回完整训练历史。
        return history

    # 保存模型、优化器和关键训练信息。
    def _save_checkpoint(self, filename, epoch, accuracy, history):
        """
        保存可继续训练和复核的 PyTorch 检查点。

        形参：
            filename  文件名，例如 best.pt 或 last.pt。
            epoch     保存时完成的 epoch。
            accuracy  保存时对应的测试集 accuracy。
            history   截止当前 epoch 的训练历史。
        """
        # 未配置运行目录时不保存文件。
        if self.run_dir is None:
            return
        # 构造检查点内容。
        checkpoint = {
            "epoch": epoch,
            "test_accuracy": accuracy,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": history,
        }
        # 把检查点保存到 weights 子目录。
        torch.save(checkpoint, self.run_dir / "weights" / filename)

    # 从 pt 文件恢复模型和优化器，可用于继续训练或单独评估。
    @staticmethod
    def load_checkpoint(model, checkpoint_path, optimizer=None, device=None):
        """
        读取 best.pt 或 last.pt 并恢复权重。

        形参：
            model            已建好的模型实例，结构必须和保存时完全一致，例如 LeNet5(num_classes=10)。
            checkpoint_path  pt 文件路径，例如 runs/时间戳/weights/best.pt。
            optimizer        优化器实例；传 None 则只恢复模型，用于纯推理评估。
            device           目标设备；传 None 时按 CUDA 是否可用自动选择。
        返回：
            checkpoint 字典，包含 epoch、test_accuracy、history 等保存时的信息。
        """
        # 没有指定设备时按 CUDA 是否可用自动选择。
        if device is None:
            # 有 GPU 用 GPU，否则用 CPU。
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 用 map_location 读取，保证 GPU 保存的权重在只有 CPU 的机器上也能加载。
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        # 把权重载入模型，strict 默认 True，要求每层名字和形状都对上。
        model.load_state_dict(checkpoint["model_state_dict"])
        # 把模型搬到目标设备，保证后续推理或训练在同一设备上。
        model.to(device)
        # 如果传入了优化器，则同时恢复优化器状态，继续训练时学习状态不从头开始。
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            # 恢复 Adam 的动量、学习率等内部状态。
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # 返回完整 checkpoint，方便读取 epoch、test_accuracy 和 history。
        return checkpoint

    # 每轮追加一行 CSV 原始指标。
    def _append_csv(self, epoch, train_loss, test_loss, train_accuracy, test_accuracy, learning_rate):
        """
        追加保存一个 epoch 的指标。

        形参：
            epoch           当前训练轮次。
            train_loss      当前轮训练集平均损失。
            test_loss       当前轮测试集平均损失。
            train_accuracy  当前轮训练集总体准确率。
            test_accuracy   当前轮测试集总体准确率。
            learning_rate   当前轮优化器学习率。
        """
        # 未配置运行目录时不保存 CSV。
        if self.run_dir is None:
            return
        # 计算 CSV 路径，并判断是否需要写表头。
        csv_path = self.run_dir / "metrics.csv"
        write_header = not csv_path.exists()
        # utf-8-sig 方便 Windows Excel 正确识别中文。
        with csv_path.open("a", newline="", encoding="utf-8-sig") as file:
            fieldnames = ["epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy", "learning_rate"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "learning_rate": learning_rate,
            })

    # 把当前轮指标写入 TensorBoard。
    def _write_tensorboard(self, epoch, train_loss, test_loss,
                           train_accuracy, test_accuracy, learning_rate):
        """
        写入 TensorBoard 标量日志。

        形参：
            epoch           当前训练轮次，作为横坐标。
            train_loss      训练集平均损失。
            test_loss       测试集平均损失。
            train_accuracy  训练集准确率。
            test_accuracy   测试集准确率。
            learning_rate   当前学习率。
        """
        # 环境缺少 tensorboard 时 writer 为 None，直接跳过。
        if self.writer is None:
            return
        # 分组写入训练和测试损失。
        self.writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        # 分组写入训练和测试准确率。
        self.writer.add_scalars("Accuracy", {"train": train_accuracy, "test": test_accuracy}, epoch)
        # 单独写入学习率。
        self.writer.add_scalar("LearningRate", learning_rate, epoch)

    # 保存完整指标历史和最佳权重摘要。
    def _save_json(self, history, best_epoch, best_accuracy, experiment_name):
        """
        保存 JSON 训练历史，便于后续不依赖图片重新绘图。

        形参：
            history          完整指标历史字典。
            best_epoch       best.pt 对应的 epoch。
            best_accuracy    best.pt 对应的测试准确率。
            experiment_name  当前实验名称。
        """
        # 未配置运行目录时不保存 JSON。
        if self.run_dir is None:
            return
        # 组合历史、最佳指标和权重路径。
        payload = {
            "experiment_name": experiment_name,
            "best_metric": "test_accuracy",
            "best_epoch": best_epoch,
            "best_accuracy": best_accuracy,
            "best_path": str(self.run_dir / "weights" / "best.pt"),
            "last_path": str(self.run_dir / "weights" / "last.pt"),
            "history": history,
        }
        # 使用 UTF-8 保存可读 JSON。
        with (self.run_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    # 打印每个类别的 TP、FP、TN、FN、P、R 和识别成功率。
    @staticmethod
    def _print_tp_fp_tn_fn(confusion_matrix, samples, class_names=None):
        """
        在控制台输出每个类别的 TP、FP、TN、FN、P、R 表格。

        其中 P 是精确率，公式为 TP / (TP + FP)，分母为零时记为 0；
        R 是召回率，公式为 TP / (TP + FN)，分母为零时记为 0。

        形参：
            confusion_matrix  二维列表，行是真实类别，列是预测类别。
            samples           测试样本列表 [(图片路径, 标签), ...]，用于统计样本数。
            class_names       按索引排列的类别名称列表；None 时显示整数索引。
        """
        # 把混淆矩阵转换为整数列表，方便按行列求和。
        matrix = [[int(value) for value in row] for row in confusion_matrix]
        # 空矩阵直接返回。
        if not matrix or not matrix[0]:
            return
        # 获取类别数量。
        num_classes = len(matrix)
        # 计算全部测试样本数量。
        total = sum(sum(row) for row in matrix)
        # 统计每个类别的测试样本数。
        class_counts = {}
        for _, label in samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        # 初始化 P 和 R 的累计值，用于文末计算 macro 平均。
        precision_sum = 0.0
        recall_sum = 0.0
        # 输出表格头，P 是精确率，R 是召回率。
        print("类别 | TP | FP | TN | FN | P | R | 测试样本数")
        # 逐个类别按 One-vs-Rest 统计并输出。
        for class_index in range(num_classes):
            # TP 是对角线，真实该类且预测该类。
            true_positive = matrix[class_index][class_index]
            # FN 是该行总数减去 TP，真实该类但预测错误。
            false_negative = sum(matrix[class_index]) - true_positive
            # FP 是该列总数减去 TP，其他类被错分成该类。
            false_positive = sum(row[class_index] for row in matrix) - true_positive
            # TN 是总数减去 TP、FP、FN。
            true_negative = total - true_positive - false_positive - false_negative
            # P 表示预测成该类的样本中有多少真的属于该类。
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
            # R 表示真实该类的样本中有多少被成功找出来。
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
            # 累加当前类别的 P 和 R，用于后续 macro 平均。
            precision_sum += precision
            recall_sum += recall
            # 生成类别显示名称。
            name = class_names[class_index] if class_names is not None else str(class_index)
            # 输出当前类别的一行统计，P 和 R 保留四位小数。
            print(
                f"{name} | TP={true_positive} | FP={false_positive} | "
                f"TN={true_negative} | FN={false_negative} | "
                f"P={precision:.4f} | R={recall:.4f} | "
                f"测试样本数={class_counts.get(class_index, 0)}"
            )
        # 计算 macro 平均 P 和 macro 平均 R，每个类别权重相同。
        macro_precision = precision_sum / num_classes
        macro_recall = recall_sum / num_classes
        # 输出 macro 平均行，便于一眼看到整体水平。
        print(f"macro平均 | P={macro_precision:.4f} | R={macro_recall:.4f}")

    # 打印每个类别的识别成功率和测试样本数量。
    @staticmethod
    def _print_class_accuracy(class_accuracy, samples):
        """
        打印测试集中每个类别的识别成功率。

        形参：
            class_accuracy  {类别索引: 识别成功率} 字典。
            samples         测试样本列表 [(图片路径, 标签), ...]。
        """
        # 统计每个类别的测试样本数。
        class_counts = {}
        for _, label in samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        # 按类别索引顺序打印准确率和样本数。
        for class_index in sorted(class_accuracy):
            print(
                f"类别 {class_index} 识别成功率: {class_accuracy[class_index]:.2%} | "
                f"测试样本数: {class_counts.get(class_index, 0)}"
            )
