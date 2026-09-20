# 导入操作系统路径工具，用于管理本地手写字符图片目录。
from pathlib import Path
# 导入随机数工具，用于固定训练过程中的随机状态。
import random
# 导入 OpenCV，用于读取、灰度化和缩放图片。
import cv2
# 导入 NumPy，用于执行数组类型转换。
import numpy as np
# 导入 PyTorch，用于张量计算和自动求导。
import torch
# 导入 PyTorch 神经网络模块，用于定义模型和损失基类。
from torch import nn
# 导入数据集基类、批量加载器和随机采样器。
from torch.utils.data import DataLoader, Dataset
# 导入 Matplotlib 的 pyplot 接口，用于绘制训练曲线和分类统计图。
import matplotlib.pyplot as plt



# 定义本地手写字符数据集类，兼容两种目录格式：split/类别名/图片 或 split/类别_编号.png。
class HandwrittenDataset(Dataset):
    # 初始化一个已经独立划分好的训练集或测试集目录。
    def __init__(self, root_dir, image_size=32, class_to_index=None):
        # 保存当前数据划分目录。
        self.root_dir = Path(root_dir)
        # 保存统一缩放后的图片边长。
        self.image_size = image_size
        # 检查当前数据划分目录是否存在。
        if not self.root_dir.is_dir():
            # 目录不存在时抛出异常，避免后续遍历目录失败。
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")
        # 获取当前划分中的类别子目录。
        """
        iterdir() 作用是列出一个目录下有什么——返回该目录里子文件夹的路径，一个一个吐给你（生成器）。
        class_dirs 是一个一维列表，里面保存的是目录路径。
        """
        class_dirs = sorted(path for path in self.root_dir.iterdir() if path.is_dir())
        # 如果存在类别子目录，则按类别子目录模式收集样本。
        if class_dirs:
            # 训练集建立类别映射，测试集复用训练集映射。
            self.class_to_index = class_to_index or {path.name: index for index, path in enumerate(class_dirs)}
            # 检查测试集中的每个类别都存在于训练集。
            missing_classes = [path.name for path in class_dirs if path.name not in self.class_to_index]
            # 测试集出现训练集没有的类别时直接报错。
            if missing_classes:
                # 抛出异常，防止标签映射错乱。
                raise ValueError(f"测试集包含训练集没有的类别: {missing_classes}")
            # 收集类别子目录下所有图片路径和标签。
            self.samples = self._collect_from_class_dirs(class_dirs)
        # 如果没有类别子目录，则按扁平文件名模式收集样本。
        else:
            # 按扁平文件名模式收集样本，同时确定类别映射。
            self.samples = self._collect_from_flat_files(class_to_index)
        # 检查当前划分是否包含图片。
        if not self.samples:
            # 没有图片时抛出异常，提示检查图片后缀和目录。
            raise ValueError(f"目录中没有找到 png、jpg、jpeg 或 bmp 图片: {self.root_dir}")
        # 过滤训练或测试过程中无法读取的损坏图片。
        self.samples, skipped_files = self._filter_unreadable(self.samples, image_size)
        # 如果跳过了损坏图片，则输出被跳过的文件数量和首个文件。
        if skipped_files:
            # 输出损坏图片统计信息，提示用户删除或修复对应文件。
            print(f"跳过 {len(skipped_files)} 张损坏图片，例如: {skipped_files[0]}")
        # 检查过滤后是否还有可用图片。
        if not self.samples:
            # 没有可用图片时抛出异常，提示数据集全部损坏。
            raise ValueError(f"目录中没有可读取的图片: {self.root_dir}")
        # 保存整数标签到类别名称的反向映射。
        self.index_to_class = {index: name for name, index in self.class_to_index.items()}
        # 保存类别数量，供外部创建输出层使用。
        self.num_classes = len(self.class_to_index)

    # 从文件名中解析类别名，要求格式为 类别_编号.png 或 类别-编号.png。
    def _parse_flat_label(self, image_path):
        # 获取不带后缀的文件名主体，例如 0_0.png 得到 0_0。
        """
        image_path.stem   去掉最后一个扩展名后的部分。是属性
        """
        stem = image_path.stem
        # 文件名包含下划线时，以下划线前的部分作为类别名。
        if "_" in stem:
            # 按下划线切分并取第一段作为类别名。
            return stem.split("_")[0].strip()
        # 文件名包含连字符时，以连字符前的部分作为类别名。
        if "-" in stem:
            # 按连字符切分并取第一段作为类别名。
            return stem.split("-")[0].strip()
        # 既没有下划线也没有连字符时无法判断类别，直接报错。
        raise ValueError(f"文件名无法解析类别，请改成 类别_编号.png 格式: {image_path}")

    # 收集类别子目录模式下的图片文件。
    def _collect_from_class_dirs(self, class_dirs):
        # 定义允许读取的图片后缀。
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        # 初始化样本列表。
        samples = []
        # 遍历每一个类别目录。
        for class_dir in class_dirs:
            # 遍历当前类别目录及其子目录中的所有文件。
            for image_path in sorted(class_dir.rglob("*")):
                # 只保留支持的图片文件。
                if image_path.is_file() and image_path.suffix.lower() in image_suffixes:
                    # 将图片路径和类别整数标签加入样本列表。
                    samples.append((image_path, self.class_to_index[class_dir.name]))
        # 返回收集完成的样本。
        return samples

    # 收集扁平文件名模式下的图片文件，文件直接放在 train 或 test 目录下。
    def _collect_from_flat_files(self, class_to_index):
        # 定义允许读取的图片后缀。
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp"}
        # 递归收集当前目录下所有支持的图片文件。
        """
        iterdir() 作用 列出当前目录下的直接子项，包括文件和目录，不递归。
        返回值：返回一个迭代器,顺序不保证,目录不存在或不是目录会报错
        
        glob()  作用 按模式匹配，通常也只看当前层
        返回值：返回一个生成器,每个元素是 pathlib.Path,只返回匹配 pattern 的路径,顺序不保证,可以匹配文件，也可以匹配目录.
        模式：   *	匹配任意字符，但不跨目录
                ?	匹配单个字符
                [abc]	匹配 a、b、c 中一个
                [!abc]	不匹配 a、b、c
                **	递归匹配所有子目录
                
        rglob() 作用 会得到所有层级的路径，包括目录和文件：
        rglob("*") 返回的是生成器，不是 list，而且顺序通常不保证，所以经常配合 sorted()。
        
        is_file() 判断是不是文件夹  没有目录或者不是就返回False  是就返回True  如果这个路径是符号链接，并且指向一个真实文件，通常返回 True；如果是断链，返回 False。
        
        path.suffix：获取路径的最后一个扩展名，包含点号。是属性。
        """
        image_files = sorted(path for path in self.root_dir.rglob("*") if path.is_file() and path.suffix.lower() in image_suffixes)
        # 当前目录下没有图片时抛出异常。
        if not image_files:
            # 抛出异常，提示检查目录和图片后缀。
            raise ValueError(f"目录中没有找到 png、jpg、jpeg 或 bmp 图片: {self.root_dir}")
        # 从所有文件名中解析类别名并去重排序，得到候选类别列表。
        discovered_labels = sorted({self._parse_flat_label(path) for path in image_files})
        # 训练集没有传入映射时，根据发现的类别名建立映射。
        if class_to_index is None:
            # 按排序后的类别名建立名称到整数标签的映射。
            self.class_to_index = {name: index for index, name in enumerate(discovered_labels)}
        # 测试集传入映射时，直接复用训练集的映射。
        else:
            # 保存训练集传入的类别映射。
            self.class_to_index = dict(class_to_index)
            # 检查测试文件中是否存在训练集没有的类别。
            unknown_labels = sorted({self._parse_flat_label(path) for path in image_files} - set(self.class_to_index))
            # 发现未知类别时直接报错，避免标签错乱。
            if unknown_labels:
                # 抛出异常，提示测试集类别超出训练集范围。
                raise ValueError(f"测试集包含训练集没有的类别: {unknown_labels}")
        # 初始化样本列表。
        samples = []
        # 遍历每一张图片文件。
        for image_path in image_files:
            # 解析当前文件的类别名。
            label_name = self._parse_flat_label(image_path)
            # 将图片路径和对应的整数标签加入样本列表。
            samples.append((image_path, self.class_to_index[label_name]))
        # 返回收集完成的样本。
        return samples

    # 过滤无法读取的损坏图片，避免训练中途因为 libpng 错误中断。
    def _filter_unreadable(self, samples, image_size):
        # 初始化可用样本列表。
        valid_samples = []
        # 初始化损坏文件列表。
        skipped_files = []
        # 逐一尝试读取每张图片。
        for image_path, label in samples:
            # 以灰度模式试读当前图片。
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            # 图片为空或者维度异常时视为损坏。
            if image is None or image.size == 0:
                # 记录损坏文件路径。
                skipped_files.append(image_path)
                # 跳过当前损坏文件。
                continue
            # 尝试缩放图片，缩放失败也视为损坏。
            try:
                # 将图片缩放为统一尺寸，参数顺序是宽、高。
                cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            # 捕获缩放过程中的 OpenCV 异常。
            except cv2.error:
                # 记录缩放失败的文件路径。
                skipped_files.append(image_path)
                # 跳过当前损坏文件。
                continue
            # 当前图片可用时加入有效样本。
            valid_samples.append((image_path, label))
        # 返回过滤后的样本和损坏文件列表。
        return valid_samples, skipped_files

    # 收集类别目录下支持的图片文件。
    def _collect_samples(self, class_dirs):
        # 直接委托给类别子目录模式的收集函数。
        return self._collect_from_class_dirs(class_dirs)

    # 读取单张图片并做统一预处理，已在初始化时过滤损坏图片。
    def _load_processed_image(self, image_path):
        # 以灰度模式读取图片，手写字符通常不需要颜色通道。
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        # 检查图片是否读取成功。
        if image is None:
            # 读取失败时抛出异常，提示检查图片文件。
            raise ValueError(f"无法读取图片: {image_path}")
        # 将图片缩放为统一尺寸，参数顺序是宽、高。
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        # 返回缩放完成的图片数组。
        return image

    # 根据索引读取并处理一张图片。
    def __getitem__(self, index):
        # 获取图片路径和整数标签。
        image_path, label = self.samples[index]
        # 读取并统一缩放当前图片。
        image = self._load_processed_image(image_path)
        # 将 uint8 像素转换为 float32，并归一化到 0 到 1。
        image_tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        # 增加通道维度，形成 PyTorch 图像格式 C,H,W。
        image_tensor = image_tensor.unsqueeze(0)
        # 返回图片张量和类别张量，交叉熵标签必须是 long 类型。
        return image_tensor, torch.tensor(label, dtype=torch.long)

    # 返回数据集样本数量。
    def __len__(self):
        # 返回当前训练集或测试集的样本数量。
        return len(self.samples)


# 定义手写字符识别网络，继承 PyTorch 的 nn.Module。
class HandwrittenClassifier(nn.Module):
    # 初始化网络结构。
    def __init__(self, num_classes, image_size=32):
        # 调用父类构造函数，注册网络参数和子模块。
        super().__init__()
        # 计算展平后的输入特征数量，单通道图片为 1*H*W。
        input_features = image_size * image_size
        # 定义多层感知机，Linear 参数是输入特征数和输出特征数。
        self.network = nn.Sequential(
            # 将 C,H,W 展平为一维向量，Flatten 不改变 batch 维度。
            nn.Flatten(),
            # 第一个全连接层，Linear 参数是输入特征数和输出特征数。
            nn.Linear(input_features, 256),
            # 第一个激活函数，ReLU 把负数置零并引入非线性。
            nn.ReLU(),
            # Dropout 随机丢弃 30% 神经元，抑制训练损失趋近于零的过拟合现象。
            nn.Dropout(0.3),
            # 第二个全连接层，Linear 参数是输入特征数和输出特征数。
            nn.Linear(256, 128),
            # 第二个激活函数，ReLU 把负数置零并引入非线性。
            nn.ReLU(),
            # 第二个 Dropout 层，进一步降低全连接层对训练样本的记忆能力。
            nn.Dropout(0.2),
            # 输出全连接层，Linear 参数是输入特征数和类别数。
            nn.Linear(128, num_classes),
        )

    # 定义前向传播，输入形状为 batch_size,1,H,W。
    def forward(self, images):
        # 返回每个类别的 logits，输出形状为 batch_size,num_classes。
        return self.network(images)

    # 定义 PyTorch 风格的预测接口，返回类别索引和最大概率。
    @torch.no_grad()
    def predict(self, images):
        # 将 logits 转换成类别概率。
        probabilities = torch.softmax(self(images), dim=1)
        # 取每个样本概率最大的类别和值。
        confidence, labels = torch.max(probabilities, dim=1)
        # 返回预测标签和对应置信度。
        return labels, confidence


# 定义自定义交叉熵损失函数，满足作业要求中的“自定义损失函数”。
class CustomCrossEntropyLoss(nn.Module):
    # 初始化损失对象。
    def __init__(self):
        # 调用 PyTorch 损失基类初始化方法。
        super().__init__()

    # 计算多分类交叉熵，logits 是未经过 softmax 的模型输出。
    def forward(self, logits, targets):
        # 使用 log_softmax 提高数值稳定性，dim=1 表示类别维度。
        log_probabilities = torch.log_softmax(logits, dim=1)
        # 根据每个样本的真实类别取出对应的对数概率。
        selected_log_probabilities = log_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)
        # 取负号并对 batch 求平均，得到标量损失。
        return -selected_log_probabilities.mean()


# 定义训练器类，负责训练、测试和每十轮输出准确率。
class Trainer:
    # 初始化模型、优化器、损失函数和设备。
    def __init__(self, model, train_loader, test_loader, device, learning_rate=1e-3):
        # 保存待训练模型。
        self.model = model.to(device)
        # 保存训练数据加载器，batch_size 决定每次取多少样本。
        self.train_loader = train_loader
        # 保存测试数据加载器。
        self.test_loader = test_loader
        # 保存运行设备，例如 cpu 或 cuda。
        self.device = device
        # 创建自定义交叉熵损失对象。
        self.criterion = CustomCrossEntropyLoss()
        # 创建 Adam 优化器，weight_decay 是 L2 正则系数，惩罚过大的权重以缓解过拟合。
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # 训练一个 epoch，并返回平均训练损失。
    def train_one_epoch(self):
        # 切换到训练模式，启用训练阶段行为。
        self.model.train()
        # 初始化累计损失。
        total_loss = 0.0
        # 遍历训练数据批次。
        for images, targets in self.train_loader:
            # 将图片和标签移动到同一个设备。
            images = images.to(self.device)
            # 将标签移动到同一个设备。
            targets = targets.to(self.device)
            # 清空上一个批次残留的梯度。
            self.optimizer.zero_grad()
            # 执行模型前向计算得到 logits。
            logits = self.model(images)
            # 使用自定义损失计算当前批次损失。
            loss = self.criterion(logits, targets)
            # 自动计算损失对模型参数的梯度。
            loss.backward()
            # 根据梯度更新模型参数。
            self.optimizer.step()
            # 按批次样本数累计损失。
            total_loss += loss.item() * images.size(0)
        # 返回整个训练集的平均损失。
        return total_loss / len(self.train_loader.dataset)

    # 在测试集上计算平均损失、总体准确率和每个数字的识别成功率。
    @torch.no_grad()
    def evaluate(self):
        # 切换到评估模式，关闭 dropout 等训练行为。
        self.model.eval()
        # 初始化正确样本数量。
        correct = 0
        # 初始化样本总数。
        total = 0
        # 初始化累计测试损失。
        total_loss = 0.0
        # 获取模型输出的类别数量。
        num_classes = self.model.network[-1].out_features
        # 初始化每个数字的预测正确数量。
        class_correct = [0] * num_classes
        # 初始化每个数字的测试样本总数。
        class_total = [0] * num_classes
        # 初始化混淆矩阵，行是真实数字，列是模型预测数字。
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        # 遍历测试数据批次。
        for images, targets in self.test_loader:
            # 将图片移动到运行设备。
            images = images.to(self.device)
            # 将标签移动到运行设备。
            targets = targets.to(self.device)
            # 计算模型输出。
            logits = self.model(images)
            # 计算当前批次的损失。
            loss = self.criterion(logits, targets)
            # 取最大 logit 对应的类别作为预测类别。
            predictions = logits.argmax(dim=1)
            # 将当前批次的真实标签移动到 CPU 并转为 Python 列表。
            targets_cpu = targets.cpu().tolist()
            # 将当前批次的预测标签移动到 CPU 并转为 Python 列表。
            predictions_cpu = predictions.cpu().tolist()
            # 累加预测正确的数量。
            correct += (predictions == targets).sum().item()
            # 累加样本数量。
            total += targets.size(0)
            # 累加当前批次损失。
            total_loss += loss.item() * images.size(0)
            # 遍历当前批次中的每个样本，统计每个数字和混淆矩阵。
            for true_label, predicted_label in zip(targets_cpu, predictions_cpu):
                # 在混淆矩阵中记录真实数字被预测成哪个数字。
                confusion_matrix[true_label][predicted_label] += 1
                # 当前真实数字的测试总数加一。
                class_total[true_label] += 1
                # 当前样本预测正确时对应数字的成功数量加一。
                if true_label == predicted_label:
                    # 当前数字识别成功数量累加。
                    class_correct[true_label] += 1
        # 防止没有测试样本时发生除零。
        if total == 0:
            # 没有样本时返回零损失、零准确率和空统计。
            return 0.0, 0.0, {}, confusion_matrix
        # 初始化每个数字识别成功率字典。
        class_accuracy = {}
        # 遍历每个数字类别的统计结果。
        for class_index in range(num_classes):
            # 当前数字没有测试样本时成功率记为零。
            if class_total[class_index] == 0:
                # 保存当前数字的成功率为零。
                class_accuracy[class_index] = 0.0
            # 当前数字存在测试样本时计算成功率。
            else:
                # 保存当前数字识别成功数量除以当前数字样本总数。
                class_accuracy[class_index] = class_correct[class_index] / class_total[class_index]
        # 返回平均损失、总体准确率、每个数字成功率和混淆矩阵。
        return total_loss / total, correct / total, class_accuracy, confusion_matrix

    # 训练指定轮数，并且每十个 epoch 输出一次准确率和每个数字成功率。
    def fit(self, epochs, plot_dir=None):
        # 初始化每个 epoch 的平均训练损失列表。
        train_losses = []
        # 初始化每个 epoch 的平均测试损失列表。
        test_losses = []
        # 初始化每个 epoch 的总体测试准确率列表。
        test_accuracies = []
        # 初始化最后一轮每个数字的成功率。
        last_class_accuracy = {}
        # 初始化最后一轮混淆矩阵。
        last_confusion_matrix = []
        # 循环执行指定数量的训练轮次。
        for epoch in range(1, epochs + 1):
            # 执行一轮训练并获取平均训练损失。
            train_loss = self.train_one_epoch()
            # 在测试集上计算损失、总体准确率、每个数字成功率和混淆矩阵。
            test_loss, test_accuracy, class_accuracy, confusion_matrix = self.evaluate()
            # 保存当前 epoch 的训练损失。
            train_losses.append(train_loss)
            # 保存当前 epoch 的测试损失。
            test_losses.append(test_loss)
            # 保存当前 epoch 的总体测试准确率。
            test_accuracies.append(test_accuracy)
            # 保存当前 epoch 的每个数字成功率，最后一轮会作为最终统计。
            last_class_accuracy = class_accuracy
            # 保存当前 epoch 的混淆矩阵，最后一轮会作为最终统计。
            last_confusion_matrix = confusion_matrix
            # 每十轮或最后一轮输出训练状态。
            if epoch % 10 == 0 or epoch == epochs:
                # 输出轮次、训练损失、测试损失和总体测试准确率。
                print(f"Epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | test_loss={test_loss:.4f} | accuracy={test_accuracy:.2%}")
                # 输出每个数字的识别成功率和样本数量。
                self._print_class_accuracy(class_accuracy, self.test_loader.dataset.samples)
        # 如果提供了图片保存目录，则绘制并保存训练曲线和分类统计图。
        if plot_dir is not None:
            # 绘制训练损失、测试损失和总体准确率曲线。
            self.plot_training_curves(train_losses, test_losses, test_accuracies, Path(plot_dir) / "training_curves.png")
            # 绘制每个数字识别成功率柱状图。
            self.plot_class_accuracy(last_class_accuracy, self.test_loader.dataset.samples, Path(plot_dir) / "class_accuracy.png")
            # 绘制混淆矩阵热力图。
            self.plot_confusion_matrix(last_confusion_matrix, Path(plot_dir) / "confusion_matrix.png")
        # 返回训练历史，便于外部继续分析。
        return train_losses, test_losses, test_accuracies

    @staticmethod
    def _print_class_accuracy(class_accuracy, samples):
        # 初始化每个数字的测试样本数量统计。
        class_counts = {}
        # 遍历测试样本列表，按标签统计每个数字的样本数量。
        for _, label in samples:
            # 当前数字的样本数量加一。
            class_counts[label] = class_counts.get(label, 0) + 1
        # 按数字标签从小到大排序后输出。
        for class_index in sorted(class_accuracy):
            # 读取当前数字的样本数量。
            count = class_counts.get(class_index, 0)
            # 读取当前数字的识别成功率。
            accuracy = class_accuracy[class_index]
            # 输出当前数字、成功率和样本数量。
            print(f"数字 {class_index} 识别成功率: {accuracy:.2%} | 测试样本数: {count}")

    @staticmethod
    def plot_training_curves(train_losses, test_losses, test_accuracies, save_path):
        # 设置 matplotlib 中文字体，Windows 自带黑体，保证中文图例正常显示。
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 解决中文字体下负号显示为方块的问题。
        plt.rcParams["axes.unicode_minus"] = False
        # 创建包含两个子图的画布，第一个子图画损失，第二个子图画准确率。
        figure, axes = plt.subplots(2, 1, figsize=(8, 6))
        # 生成 epoch 横坐标，从 1 开始。
        epochs = range(1, len(train_losses) + 1)
        # 在第一个子图中绘制训练损失曲线。
        axes[0].plot(epochs, train_losses, label="训练损失")
        # 在第一个子图中绘制测试损失曲线。
        axes[0].plot(epochs, test_losses, label="测试损失")
        # 设置第一个子图标题为损失曲线。
        axes[0].set_title("损失曲线")
        # 设置第一个子图横轴为 epoch。
        axes[0].set_xlabel("训练轮次")
        # 设置第一个子图纵轴为 loss。
        axes[0].set_ylabel("损失值")
        # 显示第一个子图图例。
        axes[0].legend()
        # 在第二个子图中绘制总体测试准确率曲线。
        axes[1].plot(epochs, test_accuracies, label="总体准确率")
        # 设置第二个子图标题为总体准确率曲线。
        axes[1].set_title("测试集总体准确率曲线")
        # 设置第二个子图横轴为 epoch。
        axes[1].set_xlabel("训练轮次")
        # 设置第二个子图纵轴为 accuracy。
        axes[1].set_ylabel("准确率")
        # 显示第二个子图图例。
        axes[1].legend()
        # 自动调整子图布局，避免文字重叠。
        figure.tight_layout()
        # 保存训练曲线图片。
        figure.savefig(save_path, dpi=200)
        # 关闭画布释放内存。
        plt.close(figure)
        # 输出训练曲线保存路径。
        print(f"训练曲线图已保存: {save_path}")

    @staticmethod
    def plot_class_accuracy(class_accuracy, samples, save_path):
        # 设置 matplotlib 中文字体，Windows 自带黑体，保证中文图例正常显示。
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 解决中文字体下负号显示为方块的问题。
        plt.rcParams["axes.unicode_minus"] = False
        # 初始化每个数字的测试样本数量统计。
        class_counts = {}
        # 遍历测试样本列表，按标签统计每个数字的样本数量。
        for _, label in samples:
            # 当前数字的样本数量加一。
            class_counts[label] = class_counts.get(label, 0) + 1
        # 按数字标签排序，确保柱状图顺序稳定。
        sorted_labels = sorted(class_accuracy)
        # 提取每个数字的识别成功率。
        accuracies = [class_accuracy[label] for label in sorted_labels]
        # 提取每个数字对应的测试样本数量。
        counts = [class_counts.get(label, 0) for label in sorted_labels]
        # 创建柱状图画布。
        figure, axes = plt.subplots(figsize=(8, 5))
        # 绘制每个数字识别成功率柱状图。
        bars = axes.bar([str(label) for label in sorted_labels], accuracies, color="#4f8ef7")
        # 在每个柱子上方标注成功率和样本数量。
        for bar, accuracy, count in zip(bars, accuracies, counts):
            # 在柱子顶部添加百分比和样本数量文字。
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{accuracy:.2%}\n(n={count})", ha="center", va="bottom", fontsize=8)
        # 设置柱状图标题。
        axes.set_title("每个数字识别成功率")
        # 设置横轴为数字类别。
        axes.set_xlabel("数字类别")
        # 设置纵轴为识别成功率。
        axes.set_ylabel("识别成功率")
        # 设置纵轴范围，从 0 到 1。
        axes.set_ylim(0, 1.1)
        # 自动调整布局。
        figure.tight_layout()
        # 保存每个数字成功率柱状图。
        figure.savefig(save_path, dpi=200)
        # 关闭画布释放内存。
        plt.close(figure)
        # 输出柱状图保存路径。
        print(f"每个数字成功率柱状图已保存: {save_path}")

    @staticmethod
    def plot_confusion_matrix(confusion_matrix, save_path):
        # 设置 matplotlib 中文字体，Windows 自带黑体，保证中文图例正常显示。
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        # 解决中文字体下负号显示为方块的问题。
        plt.rcParams["axes.unicode_minus"] = False
        # 将混淆矩阵转换为 NumPy 数组，方便绘图。
        matrix = np.array(confusion_matrix, dtype=int)
        # 检查混淆矩阵是否有效。
        if matrix.size == 0:
            # 混淆矩阵为空时直接返回。
            return
        # 创建混淆矩阵热力图画布。
        figure, axes = plt.subplots(figsize=(6, 5))
        # 使用 imshow 绘制混淆矩阵热力图。
        image = axes.imshow(matrix, cmap="Blues")
        # 在热力图右侧添加颜色条。
        figure.colorbar(image, ax=axes)
        # 设置横轴为预测数字。
        axes.set_xlabel("预测数字")
        # 设置纵轴为真实数字。
        axes.set_ylabel("真实数字")
        # 设置标题为混淆矩阵。
        axes.set_title("混淆矩阵")
        # 设置横轴刻度位置。
        axes.set_xticks(range(matrix.shape[1]))
        # 设置横轴刻度标签。
        axes.set_xticklabels(range(matrix.shape[1]))
        # 设置纵轴刻度位置。
        axes.set_yticks(range(matrix.shape[0]))
        # 设置纵轴刻度标签。
        axes.set_yticklabels(range(matrix.shape[0]))
        # 遍历矩阵每个格子并标注数量。
        for row in range(matrix.shape[0]):
            # 遍历当前行的每一列。
            for col in range(matrix.shape[1]):
                # 在每个格子中心写入真实数量。
                axes.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=8)
        # 自动调整布局。
        figure.tight_layout()
        # 保存混淆矩阵热力图。
        figure.savefig(save_path, dpi=200)
        # 关闭画布释放内存。
        plt.close(figure)
        # 输出混淆矩阵保存路径。
        print(f"混淆矩阵热力图已保存: {save_path}")


# 定义程序入口函数，集中管理数据集、模型和训练流程。
def main():
    # 设置本地手写数据集根目录，下面必须包含 train 和 test 两个目录。
    dataset_root = Path(r"D:\zyf\data\handwritten")
    # 设置训练集目录，兼容 train/类别名/图片 和 train/类别_编号.png 两种结构。
    train_root = dataset_root / "train"
    # 设置测试集目录，兼容 test/类别名/图片 和 test/类别_编号.png 两种结构。
    test_root = dataset_root / "test"
    # 设置统一图片尺寸，网络输入为 1*32*32。
    image_size = 32
    # 设置训练批大小。
    batch_size = 64
    # 设置训练轮数。
    epochs = 50
    # 设置训练图表保存目录，运行结束后自动保存损失曲线、准确率曲线、柱状图和混淆矩阵。
    plot_dir = dataset_root / "plots"
    # 如果图表目录不存在则自动创建。
    plot_dir.mkdir(parents=True, exist_ok=True)
    # 固定 Python 随机种子，保证数据划分可复现。自己写的打乱列表、随机采样逻辑要想复现必须写
    random.seed(42)
    # 固定 NumPy 随机种子，保证数据处理可复现。数据预处理、数组随机操作、很多第三方库底层用它
    np.random.seed(42)
    # 固定 PyTorch 随机种子，保证参数初始化可复现。模型参数初始化、DataLoader 的 shuffle、Dropout
    torch.manual_seed(42)
    # 根据 CUDA 是否可用选择训练设备。
    """
    作用：本身不做任何计算，它就是一个“地址标签”，用来指明张量/模型放在哪个硬件上：CPU 还是 GPU（显存）。
    目前这句函数的意思是有GPU就用GPU  没有GPU就用CPU
    torch.device("cpu")        # CPU
    torch.device("cuda")       # 第 0 块 GPU
    torch.device("cuda:1")     # 第 1 块 GPU（编号写在字符串里）
    
    torch.device("cuda", 0)    # 等价于 torch.device("cuda:0")
    torch.device("cuda", 1)    # 等价于 torch.device("cuda:1")

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 创建训练数据集，并由训练集目录建立类别名称到整数标签的映射。
    train_dataset = HandwrittenDataset(train_root, image_size=image_size)
    # 创建测试数据集，并复用训练集类别映射，保证同一类别对应同一标签。
    test_dataset = HandwrittenDataset(test_root, image_size=image_size, class_to_index=train_dataset.class_to_index)
    # 创建训练数据加载器，shuffle=True 表示每个 epoch 打乱训练样本。
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 创建测试数据加载器，测试阶段不需要打乱样本。
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 创建自定义手写字符分类模型。
    model = HandwrittenClassifier(train_dataset.num_classes, image_size=image_size)
    # 创建训练器并指定 Adam 学习率。
    trainer = Trainer(model, train_loader, test_loader, device, learning_rate=1e-3)
    # 输出数据集、类别和设备信息，便于检查配置。
    print(f"classes={train_dataset.index_to_class} | train={len(train_dataset)} | test={len(test_dataset)} | device={device}")
    # 开始训练并每十轮输出一次测试准确率和每个数字成功率，同时保存图表。
    trainer.fit(epochs, plot_dir=plot_dir)


# 只有直接运行本文件时才执行训练入口。
if __name__ == "__main__":
    # 调用主函数启动手写字符识别训练。
    main()
