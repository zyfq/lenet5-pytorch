# 导入操作系统路径工具，用于定位 runs 目录下的 pt 文件。
from pathlib import Path
# 导入 PyTorch，用于读取 pt 文件。
import torch


# 在 runs 目录下查找最新的 best.pt 和 last.pt。
def find_latest_pts(runs_dir):
    """
    查找每个时间戳目录下的权重文件，返回最新的 best.pt 和 last.pt。

    形参：
        runs_dir  runs 目录路径，每个子目录是一次训练的时间戳。
    返回：
        (best_path, last_path) 二元组，找不到时对应位置为 None。
    """
    # runs 目录不存在时直接返回空。
    if not runs_dir.is_dir():
        return None, None
    # 按修改时间排序，找出全部 best.pt 候选。
    best_candidates = sorted(runs_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    # 按修改时间排序，找出全部 last.pt 候选。
    last_candidates = sorted(runs_dir.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
    # 取最新的文件，没有时记为 None。
    best_path = best_candidates[-1] if best_candidates else None
    last_path = last_candidates[-1] if last_candidates else None
    # 返回最新权重路径。
    return best_path, last_path


# 打印单个 pt 文件的全部内容结构。
def inspect_one(checkpoint_path):
    """
    读取并打印单个 pt 文件的顶层字段、训练信息和参数形状。

    形参：
        checkpoint_path  pt 文件路径，例如 runs/时间戳/weights/best.pt。
    """
    # 打印当前检查的文件路径和字节大小。
    print(f"文件: {checkpoint_path} | 大小: {checkpoint_path.stat().st_size} 字节")
    # 用 map_location=cpu 读取，保证任何机器都能查看。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # 打印顶层字典有哪些键。
    print(f"顶层字段: {list(checkpoint.keys())}")
    # 打印保存时的轮次和测试准确率。
    print(f"epoch={checkpoint.get('epoch')} | test_accuracy={checkpoint.get('test_accuracy')}")
    # 打印 history 里记录了多少轮。
    history = checkpoint.get("history", {})
    # history 是字典，每个值是列表，长度就是已训练轮数。
    print(f"history轮数: {len(history.get('epoch', []))} | history字段: {list(history.keys())}")
    # 打印 history 最后一轮的指标，确认和 test_accuracy 是否一致。
    if history.get("epoch"):
        last_index = len(history["epoch"]) - 1
        print(
            f"history最后一轮: epoch={history['epoch'][last_index]} | "
            f"train_loss={history['train_loss'][last_index]:.4f} | "
            f"test_loss={history['test_loss'][last_index]:.4f} | "
            f"train_acc={history['train_accuracy'][last_index]:.2%} | "
            f"test_acc={history['test_accuracy'][last_index]:.2%}"
        )
    # 打印模型权重部分：每层名字、形状、数据类型。
    model_state = checkpoint.get("model_state_dict", {})
    print(f"模型共 {len(model_state)} 个参数张量:")
    total_params = 0
    for name, tensor in model_state.items():
        # 统计当前张量的元素个数。
        numel = tensor.numel()
        total_params += numel
        # 打印层名、形状、类型、均值，确认权重不是全零或异常值。
        print(f"  {name} | 形状={tuple(tensor.shape)} | 类型={tensor.dtype} | 均值={float(tensor.float().mean()):.6f} | 数量={numel}")
    # 打印模型总参数量。
    print(f"模型总参数量: {total_params}")
    # 打印优化器部分：学习率和动量状态数量。
    optimizer_state = checkpoint.get("optimizer_state_dict", {})
    param_groups = optimizer_state.get("param_groups", [])
    state = optimizer_state.get("state", {})
    print(f"优化器参数组数: {len(param_groups)} | 优化器状态数: {len(state)}")
    for index, group in enumerate(param_groups):
        print(f"  参数组{index}: lr={group.get('lr')} | weight_decay={group.get('weight_decay')}")
    print("-" * 70)


# 程序入口：自动找最新 pt 并逐个打印内容。
def main():
    """
    主流程：定位最新 best.pt 和 last.pt -> 逐个打印内部结构。
    """
    # runs 目录固定在当前文件同级。
    runs_dir = Path(__file__).resolve().parent / "runs"
    # 查找最新的两个权重文件。
    best_path, last_path = find_latest_pts(runs_dir)
    # 两个都找不到时提示先训练。
    if best_path is None and last_path is None:
        print(f"在 {runs_dir} 下没有找到任何 weights/best.pt 或 weights/last.pt，请先运行 main.py 训练。")
        return
    # 找到 best.pt 时打印其内容。
    if best_path is not None:
        print("========== best.pt ==========")
        inspect_one(best_path)
    # 找到 last.pt 时打印其内容。
    if last_path is not None:
        print("========== last.pt ==========")
        inspect_one(last_path)


# 直接运行本文件时执行查看入口。
if __name__ == "__main__":
    main()
