# LeNet5 手写字符识别（PyTorch）

使用经典 LeNet-5 完成手写字符分类任务。项目按模块化结构组织，
训练过程自动记录指标并保存曲线图与检查点。

## 实验结果

- 最佳测试准确率：**98.63%**（第 18 / 50 轮）
- 每次训练自动输出到 `runs/<时间戳>/`：`metrics.csv`、`metrics.json`、曲线图、`best.pt`、`last.pt`

## 项目结构

```
main.py                 # 入口：组装数据、模型、训练器并启动训练
model/
  backbone.py           # 卷积池化主干
  head.py               # 全连接头
  lenet5.py             # LeNet-5 主体（BackBone + Head）
  classifier.py         # 全连接对照模型（use_lenet5=False 时使用）
  trainer.py            # 训练器：训练/评估/保存/TensorBoard
dataset_opt/
  dataset.py            # HandwrittenDataset 图片数据集
loss/custom_loss.py     # 自定义损失
optimizer/build_optimizer.py
picturedraw/plots.py    # 训练曲线绘制
predict_pt.py           # 加载 .pt 权重做预测
inspect_pt.py           # 查看检查点内容
resume_train.py         # 断点续训
```

## 运行方式

```bash
python main.py
```

配置集中在 `main.py` 中：数据根目录、图片尺寸 32×32、batch_size=64、
epochs=50、学习率 1e-3、随机种子 42（可复现）。将 `use_lenet5` 改为
`False` 可切换到全连接对照模型。

## 环境依赖

- Python 3.x
- PyTorch（自动检测 CUDA，有 GPU 用 GPU，无 GPU 用 CPU）
- numpy / pandas / matplotlib（绘制曲线）
