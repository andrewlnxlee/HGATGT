# 消融实验模块 (Ablation Study)

本目录包含了用于验证 **H-GAT-GT** (Hierarchical Graph Attention Group Tracker) 各核心组件有效性的消融实验代码。


## 目录结构

| 文件名 | 说明 |
| :--- | :--- |
| `ablation_model.py` | **消融模型定义**：支持动态配置开关的 GNN 跟踪器，可切换傅里叶编码、融合模式及卷积层类型。 |
| `train_ablation.py` | **训练脚本**：用于训练不同的模型变体。为了提高效率，默认仅使用部分训练数据进行快速迭代。 |
| `run_comparison.py` | **对比评估脚本**：加载训练好的消融模型与全模型（Benchmark）进行性能指标对比。 |
| `dataset.py` | **数据集加载器**：适配消融实验需求的图数据构建工具，支持自动识别节点和边特征维度。 |
| `model/` | 存储训练好的模型权重（`.pth` 文件）。 |
| `output/result/` | 存储实验结果汇总（如 `.csv` 报告）。 |

## 消融实验项

通过配置 `AblationGNNTracker` 的参数，我们可以验证以下组件的影响：

1.  **Fourier Feature Encoder (傅里叶特征编码器)**:
    - 验证高频位置编码对小目标（行人）检测精度的提升。
    - 对应配置：`use_fourier=False`
2.  **Adaptive Layer Fusion (自适应层融合)**:
    - 验证通过注意力机制融合多层 GNN 特征对长程依赖捕捉的改进。
    - 对应配置：`fusion_mode='last'` (仅使用最后一层)
3.  **TransformerConv vs. Plain GCN**:
    - 验证多头注意力机制在边缘分类任务中相比普通卷积的优势。
    - 对应配置：`use_transformer=False`

## 使用指南

### 1. 训练消融变体
在 `train_ablation.py` 的 `if __name__ == "__main__":` 部分配置需要训练的实验项，然后运行：
```bash
python sim_env/run_ablation/train_ablation.py
```
*注意：训练完成后，模型将自动保存至 `sim_env/run_ablation/model/` 目录下。*

### 2. 运行性能对比
运行 `run_comparison.py` 脚本，它会自动加载全模型和已有的消融模型变体，并在测试集上计算对比指标（如 MOTA, Precision, Recall 等）：
```bash
python sim_env/run_ablation/run_comparison.py
```
实验结果将打印在控制台，并保存为 `output/result/ablation_comparison_final.csv`。

## 注意事项
- 确保 `config.py` 中的路径配置正确。
- **消融实验默认在 `sim_env` 环境下运行**，模型输入维度会根据加载的数据自动适配。
