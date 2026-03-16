# EWAP 环境 (ewap_env)

本目录包含了 H-GAT-GT (Hierarchical Graph Attention Tracker - Group Tracking) 模型在 EWAP (ETH Walking Pedestrians) 数据集上的核心实现、数据处理、训练、评估和可视化脚本。

## 文件说明

### 1. 配置与数据加载
*   **`config.py`**: 中央配置文件，定义了数据路径、数据集生成参数（帧块、维度）、模型超参数、训练设置（批量大小、学习率）以及评估/输出设置。
*   **`dataset.py`**: 使用 PyTorch Geometric 定义了 `RadarFileDataset` 类。负责加载处理后的 `.npy` 片段数据，构建图结构（节点代表行人，边基于空间距离），并提取节点特征（位置、速度）和边属性。

### 2. 数据准备
*   **`prepare_ewap.py`**: 预处理脚本，读取原始 EWAP 数据集标注（`obsmat.txt` 和 `groups.txt`），并将其转换为适用于 STGAT 和基准跟踪评估的 `.npy` 文件。处理坐标缩放、速度计算以及将序列切分为片段（episodes）。
*   **`prepare_pseudo_data.py`**: 利用基于物理规则的方法生成“伪标签”训练数据。它分析原始数据集中行人的距离、速度大小和运动方向（余弦相似度）来进行分组，并输出 `finetune_train` 和 `finetune_val` 数据集。这使得模型可以在没有人工标注群体标签的情况下进行训练。

### 3. 模型架构
*   **`model.py`**: 定义神经网络架构。核心类为 `GNNGroupTracker`，包含：
    *   `FourierFeatureEncoder`：将 2D 坐标编码为高维傅里叶特征，增强空间感知能力。
    *   核心图神经网络骨干：使用多个级联的 `TransformerConv` 层。
    *   `AdaptiveLayerFusion`：自适应地融合来自不同 Transformer 层的特征，而非简单的拼接。
    *   解码器：`edge_classifier`（预测两个节点是否属于同一组）和 `offset_regressor`（预测边界框/中心跟踪偏移量及其不确定性）。

### 4. 训练与评估
*   **`train_ewap.py`**: 训练脚本，用于在 EWAP 数据集上从头开始训练（或微调）GNN 模型。它加载由 `prepare_pseudo_data.py` 生成的伪标签，并使用组合损失函数（用于边连接的二元交叉熵和用于偏移回归的负对数似然）优化模型。
*   **`evaluate_ewap.py`**: 全面的评估基准脚本。在测试数据集上评估训练好的 `H-GAT-GT` 模型，并与多种先进及基准跟踪器（如 Social-STGCNN, GM-CPHD, CBMeMBer, Graph-MB）进行对比，比较 MOTA, MOTP, IDSW 和 OSPA 等标准跟踪指标。

### 5. 可视化
*   **`visualize_eth_on_video.py`**: 推理与可视化脚本。读取原始视频（`seq_eth.avi`），运行训练好的模型预测群体关联和偏移，应用后处理（连通分量）形成群体，并将边界多边形、轨迹和跟踪 ID 直接渲染到视频帧上。同时在本地推理过程中计算 ARI, Precision, Recall 和 F1-Score 等聚类指标。
