# ewap_env — EWAP 真实数据集实验环境

本目录是 H-GAT-GT 模型在 **EWAP (ETH Walking Pedestrians)** 真实行人数据集上的完整实验环境，涵盖数据预处理、伪标签生成、模型训练、多算法对比评估及可视化全流程。

---

## 目录结构总览

```
ewap_env/
├── config.py                         # 全局配置文件（数据、模型、训练、输出路径等）
├── dataset.py                        # PyG 数据集类，将 .npy 文件加载为图序列
├── prepare_ewap.py                   # EWAP 原始数据 → .npy 格式预处理脚本
├── prepare_pseudo_data.py            # 基于物理规则的伪标签生成 + 训练数据制作
├── model.py                          # H-GAT-GT 模型定义（傅里叶编码 + 多层 TransformerConv + 自适应融合）
├── train_ewap.py                     # 从头训练脚本（使用伪标签数据）
├── evaluate_ewap.py                  # 多算法对比评估脚本（6 种跟踪器）
├── visualize_eth_on_video.py         # 在 ETH 原始视频上叠加群体检测结果并统计准确率
├── model/
│   └── best_model_ewap_v1.pth        # 训练好的最佳模型权重
├── output/
│   ├── result/
│   │   ├── EWAP_compare.txt          # 多算法跟踪指标对比结果
│   │   └── group_analyse.txt         # 群体划分准确率统计报告
│   └── track_result_mp4/
│       └── eth_v1.mp4                # 可视化演示视频
└── README.md                         # 本文件
```

---

## 各文件详细说明

### 1. `config.py` — 全局配置

集中管理所有可调参数，供其他模块统一引用。

| 配置分区 | 关键参数 | 说明 |
|---------|---------|------|
| **数据生成** | `DATA_ROOT = "./data"` | 数据根目录 |
| | `NUM_TRAIN_SAMPLES = 2000` | 训练集样本数 |
| | `NUM_VAL_SAMPLES = 200` | 验证集样本数 |
| | `FRAMES_PER_SAMPLE = 50` | 每个 episode 的帧数 |
| | `MAX_GROUPS = 5` | 场景中最大群组数 |
| **模型** | `INPUT_DIM = 4` | 节点特征维度 (x, y, vx, vy) |
| | `EDGE_DIM = 6` | 边特征维度 (dx, dy, dist, dvx, dvy, cos_sim) |
| | `HIDDEN_DIM = 64` | 隐藏层维度 |
| **训练** | `LEARNING_RATE = 0.001` | 学习率 |
| | `EPOCHS = 50` | 训练轮数 |
| | `DEVICE = "cuda"` | 运行设备 |
| **EWAP 专用** | `EWAP_MODEL_SAVE_PATH` | 模型保存路径（训练时写入） |
| | `EWAP_MODEL_USE_PATH` | 模型使用路径（推理/评估时读取） |
| | `COORD_SCALE = 50.0` | 坐标缩放因子，将原始米制坐标 (~[-10, 15]) 映射到 (~[0, 1000]) |
| | `COORD_OFFSET = [500, 500]` | 坐标中心偏移 |
| **输出** | `OUTPUT_MP4_DIR` | 可视化视频输出目录 |

---

### 2. `dataset.py` — PyG 图数据集

**类：`RadarFileDataset`**（继承 `torch_geometric.data.Dataset`）

负责将 `.npy` 格式的 episode 数据加载为 PyTorch Geometric 的图数据序列。

**核心流程 (`get` 方法)：**

1. **加载 `.npy` 文件**：每个文件是一个 episode（包含多帧数据），每帧包含 `meas`（测量点）、`labels`（群组标签）、`gt_centers`（真值中心）
2. **节点特征构建**：直接使用 `meas` 数组作为节点特征 `x`，维度为 `[N, 4]`（x, y, vx, vy）
3. **图构建**：基于节点间欧氏距离，半径 `conn_radius=80.0` 内的节点对连边
4. **边特征计算**（6维）：
   - `rel_pos` (2维)：相对位置差 (dx, dy)
   - `dist` (1维)：欧氏距离
   - `rel_v` (2维)：相对速度差 (dvx, dvy)
   - `cos_sim` (1维)：速度方向余弦相似度
5. **边标签生成**：两端节点属于同一非零群组则标记为正样本（`edge_label=1`）
6. **封装为 `Data` 对象**：包含 `x`, `edge_index`, `edge_attr`, `edge_label`, `point_labels`, `gt_centers`

---

### 3. `prepare_ewap.py` — EWAP 数据预处理

将 EWAP 原始 `obsmat.txt` 文件转换为模型可用的 `.npy` 格式。

**输入数据格式 (`obsmat.txt`)：**
```
[frame_number, pedestrian_ID, pos_x, pos_z, pos_y, v_x, v_z, v_y]
```

**处理流程：**

1. **解析 `obsmat.txt`**：逐行读取，按帧号分组，提取行人 ID、位置 (x, y)、速度 (vx, vy)
2. **群组标签分配**：当前实现中强制每人独立成群（`ped_to_group = {}`），用于多目标跟踪 (MOT) 评估
3. **坐标变换**：
   - 位置缩放：`scaled_pos = raw_pos × 50.0 + [500, 500]`
   - 速度缩放：`scaled_vel = raw_vel × 0.4 × 50.0`（ETH 数据集 FPS=2.5，帧间隔 dt=0.4s）
4. **计算群质心 (`gt_centers`)**：按群组 ID 计算均值位置
5. **切分 episode**：每 50 帧切分为一个 episode，保存为 `.npy` 文件

**处理的场景：**
| 原始场景 | 输出目录 |
|---------|---------|
| `seq_eth` | `data/test_ewap_eth/` |
| `seq_hotel` | `data/test_ewap_hotel/` |

---

### 4. `prepare_pseudo_data.py` — 伪标签生成

基于物理规则自动为行人对标注"同行"关系，生成训练用伪标签数据。

**核心函数：`compute_pseudo_labels(obsmat_path)`**

**物理规则判定逻辑（判断两人是否同行）：**

1. **距离约束**：两人间距 < 2.0 米
2. **运动状态判断**：
   - 若双方均近似静止（速度 < 0.2 m/s）→ 判定为同行
   - 若双方均在运动（速度 ≥ 0.2 m/s）→ 进一步检查：
     - 速度方向余弦相似度 > 0.85（方向一致）
     - 速度差 < 0.6 m/s（速率接近）
3. **时间鲁棒性**：只有在 ≥ 5 帧中满足上述条件的行人对才被判定为同行
4. **连通分量聚类**：使用稀疏图的连通分量算法将满足条件的行人对聚合为群组

**核心函数：`generate_training_data(scene_name, ped_to_group, split)`**

将伪标签与原始轨迹数据结合，生成适配 `RadarFileDataset` 格式的 `.npy` 训练数据：
- 使用**滑动窗口**切分（步长为窗口一半），起到数据增强效果
- 输出到 `data/finetune_train/` 和 `data/finetune_val/`

---

### 5. `model.py` — H-GAT-GT 模型定义

**模型全称：** Hierarchical Graph Attention Group Tracker (H-GAT-GT)

#### 模块 1：`FourierFeatureEncoder` — 傅里叶位置编码

- 将 2D 位置坐标 `[x, y]` 通过随机高斯矩阵 `B` 映射到高维空间
- 输出 `[sin(2πxB), cos(2πxB)]`，维度为 `mapping_size × 2 = 64`
- 目的：增强模型对空间位置的感知能力

#### 模块 2：`AdaptiveLayerFusion` — 自适应层融合

- 学习 4 个层级的注意力权重 α₁, α₂, α₃, α₄（通过 softmax 归一化）
- 对多层特征进行加权求和：`h_final = Σ αᵢ × hᵢ`
- 相比简单拼接 (Concat)，参数量更小、表达更精准

#### 主模型：`GNNGroupTracker`

**架构总览：**

```
输入 → 傅里叶编码 + 原始特征拼接 → Node MLP → 4层 TransformerConv Backbone → 自适应融合 → 双头解码
```

**详细结构：**

| 阶段 | 组件 | 维度变化 |
|------|------|---------|
| **输入编码** | 位置傅里叶编码 + 原始4维特征拼接 | [N, 4] → [N, 68] |
| | Node MLP (2层 Linear + LayerNorm + GELU) | [N, 68] → [N, 96] |
| | Edge Encoder (2层 Linear + LayerNorm + GELU) | [E, 6] → [E, 96] |
| **Backbone** | TransformerConv Layer 1 (4头, hidden=96) + BN + GELU + 残差 | [N, 96] → [N, 96] |
| | TransformerConv Layer 2 + 残差 | [N, 96] → [N, 96] |
| | TransformerConv Layer 3 + 残差 | [N, 96] → [N, 96] |
| | TransformerConv Layer 4 + 残差 | [N, 96] → [N, 96] |
| **融合** | AdaptiveLayerFusion (加权融合4层输出) | 4×[N, 96] → [N, 96] |
| **解码 - 边分类** | 拼接 [src, dst, edge_attr] → MLP → Sigmoid | [E, 288] → [E, 1] |
| **解码 - 偏移回归** | MLP → (dx, dy, σ_x, σ_y) | [N, 96] → [N, 4] |

**输出：**
- `edge_scores`：边分类概率（同群为1，不同群为0）
- `pred_offsets`：节点到群质心的偏移预测 (dx, dy)
- `pred_uncertainty`：偏移预测的不确定性 (σ_x, σ_y)，经 softplus 保证正值
- `h_final`：节点的高维特征表示，可用于下游数据关联

---

### 6. `train_ewap.py` — 模型训练

从头 (from scratch) 在 ETH 伪标签数据上训练 H-GAT-GT 模型。

**数据集类：`ETHScratchDataset`**
- 继承 `RadarFileDataset`，使用 `data/finetune_{split}/` 中的伪标签数据
- 建图半径 `conn_radius = 30.0`

**损失函数：`compute_loss`**

由两部分组成：

1. **边分类损失 (`loss_edge`)**：
   - 二元交叉熵 (BCE)，作用于模型预测的边分类概率与真值标签之间
   - 支持正负样本动态加权（最大 10 倍）

2. **偏移回归损失 (`loss_reg`)**：
   - 负对数似然 (NLL) 损失，结合不确定性估计
   - 公式：`L_nll = 0.5 × [MSE/σ² + log(σ²)]`
   - 仅对有效标签点（`uid != 0` 且在 `gt_centers` 中）计算

**总损失**：`L = L_edge + 1.0 × L_reg`

**训练配置：**

| 参数 | 值 | 说明 |
|------|-----|------|
| 优化器 | AdamW | lr=0.0005, weight_decay=1e-4 |
| 学习率调度 | CosineAnnealingWarmRestarts | T₀=10, T_mult=2 |
| 梯度累积 | 8 步 | 等效扩大 batch size |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| 总 Epoch | 100 | 从头训练需更多轮次 |
| 模型保存 | 按验证集最低 loss 保存 | `best_model_eth_scratch.pth` |

---

### 7. `evaluate_ewap.py` — 多算法对比评估

在 EWAP 数据集（ETH / Hotel 两个场景）上对 **6 种跟踪算法** 进行公平对比评估。

**评估的算法：**

| 算法 | 类名 | 类型 |
|------|------|------|
| Baseline | `BaselineTracker` | DBSCAN 聚类 + 简单关联 |
| GM-CPHD | `GMCPHDTracker` | 高斯混合基数概率假设密度滤波 |
| CBMeMBer | `CBMeMBerTracker` | 基数均衡多伯努利滤波 |
| Social-STGCNN | `SocialSTGCNNTracker` | 社交时空图卷积网络 |
| Graph-MB | `GraphMBTracker` | 基于图的多伯努利滤波 |
| **H-GAT-GT (Ours)** | `GNNPostProcessor` | 本文方法 |

**评估协议：**
- 所有算法接收相同的带噪声测量点作为输入（噪声强度：位置 σ=3.0，速度 σ=0.5）
- H-GAT-GT 的输入也经过相同噪声处理，确保公平
- 边特征会在推理时根据带噪输入重新计算

**评估指标：**

| 指标 | 全称 | 含义 |
|------|------|------|
| MOTA | Multiple Object Tracking Accuracy | 多目标跟踪准确率（越高越好） |
| MOTP | Multiple Object Tracking Precision | 跟踪精度/位置误差（越低越好） |
| OSPA | Optimal Sub-Pattern Assignment | 最优子模式分配距离（越低越好） |
| IDSW | ID Switches | ID 切换次数（越低越好） |
| RMSE (Pos) | Root Mean Square Error | 位置均方根误差（越低越好） |
| Purity | Clustering Purity | 聚类纯度（越高越好） |
| Time | Inference Time | 推理耗时 |

---

### 8. `visualize_eth_on_video.py` — 视频可视化 + 准确率统计

在 ETH 原始监控视频上叠加 H-GAT-GT 的群体检测与跟踪结果，同时统计群体划分准确率。

**主要流程：**

1. **坐标转换 (`world_to_pixel`)**：利用单应性矩阵 `H` 将世界坐标转换为像素坐标
2. **逐帧推理**：
   - 构造 4D 节点特征 (位置+速度) 和 6D 边特征
   - 使用 GNN 模型预测边分类概率和偏移
   - **动态阈值**：`threshold = max(0.05, max_score × 0.28)`
   - **运动约束过滤**：余弦相似度 > 0.40 且速度差 < 0.7
   - 连通分量算法生成群组标签
3. **跟踪关联**：使用 `GNNPostProcessor` + 匈牙利算法将当前帧的群组与历史跟踪 ID 关联
4. **渲染**：
   - 多人群组：凸包/连线半透明覆盖
   - 个体轨迹：渐变色尾迹（最近 25 帧）
   - 跟踪 ID 标注

**群体划分准确率统计（与物理规则伪标签对比）：**

| 指标 | 结果值 |
|------|-------|
| 平均调整兰德系数 (ARI) | 0.7988 |
| 两两聚类准确率 (Precision) | 97.53% |
| 两两聚类召回率 (Recall) | 65.05% |
| F1-Score | 0.7805 |

---

### 9. `model/` — 模型权重目录

| 文件 | 说明 |
|------|------|
| `best_model_ewap_v1.pth` | 训练好的最佳模型权重（用于评估和可视化） |

---

### 10. `output/` — 输出结果目录

#### `output/result/EWAP_compare.txt` — 多算法对比结果

**ETH 场景：**

| 算法 | MOTA↑ | MOTP↓ | OSPA↓ | IDSW↓ | RMSE↓ |
|------|-------|-------|-------|-------|-------|
| Baseline | 0.782 | 4.615 | 15.81 | 283 | 5.669 |
| GM-CPHD | 0.847 | 4.461 | 9.063 | 738 | 5.548 |
| CBMeMBer | 0.736 | 4.497 | 17.61 | 365 | 6.183 |
| Social-STGCNN | 0.920 | 4.336 | 8.709 | 205 | 5.229 |
| Graph-MB | 0.016 | 13.21 | 49.72 | 25 | 13.72 |
| **H-GAT-GT (Ours)** | **0.957** | **3.746** | **4.073** | 384 | **4.235** |

**Hotel 场景：**

| 算法 | MOTA↑ | MOTP↓ | OSPA↓ | IDSW↓ | RMSE↓ |
|------|-------|-------|-------|-------|-------|
| Baseline | 0.722 | 4.727 | 19.60 | 216 | 5.931 |
| GM-CPHD | 0.828 | 4.617 | 11.29 | 452 | 5.793 |
| CBMeMBer | 0.699 | 4.528 | 20.00 | 237 | 5.973 |
| Social-STGCNN | 0.903 | 3.773 | 9.385 | 151 | 4.471 |
| Graph-MB | 0.037 | 11.24 | 49.27 | 2 | 11.61 |
| **H-GAT-GT (Ours)** | **0.950** | **3.732** | **4.083** | 329 | **4.195** |

> H-GAT-GT 在 MOTA、MOTP、OSPA、RMSE 上全面领先，显著优于所有基线方法。

#### `output/result/group_analyse.txt` — 群体划分性能报告

H-GAT 在 ETH 数据集上的群体划分结果，Precision 极高 (97.53%)，说明模型对群体判断非常精确。

#### `output/track_result_mp4/eth_v1.mp4` — 可视化演示视频

在 ETH 原始监控视频上叠加群体检测与跟踪结果的演示视频（10fps，1500帧采样点）。

---

## 完整工作流

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: 数据预处理                                              │
│  python prepare_ewap.py                                         │
│  (obsmat.txt → data/test_ewap_eth/, data/test_ewap_hotel/)      │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: 伪标签生成                                              │
│  python prepare_pseudo_data.py                                   │
│  (物理规则 → data/finetune_train/, data/finetune_val/)           │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: 模型训练                                                │
│  python train_ewap.py                                            │
│  (从头训练 → best_model_eth_scratch.pth)                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 4: 多算法对比评估                                          │
│  python evaluate_ewap.py                                         │
│  (6种算法 × 2个场景 → EWAP_compare.txt)                          │
├─────────────────────────────────────────────────────────────────┤
│  Step 5: 可视化                                                  │
│  python visualize_eth_on_video.py                                │
│  (GNN推理 + 视频渲染 → eth_v1.mp4 + group_analyse.txt)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 依赖关系

```
config.py ←── (被所有模块引用)
dataset.py ←── train_ewap.py, evaluate_ewap.py
model.py ←── train_ewap.py, evaluate_ewap.py, visualize_eth_on_video.py
prepare_pseudo_data.py ←── visualize_eth_on_video.py (运行时导入 compute_pseudo_labels)
trackers/* (上级目录) ←── evaluate_ewap.py, visualize_eth_on_video.py
metrics.py (上级目录) ←── evaluate_ewap.py
```

---

## 关键技术要点

1. **傅里叶位置编码**：通过随机高斯映射将低维坐标映射到高维空间，增强空间感知力
2. **多尺度 TransformerConv**：4层图注意力网络逐步聚合多跳邻域信息，每层使用残差连接
3. **自适应层融合**：学习各层的重要性权重，替代简单的拼接操作
4. **不确定性感知回归**：偏移预测同时输出方差，通过 NLL 损失约束，提供可靠的置信度估计
5. **物理规则伪标签**：无需人工标注，利用行人运动学规律（距离+速度方向+速率）自动生成群组标签
6. **动态阈值 + 运动约束**：推理时结合边分类概率与速度方向/速率约束，提高群组划分的鲁棒性
