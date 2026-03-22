# run_ablation — 消融实验模块

本目录用于对 HGATGT（Heterogeneous Graph Attention Group Target Tracker）模型进行**消融实验（Ablation Study）**，通过逐一移除模型的核心组件，验证每个组件对多目标跟踪性能的贡献。

---

## 目录结构

```
run_ablation/
├── ablation_model.py          # 可配置的消融模型定义
├── dataset.py                 # 数据集加载器重导出（实际逻辑在 sim_env/dataset.py）
├── train_ablation.py          # 消融变体训练脚本
├── run_comparison.py          # 全模型与消融变体的统一评估对比脚本（群组级）
├── run_point_comparison.py    # 全模型与消融变体的统一评估对比脚本（点级）
├── model/                     # 当前标准命名权重与部分历史遗留权重并存
│   ├── model_No_Fourier.pth
│   ├── model_model_No_Fourier_v2.pth
│   ├── model_No_Adaptive_Fusion.pth
│   ├── model_model_No_Adaptive_Fusion_v2.pth
│   ├── model_Plain_GCN.pth
│   └── model_Plain_GCN_v2.pth
├── output/result/             # 群组级 / 点级消融结果输出
│   ├── ablation_comparison_final.csv
│   ├── ablation_point_comparison.csv
│   └── 实验结果.md
└── README.md                  # 本文件
```

---

## 文件详细说明

### 1. `ablation_model.py` — 消融模型定义

该文件定义了三个类，构成消融实验的模型核心：

#### 1.1 `FourierFeatureEncoder`（傅里叶特征编码器）

| 属性 | 说明 |
|---|---|
| 作用 | 将低维位置坐标 (x, y) 通过随机傅里叶特征映射到高维空间，增强模型对空间位置的感知能力 |
| 输入维度 | 当前仿真消融实验中为 2（x, y 坐标） |
| 映射维度 | 实际使用时 `mapping_size=32`，输出为 `2 * 32 = 64` 维（sin 和 cos 各 32 维） |
| 缩放因子 | `scale=2.0`，控制随机高斯矩阵 B 的标准差 |
| 原理 | 利用随机傅里叶特征（Random Fourier Features）近似高斯核函数，将二维坐标投影到高维空间：`output = [sin(2π·x·B), cos(2π·x·B)]` |
| 关键实现 | 矩阵 B 通过 `register_buffer` 注册为非训练参数（固定不变），保证编码的一致性 |

#### 1.2 `AdaptiveLayerFusion`（自适应层融合模块）

| 属性 | 说明 |
|---|---|
| 作用 | 学习各 GNN 层输出的最优加权组合，而非简单取最后一层 |
| 参数 | `attn_vector`：形状为 `(num_layers, hidden_dim)` 的可训练参数 |
| 融合方式 | 对 `attn_vector` 取均值后通过 Softmax 得到每层权重 α_i，输出为 `Σ α_i · layer_i` |
| 层数 | 默认 `num_layers=4`，与 GNN 的 4 层卷积对应 |
| 优势 | 不同层捕获不同尺度的图结构信息：浅层关注局部邻域，深层捕获全局结构。自适应融合可以让模型自动选择最优的信息组合 |

#### 1.3 `AblationGNNTracker`（消融 GNN 跟踪器）

这是核心消融模型，通过三个开关参数控制组件的启用/禁用：

| 参数 | 默认值 | 功能 |
|---|---|---|
| `use_fourier` | `True` | 是否启用傅里叶特征编码 |
| `fusion_mode` | `'adaptive'` | 层融合方式：`'adaptive'`（自适应加权）或 `'last'`（仅用最后一层） |
| `use_transformer` | `True` | 是否使用 TransformerConv；`False` 则退化为 GCNConv |
| `hidden_dim` | 跟随 `config.HIDDEN_DIM`（当前 `sim_env/config.py` 为 64） | 隐层维度 |
| `input_node_dim` | `config.INPUT_DIM` (2) | 节点输入特征维度 |
| `input_edge_dim` | `config.EDGE_DIM` (3) | 边输入特征维度 |

**模型前向传播流程：**

```
输入 graph(x, edge_index, edge_attr)
    │
    ├─ [若 use_fourier=True] x（当前仿真输入即 2D [x, y]）
    │   → FourierFeatureEncoder → 64维
    │   → concat(x, fourier_features) → node_mlp → h (hidden_dim 维)
    │
    ├─ [若 use_fourier=False] x → node_mlp → h (hidden_dim 维)
    │
    ├─ edge_attr → edge_encoder → e (hidden_dim 维)
    │
    ├─ 4层 GNN 卷积（TransformerConv 或 GCNConv）
    │   每层: conv → BatchNorm → GELU → 残差连接
    │   收集每层输出 → layers[]
    │
    ├─ [若 fusion='adaptive'] AdaptiveLayerFusion(layers) → h_final
    ├─ [若 fusion='last'] layers[-1] → h_final
    │
    ├─ 边分类头: concat(h_final[src], h_final[dst], e) → MLP → Sigmoid → edge_scores
    │
    ├─ 群级偏移头: h_final → MLP → [group_offsets (2维),
    │                               group_uncertainty (2维, softplus + 1e-6)]
    │
    └─ 点级偏移头: h_final → MLP → [point_offsets (2维),
                                    point_uncertainty (2维, softplus + 1e-6)]
```

**输出：** `TrackerOutputs(edge_scores, group_offsets, group_uncertainty, point_offsets, point_uncertainty)`
- `edge_scores`：每条边属于同一目标群组的概率
- `group_offsets` / `group_uncertainty`：群级中心偏移预测及其不确定性
- `point_offsets` / `point_uncertainty`：点级 member 偏移预测及其不确定性

**关键细节：**
- TransformerConv 使用 4 个注意力头，每头输出 `hidden_dim // 4` 维
- GCNConv 不支持 `edge_attr` 参数，因此在 `use_transformer=False` 时会忽略边特征
- 空边情况兼容处理：当 `edge_attr` 的列数与期望不匹配时，自动创建正确维度的空张量
- 群组级评估脚本最终使用的是群级偏移分支；点级评估脚本使用 `head='point'` 的点级偏移分支

---

### 2. `dataset.py` — 数据集加载器

#### `RadarFileDataset`

`run_ablation/dataset.py` 本身只是一个薄包装，直接重导出 `sim_env.dataset.RadarFileDataset`；实际的建图逻辑在 `sim_env/dataset.py`。

| 属性 | 说明 |
|---|---|
| 基类 | `torch_geometric.data.Dataset` |
| 数据源 | `config.DATA_ROOT/{split}/` 目录下的 `.npy` 文件 |
| 数据组织 | 每个 `.npy` 文件为一个 episode（场景序列），包含多帧数据 |
| 建图半径 | `conn_radius = 30.0` |
| 节点特征 | 2D：`[x, y]` |
| 边特征 | 3D：`[rel_x, rel_y, dist]` |

**每帧数据处理流程：**

1. **节点特征构建**：直接使用测量点坐标 `meas`，构成 2D 节点特征 `[x, y]`
2. **图构建**：基于欧氏距离，在 `conn_radius=30.0` 内的点对之间建边（排除自环）
3. **边属性计算**：`edge_attr = [rel_x, rel_y, dist]`（3 维）
4. **边标签生成**：若两端节点属于同一目标（且标签非 0），则边标签为 1，否则为 0
5. **封装为 PyG Data 对象**：包含 `x`、`edge_index`、`edge_attr`、`edge_label`、`point_labels`、`gt_centers`，以及点级训练 / 评估会用到的 `gt_points`、`point_ids`、`has_gt_points`、`has_point_ids`

**返回值：** 一个 `graph_list`（每个 episode 中所有有效帧的图列表）

---

### 3. `train_ablation.py` — 消融变体训练脚本

该脚本用于训练各消融变体模型。

#### `run_experiment(name, model_config, skip_train=False)`

| 步骤 | 说明 |
|---|---|
| 数据加载 | 训练集最多使用前 500 个样本（加速消融实验），验证集使用全量 |
| 维度检测 | 自动从验证集（若为空则回退训练集）检测实际的 `input_node_dim` 和 `input_edge_dim`，避免硬编码不匹配 |
| 模型创建 | 使用 `AblationGNNTracker(**model_config)` |
| 优化器 | AdamW，学习率来自 `config.LEARNING_RATE`（当前为 `0.0005`），权重衰减 `1e-4` |
| 训练轮数 | 10 个 Epoch（对消融实验足够区分性能差异） |
| 梯度裁剪 | `max_norm=1.0` |
| 损失函数 | 复用 `sim_env.train_sim.compute_frame_loss` |
| 模型保存 | 验证集 loss 最优时保存至 `sim_env/run_ablation/model/model_{name}.pth` |
| NaN 保护 | 跳过 loss 为 NaN 或 Inf 的样本 |

**损失函数细节（来自 `sim_env.train_sim.compute_frame_loss`）：**

```
总损失 = loss_edge
       + config.LAMBDA_GROUP * loss_group
       + config.LAMBDA_POINT * loss_point
       + config.LAMBDA_TEMP  * loss_temp
```

- `loss_edge`：边分类 BCE 损失
- `loss_group`：群级 offset 的异方差回归损失
- `loss_point`：点级 offset 的异方差回归损失
- `loss_temp`：基于相邻帧点级校正结果的时序一致性损失
- 以上权重来自 `sim_env/config.py`（当前默认：`LAMBDA_GROUP=0.75`、`LAMBDA_POINT=1.25`、`LAMBDA_TEMP=0.2`）
- 当 `LAMBDA_POINT > 0` 或 `LAMBDA_TEMP > 0` 时，训练样本必须包含 `gt_points / point_ids`，否则 `compute_frame_loss` 会直接报错

**实验配置（在 `__main__` 中）：**

```python
experiments = {
    "No_Fourier": {"use_fourier": False, "fusion_mode": 'adaptive', "use_transformer": True},
    "No_Adaptive_Fusion": {"use_fourier": True, "fusion_mode": 'last', "use_transformer": True},
    "Plain_GCN": {"use_fourier": True, "fusion_mode": 'adaptive', "use_transformer": False},
}
```

---

### 4. `run_comparison.py` — 统一评估对比脚本

该脚本是消融实验的核心评估管线，将完整模型与所有消融变体在相同测试集上进行公平对比。

#### `evaluate_variant(name, model, weight_path, is_full_model=False)`

**评估流程：**

```
加载预训练权重
    │
    ├─ 遍历测试集中每个 episode
    │   ├─ 遍历每一帧图数据
    │   │   ├─ 模型推理 → TrackerOutputs(...)
    │   │   ├─ 群组级位置校正: corrected_pos = meas_points + group_offsets
    │   │   │   （脚本当前通过兼容写法读取第二个输出，本质上对应 group_offsets）
    │   │   │
    │   │   ├─ 聚类分组:
    │   │   │   ├─ [Full Model] 尝试连通分量法：
    │   │   │   │   edge_scores > 0.5 的边 → 稀疏邻接矩阵 → connected_components
    │   │   │   │   若失败则回退 DBSCAN
    │   │   │   │
    │   │   │   └─ [消融变体] DBSCAN(eps=30, min_samples=3)
    │   │   │
    │   │   ├─ 群组中心计算: 每个聚类的校正后位置均值
    │   │   ├─ 群组形状估计: 90%分位距（percentile 95 - percentile 5），最小 3.0
    │   │   │
    │   │   └─ 多目标跟踪:
    │   │       └─ GNNPostProcessor.update(det_centers, det_shapes)
    │   │           → 两阶段级联关联（匈牙利算法）
    │   │           → 6 维恒加速 Kalman 滤波
    │   │           → 轨迹管理（出生/死亡/ID 维护）
    │   │
    │   └─ 指标更新: metrics.update(gt_centers, gt_ids, pred_centers, pred_ids)
    │
    └─ 返回 metrics.compute()
```

**GNNPostProcessor 跟踪器细节：**

| 组件 | 说明 |
|---|---|
| 状态模型 | 6 维恒加速 Kalman 滤波器：状态 = [x, y, vx, vy, ax, ay] |
| 关联策略 | 两阶段级联匈牙利匹配 |
| 第一阶段 | 阈值 40.0，代价 = 距离 + 形状相似度（权重 w_dist=1.0, w_shape=20.0） |
| 第二阶段 | 阈值 90.0，仅匹配剩余的未关联轨迹和检测 |
| 轨迹出生 | 两阶段均未匹配的检测创建新轨迹 |
| 轨迹死亡 | 超过 15 帧未匹配的轨迹删除 |
| 老化处理 | 未匹配帧加速度衰减 ×0.5，速度衰减 ×0.9 |
| 形状更新 | 指数滑动平均：`0.5 * old + 0.5 * new` |

**评估入口（`__main__`）：**

```python
# 1. 评估完整模型（Full_Model）
#    使用 GNNGroupTracker
#    权重优先取 sim_env/config.py 的 MODEL_SAVE_PATH，若不存在则回退 MODEL_USE_PATH

# 2. 评估消融变体
#    脚本里显式使用 hidden_dim=64，并默认加载：
#    No_Fourier:           model_No_Fourier.pth
#    No_Adaptive_Fusion:   model_No_Adaptive_Fusion.pth
#    Plain_GCN:            model_Plain_GCN.pth
```

说明：完整模型的权重路径不是写死在本文件里，而是通过 `sim_env/config.py` 的 `MODEL_SAVE_PATH / MODEL_USE_PATH` 获取；完整模型的隐藏维度跟随当前模型 / 配置，`run_comparison.py` 里的消融分支则显式使用 `hidden_dim=64`。

#### 补充：`run_point_comparison.py` — 点级消融对比

该脚本用于补充**点级（point-level）**消融评估，复用 `sim_env/evaluate_single.py` 中已经验证过的点级后处理链路：

- 使用 `infer_corrected_points(..., head='point')`，即点偏移分支而不是群级偏移分支
- 先通过 `filter_clustered_points` 做 DBSCAN 噪声过滤，只保留非噪声点进入后续点跟踪
- 群组级轨迹仍由 `GNNPostProcessor` 维护
- 点 ID 通过 `GroupConstrainedPointAssociator` 在群组约束下完成关联
- 最终结果输出到 `sim_env/run_ablation/output/result/ablation_point_comparison.csv`
- 历史 legacy 单头 / 仅群级权重可能无法直接加载到这条新点级管线；脚本会给出 warning 并跳过对应 checkpoint

---

### 5. `model/` — 模型权重文件

| 文件 / 命名模式 | 对应变体 | 说明 |
|---|---|---|
| `model_No_Fourier.pth` | No_Fourier | 当前 `train_ablation.py` 的标准输出命名；`run_comparison.py` / `run_point_comparison.py` 默认读取 |
| `model_No_Adaptive_Fusion.pth` | No_Adaptive_Fusion | 当前 `train_ablation.py` 的标准输出命名；`run_comparison.py` / `run_point_comparison.py` 默认读取 |
| `model_Plain_GCN.pth` | Plain_GCN | 当前 `train_ablation.py` 的标准输出命名；`run_comparison.py` / `run_point_comparison.py` 默认读取 |
| `model_model_No_Fourier_v2.pth` | No_Fourier（legacy） | 历史遗留命名，保留在目录中供对照；不属于当前脚本默认保存路径 |
| `model_model_No_Adaptive_Fusion_v2.pth` | No_Adaptive_Fusion（legacy） | 历史遗留命名，保留在目录中供对照；不属于当前脚本默认保存路径 |
| `model_Plain_GCN_v2.pth` | Plain_GCN（legacy） | 历史遗留 `_v2` 权重；不属于当前脚本默认保存路径 |

补充说明：当前脚本默认读写的是不带 `_v2` 的 `model_{name}.pth`；目录中的 `_v2` / `model_model_*` 文件表示历史训练产物仍在保留，部分旧权重可能不兼容新的双 head 点级评估管线。

---

### 6. `output/result/` — 实验结果

#### `ablation_comparison_final.csv`

最终消融实验对比结果，包含以下指标：

| 指标 | 含义 | 理想方向 |
|---|---|---|
| **MOTA** | 多目标跟踪精度 = 1 - (漏检 + 误检 + ID切换) / 真值总数 | ↑ 越高越好 |
| **MOTP** | 多目标跟踪精度（匹配对的平均欧氏距离） | ↓ 越低越好 |
| **OSPA (Total)** | 最优子模式分配距离（综合定位 + 基数误差） | ↓ 越低越好 |
| **OSPA (Loc)** | OSPA 定位分量 | ↓ 越低越好 |
| **OSPA (Card)** | OSPA 基数分量（目标数量估计误差） | ↓ 越低越好 |
| **RMSE (Pos)** | 匹配目标中心位置的均方根误差 | ↓ 越低越好 |
| **IDSW** | ID 切换总数 | ↓ 越低越好 |
| **FAR** | 虚警率（每帧误检数） | ↓ 越低越好 |
| **Count Err** | 平均每帧目标计数误差 | ↓ 越低越好 |

#### `ablation_point_comparison.csv`

由 `run_point_comparison.py` 生成的点级消融对比结果。它对应的是 `head='point'` 的点偏移校正管线，并结合 DBSCAN 噪声过滤、`GNNPostProcessor` 群组跟踪和 `GroupConstrainedPointAssociator` 组约束点关联，便于重点比较 OSPA / RMSE / IDSW / MOTA 等点级指标。

#### 实验结果总览

| 模型变体 | MOTA ↑ | MOTP ↓ | OSPA (Total) ↓ | OSPA (Loc) ↓ | OSPA (Card) ↓ | RMSE ↓ | IDSW ↓ | FAR ↓ | Count Err ↓ |
|---|---|---|---|---|---|---|---|---|---|
| **Full_Model**（完整模型） | **0.7848** | **2.6126** | **12.5528** | **2.5597** | **10.9349** | **3.4858** | 444 | **0.0050** | **0.4992** |
| No_Fourier | 0.7684 | 2.8943 | 13.9698 | 2.8650 | 12.3415 | 3.9894 | **405** | 0.0070 | 0.5504 |
| No_Adaptive_Fusion | 0.7692 | 2.8227 | 13.8471 | 2.7973 | 12.2658 | 3.8976 | 410 | 0.0068 | 0.5472 |
| Plain_GCN | 0.7185 | 4.8486 | 17.6639 | 4.8033 | 15.2740 | 7.3363 | 499 | 0.0124 | 0.6618 |

---

## 消融实验设计分析

### 实验设计理念

消融实验遵循**控制变量法**，在完整模型的基础上，每次移除一个核心组件，观察性能变化幅度以量化该组件的贡献：

| 消融变体 | 移除的组件 | 保留的组件 | 验证的假设 |
|---|---|---|---|
| **No_Fourier** | 傅里叶位置编码 | TransformerConv + 自适应融合 | 傅里叶特征编码能否增强空间位置感知 |
| **No_Adaptive_Fusion** | 自适应层融合 | 傅里叶编码 + TransformerConv | 多层信息融合是否优于单层输出 |
| **Plain_GCN** | TransformerConv（替换为 GCNConv） | 傅里叶编码 + 自适应融合 | 注意力机制对图神经网络跟踪的重要性 |

### 结果解读

#### (1) TransformerConv vs GCNConv（最显著的性能差异）

Plain_GCN 变体性能下降最为显著：
- MOTA 从 0.7848 降至 0.7185（降幅 **8.4%**）
- MOTP 从 2.61 升至 4.85（定位精度恶化 **85.6%**）
- RMSE 从 3.49 升至 7.34（翻倍）
- OSPA 从 12.55 升至 17.66（增大 **40.7%**）

**结论：** TransformerConv 中的多头注意力机制是模型性能的核心支撑。GCNConv 无法利用边特征（边特征在 forward 中被忽略），且等权聚邻居信息的方式无法区分不同邻居的重要性，在密集多目标场景下严重制约了群组判别能力。

#### (2) 傅里叶位置编码的贡献

移除傅里叶编码后：
- MOTA 降低 **2.1%**（0.7848 → 0.7684）
- MOTP 升高 10.8%（2.61 → 2.89）
- RMSE 升高 14.5%（3.49 → 3.99）

**结论：** 傅里叶位置编码通过将低维坐标映射到高维空间，显著提升了模型对空间位置的细粒度感知能力，尤其改善了定位精度（MOTP、RMSE）。

#### (3) 自适应层融合的贡献

移除自适应融合后：
- MOTA 降低 **2.0%**（0.7848 → 0.7692）
- MOTP 升高 8.0%（2.61 → 2.82）
- OSPA (Card) 升高 12.2%（10.93 → 12.27）

**结论：** 自适应层融合通过加权组合多层 GNN 输出，有效整合了不同感受野范围的图结构信息。其对基数估计（OSPA Card）的改善尤为明显，表明多尺度信息融合有助于更准确地估计目标数量。

#### (4) IDSW（ID 切换）的异常

值得注意的是，No_Fourier（405）和 No_Adaptive_Fusion（410）的 IDSW 反而低于 Full_Model（444）。这可能是因为：
- 完整模型的更高检测灵敏度导致了更多的跟踪目标，从而增加了 ID 切换机会
- IDSW 与 MOTA 之间存在 trade-off：更高的 MOTA（更少漏检/误检）可能伴随略多的 ID 切换

### 各组件贡献排序（按对 MOTA 的影响）

```
TransformerConv 注意力机制 >> 傅里叶位置编码 ≈ 自适应层融合
    (ΔMota ≈ 8.4%)           (ΔMota ≈ 2.0%)  (ΔMota ≈ 2.0%)
```

---

## 运行方式

### 训练消融变体

```bash
# 推荐从仓库根目录运行
python -m sim_env.run_ablation.train_ablation

# 若已切到 sim_env/ 目录，也可运行
# python -m run_ablation.train_ablation
```

在 `train_ablation.py` 的 `__main__` 中修改 `experiments` 字典来选择要训练的变体。

### 运行评估对比

```bash
# 推荐从仓库根目录运行
python -m sim_env.run_ablation.run_comparison

# 若已切到 sim_env/ 目录，也可运行
# python -m run_ablation.run_comparison
```

结果将保存至 `sim_env/run_ablation/output/result/ablation_comparison_final.csv`。

### 运行点级评估对比

```bash
# 推荐从仓库根目录运行
python -m sim_env.run_ablation.run_point_comparison

# 若已切到 sim_env/ 目录，也可运行
# python -m run_ablation.run_point_comparison
```

结果将保存至 `sim_env/run_ablation/output/result/ablation_point_comparison.csv`。

补充说明：本目录部分脚本仍带有项目内相对路径假设，尤其 `run_comparison.py` 的结果 CSV 输出路径是相对写法；因此最好按上面的方式或直接从仓库根目录执行。

---

## 依赖关系

本模块依赖项目中的以下文件：
- `sim_env/config.py`：全局配置（数据路径、维度、设备、loss 权重、checkpoint 路径等）
- `sim_env/model.py`：完整模型 `GNNGroupTracker` 与 `TrackerOutputs` 定义
- `sim_env/train_sim.py`：`compute_frame_loss`，定义当前的 `edge + group + point + temp` 训练损失
- `sim_env/dataset.py`：实际 `RadarFileDataset` 构图逻辑（2D 节点、3D 边、`conn_radius=30.0`，并附带 `gt_points/point_ids` 等字段）
- `sim_env/run_ablation/dataset.py`：仅重导出 `sim_env.dataset.RadarFileDataset`
- `sim_env/evaluate_single.py`：点级评估中复用的 `infer_corrected_points`、`filter_clustered_points`、`GroupConstrainedPointAssociator` 等逻辑
- `metrics.py`：`TrackingMetrics` 评估指标
- `trackers/gnn_processor.py`：`GNNPostProcessor` 多目标跟踪后处理器
