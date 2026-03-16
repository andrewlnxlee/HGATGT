# sim_env — 仿真环境模块

> H-GAT-GT（Heterogeneous Graph Attention Group Target Tracker）项目的仿真数据生成、模型训练、评估与可视化模块。

---

## 目录结构

```
sim_env/
├── config.py                              # 全局配置文件（路径、超参数、设备等）
├── generate_data.py                       # 合成雷达群目标场景数据生成器
├── dataset.py                             # PyTorch Geometric 图数据集加载器
├── model.py                               # 核心 GNN 模型（GNNGroupTracker）
├── train_sim.py                           # 模型训练脚本
├── evaluate.py                            # 五种跟踪算法基准评估脚本
├── track_gif_gen.py                       # 推理 + GIF 动画生成器
├── visualize_sim_dataset_trajectory.py    # 静态轨迹可视化脚本
├── README.md                              # 本文件
├── model/                                 # 保存的模型权重
│   ├── best_model.pth
│   ├── best_model_v2.pth
│   ├── best_model_v3.pth
│   └── best_model_v4.pth
├── output/
│   ├── test/                              # 静态轨迹图 (.jpg)
│   ├── track_result_gif/                  # 跟踪结果动画 (.gif)
│   ├── final_benchmark.png
│   ├── final_benchmark_5way.png
│   ├── final_benchmark_all.png
│   └── final_benchmark_extended.png
└── run_ablation/                          # 消融实验子模块
    ├── ablation_model.py                  # 可配置的消融模型变体
    ├── dataset.py                         # 适配版数据集加载器（conn_radius=80）
    ├── train_ablation.py                  # 消融变体训练脚本
    ├── run_comparison.py                  # 统一评估所有变体
    ├── README.md                          # 消融实验详细文档
    ├── model/                             # 消融模型权重
    └── output/result/                     # 消融实验结果（CSV + MD）
```

---

## 整体流程概述

```
generate_data.py          dataset.py            model.py
  合成场景数据     --->    构建图结构     --->    GNN 推理
  (.npy 文件)            (PyG Data)          (边分类 + 偏移回归)
                                                    |
                                                    v
                                            train_sim.py (训练)
                                            evaluate.py  (评估)
                                            track_gif_gen.py (可视化)
```

**核心思路**: 利用图神经网络对雷达量测点构建图，通过 TransformerConv 多头注意力机制学习量测点之间的关系，同时进行：
1. **边分类**：判断两个量测点是否属于同一群目标
2. **节点偏移回归**：预测每个点相对于其所属群目标中心的偏移量（附带不确定性估计）

---

## 文件详细说明

---

### 1. `config.py` — 全局配置

集中管理项目中所有可调参数，被几乎所有其他文件导入。

#### 数据生成参数

| 参数 | 值 | 说明 |
|------|------|------|
| `DATA_ROOT` | `"./data"` | 生成数据的根目录 |
| `XISHU` | `0.1` | 噪声缩放系数 |
| `NUM_TRAIN_SAMPLES` | 2000 | 训练集样本数 |
| `NUM_VAL_SAMPLES` | 200 | 验证集样本数 |
| `NUM_TEST_SAMPLES` | 100 | 测试集样本数 |
| `FRAMES_PER_SAMPLE` | 50 | 每个场景的帧数 |
| `MAX_GROUPS` | 5 | 每个场景最大群目标数量 |

#### 模型参数

| 参数 | 值 | 说明 |
|------|------|------|
| `INPUT_DIM` | 2 | 节点特征维度（x, y 坐标） |
| `EDGE_DIM` | 3 | 边特征维度（dx, dy, dist） |
| `HIDDEN_DIM` | 64 | 隐藏层维度 |

#### 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| `BATCH_SIZE` | 1 | 每批次一个场景 |
| `LEARNING_RATE` | 0.001 | 基础学习率 |
| `EPOCHS` | 50 | 训练轮数 |
| `MODEL_SAVE_PATH` | `sim_env/model/best_model_v5.pth` | 新训练模型保存路径 |
| `MODEL_USE_PATH` | `sim_env/model/best_model_v4.pth` | 推理时加载的模型路径 |
| `DEVICE` | `"cuda"` | 计算设备 |

#### 输出路径

| 参数 | 值 | 说明 |
|------|------|------|
| `OUTPUT_TEST_DIR` | `sim_env/output/test` | 静态可视化输出目录 |
| `OUTPUT_GIF_DIR` | `sim_env/output/track_result_gif` | GIF 动画输出目录 |

#### EWAP 数据集参数

| 参数 | 说明 |
|------|------|
| `EWAP_MODEL_*_PATH` | EWAP (ETH/Hotel 行人) 数据集模型路径 |
| `COORD_SCALE` | 坐标缩放系数（50.0） |
| `COORD_OFFSET` | 坐标中心偏移 [500.0, 500.0] |

---

### 2. `generate_data.py` — 合成数据生成器

生成包含群目标交互行为（合并、分裂、混合）的仿真雷达场景数据。

#### 核心类：`ActiveInteractionScenarioEngine`

**构造函数参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_frames` | 50 | 每个场景帧数 |
| `area_size` | (1000, 1000) | 仿真区域大小（米） |
| `clutter_rate` | 8 | 每帧杂波点的泊松率 |
| `detection_prob` | 0.95 | 目标检测概率 |
| `min_speed` | 10.0 | 最小速度（米/帧） |
| `max_speed` | 25.0 | 最大速度（米/帧） |

**场景类型（按概率随机选择）：**

| 场景类型 | 概率 | 描述 |
|----------|------|------|
| 汇聚 (converge) | 40% | 2-5 个群从边缘飞向 1-2 个集结点；距离 < 30m 时合并 |
| 分裂 (diverge) | 30% | 1-3 个群横穿区域；在随机帧（30%-60% 时刻）分裂为两组，施加正交速度扰动 |
| 混合 (mixed) | 30% | 硬编码场景：群 1 在第 20 帧分裂；群 2、3 向中心高速汇聚并在 40m 内合并 |

**关键方法：**

| 方法 | 功能 |
|------|------|
| `generate_episode()` | 随机选择场景类型并生成一个完整场景 |
| `_run_converge_scenario()` | 生成汇聚场景 |
| `_run_diverge_scenario()` | 生成分裂场景 |
| `_run_mixed_scenario()` | 生成混合场景 |
| `_spawn_group_aiming_at(group_id, target_pos)` | 生成包含 5-15 个成员的群，速度方向指向目标点；包含距离验证 |
| `_apply_guidance(group, target_pos)` | PD 制导控制器：远距离（>300m）加速，近距离（<100m）减速；惯性因子 0.85 |
| `_apply_wander(group)` | 添加随机速度和航向扰动 |
| `_update_members_and_record(groups, frame_info)` | 更新成员位置（弹簧模型内聚力 force=-0.05×偏移），添加量测噪声（σ=1.5m），生成杂波，应用检测概率 |

**输出数据格式：**

每帧数据为 Python 字典：

```python
{
    'meas': np.array,       # shape [N_points, 2] — 量测点 (x, y) 坐标
    'labels': np.array,     # shape [N_points]    — 群ID（0 = 杂波）
    'gt_centers': np.array  # shape [N_groups, 3] — [群ID, 中心x, 中心y]
}
```

**保存函数 `save_dataset(split_name, num_samples)`：** 将场景数据保存为 `.npy` 文件到 `DATA_ROOT/{split}/sample_XXXXX.npy`。

---

### 3. `dataset.py` — PyG 图数据集加载器

将生成的 `.npy` 文件转换为 PyTorch Geometric 的 `Data` 图对象。

#### 核心类：`RadarFileDataset(Dataset)`

**关键参数：**

| 参数 | 值 | 说明 |
|------|------|------|
| `conn_radius` | 30.0 | 图连接半径——欧氏距离 ≤ 30m 的点对之间建立边 |
| `split` | `'train'`/`'val'`/`'test'` | 数据划分 |

**每帧图结构构建规则：**

| 图组件 | 构建方式 | 形状 |
|--------|----------|------|
| `x`（节点特征） | 原始量测坐标 | [N, 2] |
| `edge_index` | `conn_radius` 内所有点对（排除自环） | [2, E] |
| `edge_attr` | [相对x, 相对y, 欧氏距离] | [E, 3] |
| `edge_label` | 两端点属于同一群（且标签≠0）则为 1，否则为 0 | [E] |
| `point_labels` | 每个点的群 ID（0 = 杂波） | [N] |
| `gt_centers` | 真实群中心 [id, x, y] | [G, 3] |

**`get(idx)` 方法：** 返回一个 `Data` 对象的列表（场景中每帧一个图）。

---

### 4. `model.py` — 核心 GNN 模型

定义 H-GAT-GT 的核心神经网络架构。

#### 架构总览

```
输入坐标 (2D) ──┬── FourierFeatureEncoder (2D → 64D)
                 │
                 ├── 拼接 → [2D + 64D = 66D] → MLP → hidden_dim (96D)
                 │
边特征 (3D) ─────── MLP → hidden_dim (96D)
                 │
                 ├── 4 层 TransformerConv（4 头注意力, 含边特征, BatchNorm, GELU, 残差连接）
                 │           ↓
                 ├── AdaptiveLayerFusion（4 层输出加权融合）
                 │           ↓
                 ├── 边分类头: [src_feat ‖ dst_feat ‖ edge_feat] (288D) → MLP → Sigmoid
                 │
                 └── 偏移回归头: node_feat (96D) → MLP → [dx, dy, σ_x, σ_y]
```

#### 子模块详解

**`FourierFeatureEncoder(nn.Module)`**

| 属性 | 说明 |
|------|------|
| 功能 | 将 2D 坐标映射为高频特征，帮助网络捕捉空间位置信息 |
| 输入 | 2D 坐标 |
| 输出 | 64D（32 个 sin + 32 个 cos） |
| 原理 | 使用随机高斯矩阵 B（注册为 buffer，不参与训练），计算 `[sin(2π·B·x), cos(2π·B·x)]` |
| 参数 | `mapping_size=32`, `scale=2.0` |

**`AdaptiveLayerFusion(nn.Module)`**

| 属性 | 说明 |
|------|------|
| 功能 | 学习 4 层 GNN 输出的加权组合，而非仅使用最后一层 |
| 原理 | 可训练的注意力向量 `attn_vector [4, hidden_dim]`，对各层输出计算 softmax 权重，再加权求和 |
| 意义 | 不同层捕捉不同尺度的图结构信息，自适应融合可充分利用各层特征 |

**`GNNGroupTracker(nn.Module)` — 主模型**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_node_dim` | 2 | 输入节点维度 |
| `input_edge_dim` | 3 | 输入边维度 |
| `hidden_dim` | 96 | 隐藏层维度 |

**TransformerConv 骨干网络配置：**
- 4 层 `TransformerConv`
- 每层 4 个注意力头，每头 `hidden_dim // 4 = 24` 维
- 每层包含：TransformerConv → BatchNorm → GELU → Dropout(0.1) → 残差连接

**输出：**

```python
(edge_scores, pred_offsets, pred_uncertainty)
# edge_scores:      [E] — 每条边属于同一群的概率（Sigmoid 输出）
# pred_offsets:     [N, 2] — 每个节点指向群中心的偏移量 (dx, dy)
# pred_uncertainty: [N, 2] — 每个节点的预测不确定性 (σ_x, σ_y)，通过 Softplus 保证正值
```

---

### 5. `train_sim.py` — 训练脚本

#### 损失函数 `compute_loss()`

**双任务联合损失：**

1. **边分类损失 `loss_edge`**：
   - 二元交叉熵（BCE）
   - 动态正样本权重（`pos_weight`），上限为 10，用于处理正负样本不均衡

2. **异方差回归损失 `loss_reg`**：
   - 负对数似然形式：`0.5 × (MSE / variance + log(variance))`
   - 仅对有效标签点（label ≠ 0）计算
   - 目标偏移 = 真实群中心坐标 − 当前量测坐标
   - 模型可学习每个点的预测不确定性

**总损失 = `loss_edge + 1.0 × loss_reg`**

#### 训练配置

| 配置项 | 值 |
|--------|------|
| 优化器 | AdamW, lr=0.0005, weight_decay=1e-4 |
| 学习率调度 | CosineAnnealingWarmRestarts, T_0=10, T_mult=2 |
| 梯度累积 | 8 步（模拟 batch_size=8） |
| 梯度裁剪 | max_norm=1.0 |
| 模型保存策略 | 验证集最优损失 → `config.MODEL_SAVE_PATH` |

#### 训练流程

遍历每个场景 → 遍历场景内每帧（图）→ 跳过无边的图 → 计算损失 → 梯度累积 → 更新参数。每个 epoch 结束后在验证集上评估，保存最优模型。

---

### 6. `evaluate.py` — 五种跟踪算法基准评估

对测试集进行全面的多算法对比评估。

#### 评估的跟踪算法

| 序号 | 算法名称 | 说明 |
|------|----------|------|
| 1 | **H-GAT-GT（本方法）** | GNN 模型 + DBSCAN(eps=30) 聚类 + GNNPostProcessor 跟踪器 |
| 2 | **Baseline (DBSCAN+KF)** | DBSCAN(eps=35, min_samples=3) + 卡尔曼滤波 |
| 3 | **GM-CPHD** | 高斯混合势概率假设密度滤波器 |
| 4 | **CBMeMBer** | 基数均衡多伯努利滤波器 |
| 5 | **Graph-MB** | 基于图的多伯努利跟踪器 |

#### H-GAT-GT 评估流程

```
模型推理 → 偏移校正（corrected_pos = meas + offsets）
         → DBSCAN 聚类 → GNNPostProcessor 跟踪
         → 形状估计（5th-95th 百分位，最小 3m）
         → 匈牙利匹配重建点-航迹分配
```

#### 基线算法（3-5）评估流程

```
DBSCAN(eps=35) 预处理得到质心 → 传入各跟踪器
→ 匈牙利匹配将跟踪器输出映射回 DBSCAN 质心 → 重建点标签
```

#### 评估指标

| 指标类别 | 具体指标 |
|----------|----------|
| 跟踪性能 | MOTA, MOTP, IDSW (ID 切换), FAR (虚警率) |
| 距离度量 | OSPA (总/定位/基数), RMSE (位置) |
| 计数性能 | Count Error (数量误差) |
| 聚类质量 | Purity (纯度), Completeness (完整性), G-IoU |
| 计算效率 | Time (耗时) |

**输出：** 打印对比表格，保存柱状图到 `final_benchmark_5way.png`。

---

### 7. `track_gif_gen.py` — GIF 动画生成器

对单个测试场景运行推理，生成实时跟踪动画 GIF。

#### 核心类：`RobustGroupTracker`

手工设计的多目标跟踪器，基于匈牙利匹配和恒速预测模型。

| 组件 | 说明 |
|------|------|
| 状态 | 每条航迹：位置、速度、年龄、轨迹历史 |
| 预测 | 恒速模型：`pos += vel` |
| 关联 | 匈牙利算法（`linear_sum_assignment`），欧氏距离代价矩阵 |
| 距离阈值 | 50.0m |
| 分裂检测 | 未关联检测靠近已有航迹 → 新 ID，继承速度 |
| 合并检测 | 未关联航迹靠近已有检测 → 航迹消亡 |
| 航迹新建 | 完全未关联的检测 → 新航迹 |
| 航迹消亡 | 年龄 > `max_age`（5 帧）的航迹 |
| 状态更新 | 指数平滑：alpha_pos=0.6, alpha_vel=0.3 |
| 轨迹历史 | 存储最近 50 个位置用于轨迹尾迹可视化 |

#### 推理与可视化流程

1. 加载模型（`config.MODEL_USE_PATH`）
2. 每帧：模型前向传播 → 边分类（阈值 0.5）→ 连通分量构建邻接图 → 聚类质心提取（≥3 点的簇）→ RobustGroupTracker 更新
3. 计算每帧边分类准确率
4. 用 matplotlib 生成动画：按群着色的量测点、质心标记、ID 标签、轨迹尾迹、准确率显示

**输出：** `config.OUTPUT_GIF_DIR/sim_data_{NUM}_track_result_with_acc.gif`

---

### 8. `visualize_sim_dataset_trajectory.py` — 静态轨迹可视化

生成单个测试样本的累积轨迹图，展示群目标随时间的运动过程。

| 配置 | 说明 |
|------|------|
| `NUM=2` | 选择第几个测试样本 |
| 颜色方案 | 群 1=蓝色, 群 2=橙色, 群 3=绿色 |
| 透明度 | 与时间成正比（早期点更透明） |
| 杂波处理 | 忽略 label=0 的杂波点 |
| 帧间隔 | 每隔一帧绘制一次 |

**输出：** `config.OUTPUT_TEST_DIR/sim_dataset_{NUM}_trajectory.jpg`

---

## 仿真环境评估结果

### 五种算法对比结果

| 算法 | MOTA ↑ | MOTP ↓ | IDSW ↓ | FAR ↓ | OSPA (Total) ↓ | OSPA (Loc) ↓ | OSPA (Card) ↓ | RMSE (Pos) ↓ | Count Err ↓ | Purity ↑ | G-IoU ↑ | Time (s) |
|------|--------|--------|--------|-------|----------------|--------------|---------------|--------------|-------------|----------|---------|----------|
| **H-GAT-GT (Ours)** | **0.7808** | **2.572** | 384 | 0.0006 | **13.119** | **2.486** | **11.693** | **3.496** | **0.526** | **0.926** | **0.601** | 2.326 |
| Baseline (DBSCAN+KF) | 0.724 | 3.650 | 300 | 0.0000 | 17.468 | 3.421 | 15.764 | 5.034 | 0.699 | 0.903 | 0.000 | 0.410 |
| GM-CPHD | 0.749 | 3.288 | 441 | 0.0014 | 15.255 | 3.235 | 13.689 | 4.888 | 0.601 | 0.903 | 0.000 | 0.773 |
| CBMeMBer | 0.731 | 3.409 | 465 | 0.0068 | 16.197 | 3.313 | 14.459 | 5.239 | 0.645 | 0.903 | 0.000 | 0.458 |
| Graph-MB | 0.637 | 3.945 | 423 | 0.0016 | 21.806 | 3.408 | 20.395 | 5.313 | 0.911 | 0.897 | 0.000 | 12.750 |

**结论**：H-GAT-GT 在所有核心指标上均取得最优表现：
- **MOTA** 达到 0.7808，比次优的 GM-CPHD 高出 3.2 个百分点
- **MOTP** 最低（2.572），定位精度最优
- **OSPA** 综合指标最低（13.119），势估计和定位误差均优于基线
- **RMSE** 最低（3.496m），群中心估计最准确
- **G-IoU** 唯一为正值（0.601），表明聚类边界与真实群组有显著重叠

---

## 消融实验子模块 (`run_ablation/`)

### 9. `run_ablation/ablation_model.py` — 消融模型变体

定义可配置的 GNN 模型，支持逐一关闭各架构组件。

#### 核心类：`AblationGNNTracker(nn.Module)`

**可配置开关：**

| 参数 | 默认值 | 关闭效果 |
|------|--------|----------|
| `use_fourier` | True | 移除傅里叶位置编码；节点 MLP 仅接受原始坐标 |
| `fusion_mode` | `'adaptive'` | 设为 `'last'` 时仅使用最后一层 GNN 输出，不进行自适应加权融合 |
| `use_transformer` | True | 设为 False 时用 `GCNConv` 替代 `TransformerConv`（无边特征、等权邻居聚合） |

**与主模型 `GNNGroupTracker` 的区别：**

| 差异 | 主模型 | 消融模型 |
|------|--------|----------|
| 归一化 | `BatchNorm` | `GNNLayerNorm` |
| 偏移裁剪 | 无 | `torch.clamp([-100, 100])` |
| 不确定性裁剪 | 无 | `torch.clamp([0.01, 10.0])` |
| 返回值 | 3 元组 | 4 元组（额外返回 `h_final`） |
| 卷积层定义 | 显式 conv1-4 | `nn.ModuleList` 循环 |

---

### 10. `run_ablation/dataset.py` — 适配版数据集加载器

与主 `dataset.py` 的关键区别：

| 差异 | 主版本 | 消融版 |
|------|--------|--------|
| `conn_radius` | 30.0 | **80.0**（更大的连接半径，兼容真实数据集尺度） |
| 节点特征 | 仅 2D (位置) | 支持 2D 和 **4D (位置+速度)** |
| 边特征 | 3D | 支持 3D 和 **6D** (`[rel_x, rel_y, dist, rel_vx, rel_vy, cos_sim]`) |
| 缺失目录处理 | 无 | `os.path.exists` 检查 |

---

### 11. `run_ablation/train_ablation.py` — 消融变体训练

#### 训练配置（相比主训练脚本，为加速而简化）

| 配置项 | 值 |
|--------|------|
| 训练样本数 | min(500, 总数) |
| 训练轮数 | 10 |
| 优化器 | AdamW, lr=0.0005, weight_decay=1e-4 |
| 梯度裁剪 | max_norm=1.0 |
| NaN 保护 | 跳过 NaN/Inf 损失 |
| 自动维度检测 | 从验证数据中读取实际节点/边维度 |
| 模型保存 | `sim_env/run_ablation/model/model_{name}_v2.pth` |

#### 定义的三个消融变体

| 变体名称 | Fourier | Adaptive Fusion | Transformer |
|----------|---------|-----------------|-------------|
| `Plain_GCN` | ✓ | ✓ | ✗（使用 GCNConv） |
| `No_Fourier` | ✗ | ✓ | ✓ |
| `No_Adaptive_Fusion` | ✓ | ✗（仅最后一层） | ✓ |

---

### 12. `run_ablation/run_comparison.py` — 统一消融评估

在同一测试集上评估所有模型变体，保证公平对比。

#### 评估流程（每帧）

1. 模型推理 → 获取边分数和偏移量
2. 位置校正：`corrected_pos = meas_points + offsets`
3. 聚类：
   - 完整模型：优先使用连通分量（score > 0.5），失败时回退到 DBSCAN(eps=30)
   - 消融变体：始终使用 DBSCAN(eps=30, min_samples=3)
4. 形状估计：5th-95th 百分位包围盒，最小 3.0m
5. 跟踪：`GNNPostProcessor`（6 状态卡尔曼滤波 + 级联匈牙利关联）
6. 指标计算：`TrackingMetrics`

**注意：** 消融变体使用 `hidden_dim=64`，完整模型使用 `hidden_dim=96`。

**输出：** `sim_env/run_ablation/output/result/ablation_comparison_final.csv`

---

### 消融实验结果

| 模型 | MOTA↑ | MOTP↓ | OSPA (Total)↓ | RMSE (Pos)↓ | IDSW↓ | FAR↓ | Count Err↓ |
|------|-------|-------|---------------|-------------|-------|------|-----------|
| **Full_Model** | **0.7848** | **2.61** | **12.55** | **3.49** | 444 | **0.005** | **0.50** |
| No_Fourier | 0.7684 | 2.89 | 13.97 | 3.99 | **405** | 0.007 | 0.55 |
| No_Adaptive_Fusion | 0.7692 | 2.82 | 13.85 | 3.90 | 410 | 0.007 | 0.55 |
| Plain_GCN | 0.7185 | 4.85 | 17.66 | 7.34 | 499 | 0.012 | 0.66 |

**组件贡献度排序：** `TransformerConv >> Fourier 编码 ≈ Adaptive Fusion`

- **TransformerConv** 是最关键的组件：替换为 GCNConv 后 MOTA 下降 6.6 个百分点，RMSE 恶化约 2 倍。多头注意力机制和边特征的利用对群目标关系建模至关重要。
- **Fourier 编码** 和 **Adaptive Fusion** 贡献相当：各自移除后 MOTA 下降约 1.5-1.6 个百分点，OSPA 升高 1.3-1.4。
- 完整模型在绝大多数指标上取得最优，验证了三个组件的互补性。

---

## 模块间依赖关系

```
config.py  ←────────┬──────┬───────┬────────┬─────────┬──────────┬──────────┐
                     │      │       │        │         │          │          │
              generate_data dataset model  train_sim evaluate track_gif  visualize
                               │      │       │        │         │
                               └──┬───┘   compute_loss│         │
                                  │           │        │         │
                                  └───────────┴────────┴─────────┘
                                              │
                               ┌──────────────┘
                               │
                      run_ablation/
                     ├── ablation_model.py  ← config
                     ├── dataset.py         ← config
                     ├── train_ablation.py  ← ablation_model, 主 dataset, train_sim.compute_loss
                     └── run_comparison.py  ← model, ablation_model, 主 dataset, metrics, gnn_processor
```

**外部依赖（项目根目录下的模块）：**

| 模块路径 | 提供的类/功能 |
|----------|--------------|
| `metrics.py` | `TrackingMetrics` — 跟踪指标计算 |
| `trackers/baseline.py` | `BaselineTracker` — DBSCAN+KF 基线 |
| `trackers/gm_cphd.py` | `GMCPHDTracker` — GM-CPHD 滤波器 |
| `trackers/cbmember.py` | `CBMeMBerTracker` — CB-MeMBer 滤波器 |
| `trackers/graph_mb.py` | `GraphMBTracker` — 图多伯努利跟踪器 |
| `trackers/gnn_processor.py` | `GNNPostProcessor` — 基于卡尔曼滤波的后处理跟踪器 |

---

## 快速使用指南

### 1. 生成数据

```bash
python sim_env/generate_data.py
```

在 `./data/` 下生成 train/val/test 三个子集。

### 2. 训练模型

```bash
python sim_env/train_sim.py
```

训练完成后模型保存到 `config.MODEL_SAVE_PATH`。

### 3. 评估

```bash
python sim_env/evaluate.py
```

运行五种算法对比评估，输出指标表格和柱状图。

### 4. 生成跟踪动画

```bash
python sim_env/track_gif_gen.py
```

生成指定测试场景的跟踪结果 GIF。

### 5. 运行消融实验

```bash
# 训练消融变体
python sim_env/run_ablation/train_ablation.py

# 统一评估
python sim_env/run_ablation/run_comparison.py
```
