# H-GAT-GT: Hierarchical Graph Attention Group Target Tracker

基于层次化图注意力网络的群目标跟踪系统。本项目提出了一种基于 **TransformerConv 图神经网络** 的多群目标跟踪方法（H-GAT-GT），通过边分类与偏移回归双任务联合学习，实现对雷达仿真场景和真实行人轨迹数据（EWAP/ETH 数据集）中群目标的检测与跟踪。

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [模型架构](#模型架构)
- [环境依赖](#环境依赖)
- [快速开始](#快速开始)
  - [仿真环境](#仿真环境-sim_env)
  - [EWAP 真实数据环境](#ewap-真实数据环境-ewap_env)
- [数据流水线](#数据流水线)
- [训练细节](#训练细节)
- [评估与基线对比](#评估与基线对比)
- [消融实验](#消融实验)
- [可视化](#可视化)
- [评估指标](#评估指标)
- [实验结果](#实验结果)

---

## 项目概述

H-GAT-GT 是一个面向**多群目标跟踪**（Multi-Group Target Tracking, MGTT）的深度学习框架。系统将每一帧的量测点构建为图结构，利用图注意力网络进行特征提取，并通过**双头解码器**同时完成：

1. **边分类（Edge Classification）**：判断两个量测点是否属于同一群组
2. **偏移回归（Offset Regression）**：预测每个量测点到其群中心的偏移量，并附带不确定性估计

系统包含两个独立的实验环境：
- **仿真环境（sim_env）**：使用合成雷达群目标数据，包含汇聚、分裂、混合三种典型场景
- **EWAP 环境（ewap_env）**：使用真实的 ETH Walking Pedestrians 行人轨迹数据集

---

## 核心特性

- **傅里叶位置编码（Fourier Feature Encoding）**：通过随机高斯映射将低维坐标投射到高维特征空间，增强模型的空间感知能力
- **4 层 TransformerConv 骨干网络**：利用边特征感知的多头注意力机制（4 heads），支持边属性参与注意力计算
- **自适应层融合（Adaptive Layer Fusion）**：学习每层输出的重要性权重，通过 Softmax 加权融合替代简单拼接
- **残差连接（Residual Connection）**：每层图卷积后添加残差连接，缓解深层网络的梯度消失问题
- **异方差不确定性估计（Heteroscedastic Uncertainty）**：偏移回归头同时预测均值和方差，使用 NLL 损失实现自适应的不确定性建模
- **全面的基线对比**：提供 6 种跟踪算法实现，涵盖经典随机有限集方法、图网络方法和学习基方法

---

## 项目结构

```
HGATGT/
├── README.md                           # 本文件
├── metrics.py                          # 通用跟踪评估指标 (MOTA, MOTP, OSPA, G-IoU 等)
├── .gitignore
│
├── sim_env/                            # ===== 仿真雷达环境 =====
│   ├── config.py                       # 仿真环境配置（数据、模型、训练参数）
│   ├── model.py                        # GNNGroupTracker 模型定义 (2D 输入)
│   ├── dataset.py                      # RadarFileDataset 数据集类
│   ├── generate_data.py                # 合成数据生成器 (汇聚/分裂/混合场景)
│   ├── train_sim.py                    # 仿真数据训练脚本
│   ├── evaluate.py                     # 5 种算法基准评估
│   ├── track_gif_gen.py                # 跟踪结果 GIF 动画生成
│   ├── visualize_sim_dataset_trajectory.py  # 静态轨迹可视化
│   ├── model/                          # 训练权重存放
│   │   ├── best_model_v4.pth           # 当前使用的最优模型
│   │   └── ...
│   ├── output/                         # 输出目录（图表、GIF 等）
│   └── run_ablation/                   # ===== 消融实验子模块 =====
│       ├── ablation_model.py           # 可配置消融变体模型
│       ├── train_ablation.py           # 消融实验训练脚本
│       ├── run_comparison.py           # 消融结果对比脚本
│       └── output/                     # 消融实验结果
│
├── ewap_env/                           # ===== EWAP 真实数据环境 =====
│   ├── config.py                       # EWAP 环境配置（4D 输入特征）
│   ├── model.py                        # GNNGroupTracker 模型定义 (4D 输入, 6D 边特征)
│   ├── dataset.py                      # RadarFileDataset (含速度特征)
│   ├── prepare_ewap.py                 # EWAP 原始数据预处理与格式转换
│   ├── prepare_pseudo_data.py          # 伪标签生成（基于物理规则的行人同伴检测）
│   ├── train_ewap.py                   # EWAP 数据训练脚本
│   ├── evaluate_ewap.py               # 6 种算法基准评估
│   ├── visualize_eth_on_video.py       # ETH 视频叠加可视化
│   ├── model/                          # 训练权重存放
│   │   └── best_model_ewap_v1.pth     # 当前使用的最优模型
│   └── output/                         # 输出目录（对比结果、视频等）
│       └── result/
│           ├── EWAP_compare.txt        # 算法对比结果
│           └── group_analyse.txt       # 群分析结果
│
├── trackers/                           # ===== 基线跟踪算法实现 =====
│   ├── baseline.py                     # DBSCAN + 卡尔曼滤波 基线
│   ├── gm_cphd.py                      # 高斯混合 CPHD 滤波器
│   ├── gm_phd.py                       # 高斯混合 PHD 滤波器
│   ├── cbmember.py                     # CB-MeMBer 滤波器
│   ├── graph_mb.py                     # 图辅助多伯努利跟踪器 (UKF)
│   ├── gnn_processor.py               # GNN 后处理器（级联匈牙利关联）
│   ├── social_stgcnn_tracker.py        # Social-STGCNN 跟踪器
│   └── kalman_box.py                   # 卡尔曼滤波辅助类
│
├── datasets/                           # ===== 原始数据集 =====
│   └── ewap_dataset/
│       ├── seq_eth/                    # ETH 场景 (obsmat, 视频, 地图, 单应矩阵)
│       └── seq_hotel/                  # Hotel 场景
│
├── Social-STGCNN-master/              # ===== 第三方 Social-STGCNN 模型 =====
│   ├── model.py                        # STGCNN 模型定义
│   ├── train.py, test.py               # 训练与测试
│   ├── checkpoint/                     # 预训练权重 (eth, hotel, univ, zara1, zara2)
│   └── datasets/                       # ETH/UCY 数据集
│
└── temp/                               # 临时文件
```

---

## 模型架构

### GNNGroupTracker

模型采用四阶段架构设计：

```
输入数据 (量测点坐标 + 边关系)
        │
        ▼
┌──────────────────────────────┐
│  Stage 1: 输入编码            │
│  ┌────────────────────────┐  │
│  │ 傅里叶位置编码 (FFE)     │  │   [x, y] → Random Gaussian → [sin, cos] → 64D
│  │ + 原始坐标拼接           │  │   [x, y, FFE(x,y)] → MLP → 96D 节点特征
│  └────────────────────────┘  │
│  ┌────────────────────────┐  │
│  │ 边特征编码               │  │   [dx, dy, dist] → MLP → 96D 边特征
│  └────────────────────────┘  │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 2: TransformerConv    │
│  骨干网络 (4 层)              │
│                              │
│  每层结构:                    │
│  TransformerConv(96→24×4heads)│
│  → BatchNorm → GELU          │
│  → 残差连接 (h_out + h_in)   │
│                              │
│  输出: [h1, h2, h3, h4]     │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 3: 自适应层融合        │
│                              │
│  α = Softmax(mean(W))        │
│  h_final = Σ αᵢ · hᵢ        │
└──────────────────────────────┘
        │
        ▼
┌──────────────────────────────┐
│  Stage 4: 双头解码器          │
│                              │
│  ┌──────────────┐            │
│  │ 边分类头      │            │   [h_src ∥ h_dst ∥ e] → MLP → Sigmoid → P(同群)
│  └──────────────┘            │
│  ┌──────────────┐            │
│  │ 偏移回归头    │            │   h_final → MLP → [dx, dy, σx, σy]
│  └──────────────┘            │
└──────────────────────────────┘
```

### 模型参数

| 参数 | 仿真环境 (sim_env) | EWAP 环境 (ewap_env) |
|------|-------------------|---------------------|
| 节点输入维度 | 2 (x, y) | 4 (x, y, vx, vy) |
| 边输入维度 | 3 (dx, dy, dist) | 6 (dx, dy, dist, dvx, dvy, cos_sim) |
| 隐藏维度 | 96 | 96 |
| 注意力头数 | 4 | 4 |
| 骨干层数 | 4 | 4 |
| 傅里叶映射维度 | 64 (mapping_size=32, scale=2.0) | 64 (mapping_size=32, scale=2.0) |

---

## 环境依赖

### 核心依赖

```
Python >= 3.8
PyTorch >= 1.12
PyTorch Geometric >= 2.0
```

### Python 包

```
# 核心框架
torch
torch_geometric

# 科学计算
numpy
scipy
scikit-learn

# 可视化
matplotlib
imageio

# 数据处理
pandas
tqdm

# 视频处理 (仅 EWAP 可视化需要)
opencv-python (cv2)
```

### 安装步骤

```bash
# 1. 创建 conda 环境 (推荐)
conda create -n hgatgt python=3.9
conda activate hgatgt

# 2. 安装 PyTorch (根据 CUDA 版本选择)
# CUDA 11.8 示例：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 PyTorch Geometric
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# 4. 安装其他依赖
pip install numpy scipy scikit-learn matplotlib imageio pandas tqdm opencv-python
```

---

## 快速开始

### 仿真环境 (sim_env)

#### 1. 生成合成数据

```bash
cd sim_env
python generate_data.py
```

这将在 `./data/` 下生成训练集（2000 samples）、验证集（200 samples）和测试集（100 samples），每个样本包含 50 帧的雷达量测序列。

数据场景分布：
- **汇聚场景 (40%)**：2-5 个群朝 1-2 个集结点飞行，到达后合并
- **分裂场景 (30%)**：1-3 个群在行进中随机分裂
- **混合场景 (30%)**：预设的分裂+合并复合场景

物理参数：
- 区域大小：1000×1000
- 量测噪声标准差：1.5m
- 检测概率：0.95
- 泊松杂波率：8

#### 2. 训练模型

```bash
python train_sim.py
```

训练参数：
- 优化器：AdamW (lr=0.0005, weight_decay=1e-4)
- 学习率调度：CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
- 梯度累积步数：8
- 梯度裁剪：max_norm=1.0
- 训练轮数：50 epochs
- 最优模型自动保存到 `sim_env/model/best_model_v5.pth`

#### 3. 评估

```bash
python evaluate.py
```

在 100 个测试场景上对比 5 种算法：Baseline (DBSCAN+KF)、GM-CPHD、CB-MeMBer、Graph-MB、H-GAT-GT。

#### 4. 可视化

```bash
# 生成跟踪动画 GIF
python track_gif_gen.py

# 生成静态轨迹图
python visualize_sim_dataset_trajectory.py
```

---

### EWAP 真实数据环境 (ewap_env)

#### 1. 数据预处理

```bash
cd ewap_env

# 步骤一：将原始 EWAP 数据转换为统一格式
python prepare_ewap.py

# 步骤二：生成伪标签（基于物理规则的同伴检测）
python prepare_pseudo_data.py
```

伪标签生成规则：
- 距离阈值：< 2.0m
- 静止行人：速度 < 0.2 m/s 时直接归为同伴
- 运动行人：余弦相似度 > 0.85 且速度差 < 0.6 m/s
- 最少共现帧数：>= 5 帧
- 使用连通分量聚类形成群组

坐标转换：`scaled = raw × 50.0 + [500, 500]`

#### 2. 训练模型

```bash
python train_ewap.py
```

从头训练 100 epochs，使用伪标签数据。最优模型保存到 `ewap_env/model/best_model_ewap_v2.pth`。

#### 3. 评估

```bash
python evaluate_ewap.py
```

在 ETH 和 Hotel 两个场景上对比 6 种算法（比仿真多了 Social-STGCNN）。所有算法接收相同的加噪量测（位置噪声 σ=3.0，速度噪声 σ=0.5）。

#### 4. 视频可视化

```bash
python visualize_eth_on_video.py
```

将 GNN 群检测结果叠加到 ETH 监控视频上，使用单应矩阵 (H) 进行世界坐标到像素坐标的转换。输出 MP4 视频。

---

## 数据流水线

### 数据格式

每帧数据以 `.npy` 格式存储，包含以下字段：

```python
{
    'meas':       np.array shape [N, 2],   # N 个量测点的 (x, y) 坐标
    'labels':     np.array shape [N],       # 每个量测点的群 ID (0 = 杂波)
    'gt_centers': np.array shape [G, 3],    # G 个群的 [group_id, center_x, center_y]
}
```

### 图构建过程 (RadarFileDataset)

```
原始量测点 → 半径图构建 → 节点/边特征提取 → PyG Data 对象
```

1. **半径图构建**：以 `conn_radius` 为阈值构建邻接关系
   - 仿真环境：`conn_radius = 30.0`
   - EWAP 环境：`conn_radius = 80.0`

2. **节点特征**：
   - 仿真：`[x, y]` (2D)
   - EWAP：`[x, y, vx, vy]` (4D)

3. **边特征**：
   - 仿真：`[rel_x, rel_y, dist]` (3D)
   - EWAP：`[rel_x, rel_y, dist, rel_vx, rel_vy, cos_sim]` (6D)

4. **边标签**：若两端点有相同的非零群 ID 则为 1，否则为 0

---

## 训练细节

### 损失函数

训练使用联合损失：

```
L_total = L_edge + λ · L_regression
```

其中 λ = 1.0。

#### 边分类损失

```python
L_edge = BCE(pred_scores, edge_labels)
```

使用动态计算的正样本权重处理类别不平衡（权重上限 10.0）。

#### 偏移回归损失（异方差 NLL）

```python
L_reg = 0.5 × mean[ (μ - target)² / σ² + log(σ²) ]
```

其中 `μ` 为预测偏移，`σ` 为预测不确定性（通过 Softplus 保证正值），`target` 为量测点到群中心的真实偏移。

### 训练策略

| 策略 | 说明 |
|------|------|
| 梯度累积 | 每 8 步更新一次参数，模拟更大的 batch size |
| 梯度裁剪 | max_norm=1.0，防止 Transformer 梯度爆炸 |
| 学习率调度 | CosineAnnealingWarmRestarts (T_0=10, T_mult=2) |
| 权重衰减 | 1e-4 (AdamW) |
| Dropout | 0.1 (TransformerConv + 边分类头) |

---

## 评估与基线对比

### 对比算法

| 算法 | 类型 | 文件 | 说明 |
|------|------|------|------|
| **Baseline** | 传统 | `trackers/baseline.py` | DBSCAN 聚类 + 恒速卡尔曼滤波 |
| **GM-PHD** | RFS | `trackers/gm_phd.py` | 高斯混合概率假设密度滤波器 |
| **GM-CPHD** | RFS | `trackers/gm_cphd.py` | 带势估计的 GM-PHD 滤波器 |
| **CB-MeMBer** | RFS | `trackers/cbmember.py` | 势平衡多伯努利滤波器 |
| **Graph-MB** | 图+RFS | `trackers/graph_mb.py` | 图辅助多伯努利 + UKF (5D 状态) |
| **Social-STGCNN** | 学习 | `trackers/social_stgcnn_tracker.py` | 时空图卷积网络 (仅 EWAP 评估使用) |
| **H-GAT-GT (Ours)** | 学习 | `sim_env/model.py` + `trackers/gnn_processor.py` | 本文方法 |

### H-GAT-GT 推理流程

```
原始量测 → GNN 边分类 + 偏移预测 → 偏移修正后坐标 → DBSCAN(eps=30) 聚类
         → GNN 后处理器（6D 恒加速度卡尔曼滤波 + 两级级联匈牙利关联）→ 跟踪结果
```

GNN 后处理器 (`GNNPostProcessor`) 特点：
- 6 维状态：`[x, y, vx, vy, ax, ay]`（恒加速度模型）
- 两级级联匈牙利匹配（阈值 40 和 90）
- 形状相似度加权代价矩阵

---

## 消融实验

消融实验在仿真环境上进行，验证各关键组件的贡献。

### 消融变体

| 变体名称 | 修改内容 | 实现方式 |
|----------|---------|---------|
| **Full_Model** | 完整模型（基准） | 所有组件启用 |
| **No_Fourier** | 移除傅里叶位置编码 | `use_fourier=False` |
| **No_Adaptive_Fusion** | 仅使用最后一层输出 | `fusion_mode='last'` |
| **Plain_GCN** | 用 GCNConv 替代 TransformerConv | `use_transformer=False` |

### 运行消融实验

```bash
cd sim_env/run_ablation

# 训练各消融变体
python train_ablation.py

# 对比评估
python run_comparison.py
```

### 消融结果

| 变体 | MOTA | 变化 |
|------|------|------|
| Full_Model | **0.7848** | — |
| No_Fourier | 0.7684 | -2.1% |
| No_Adaptive_Fusion | 0.7692 | -2.0% |
| Plain_GCN | 0.7185 | **-8.4%** |

**结论**：TransformerConv 是最关键的组件（贡献 8.4%），因其能利用边特征参与注意力计算。傅里叶编码和自适应融合的贡献相当（约 2%）。

---

## 可视化

### 仿真环境可视化

| 脚本 | 输出 | 说明 |
|------|------|------|
| `track_gif_gen.py` | `sim_env/output/track_result_gif/*.gif` | 逐帧跟踪动画（含群边界框） |
| `visualize_sim_dataset_trajectory.py` | `sim_env/output/*.png` | 累积轨迹静态图 |

### EWAP 环境可视化

| 脚本 | 输出 | 说明 |
|------|------|------|
| `visualize_eth_on_video.py` | `ewap_env/output/track_result_mp4/*.mp4` | ETH 视频叠加 GNN 群检测结果 |

EWAP 可视化使用单应矩阵 (H) 将世界坐标转换为像素坐标，并计算配对聚类指标：
- ARI (调整兰德指数): 0.7988
- Precision: 97.53%
- Recall: 65.05%
- F1-Score: 0.7805

---

## 评估指标

本项目使用 `metrics.py` 中的 `TrackingMetrics` 类进行统一评估，涵盖以下指标：

### 跟踪质量指标

| 指标 | 说明 | 方向 |
|------|------|------|
| **MOTA** | 多目标跟踪精度 = 1 - (Miss + FP + IDSW) / GT | 越大越好 |
| **MOTP** | 多目标跟踪精度（匹配距离均值） | 越小越好 |
| **IDSW** | 身份切换次数 | 越小越好 |
| **FAR** | 虚警率 (每帧虚警数) | 越小越好 |

### 定位质量指标

| 指标 | 说明 | 方向 |
|------|------|------|
| **OSPA (Total)** | 最优子模式分配距离（综合） | 越小越好 |
| **OSPA (Loc)** | OSPA 定位分量 | 越小越好 |
| **OSPA (Card)** | OSPA 势分量 | 越小越好 |
| **RMSE (Pos)** | 位置均方根误差 | 越小越好 |
| **Count Err** | 群数量绝对误差 | 越小越好 |

### 聚类质量指标

| 指标 | 说明 | 方向 |
|------|------|------|
| **Purity** | 聚类纯度（每簇最大类比例） | 越大越好 |
| **Completeness** | 聚类完整度（每类最大簇比例） | 越大越好 |
| **G-IoU** | 广义交并比 [-1, 1] | 越大越好 |

### 关键参数
- 匹配阈值 (MOTA/MOTP)：40.0
- OSPA 参数：c=50.0, p=2

---

## 实验结果

### 仿真环境对比结果

| 算法 | MOTA ↑ | MOTP ↓ | IDSW ↓ | FAR ↓ | OSPA ↓ | RMSE ↓ | Count Err ↓ | G-IoU ↑ |
|------|--------|--------|--------|-------|--------|--------|-------------|---------|
| **H-GAT-GT (Ours)** | **0.781** | **2.572** | 384 | 0.0006 | **13.119** | **3.496** | **0.526** | **0.601** |
| GM-CPHD | 0.749 | 3.288 | 441 | 0.0014 | 15.255 | 4.888 | 0.601 | 0.000 |
| CB-MeMBer | 0.731 | 3.409 | 465 | 0.0068 | 16.197 | 5.239 | 0.645 | 0.000 |
| Baseline | 0.724 | 3.650 | 300 | 0.0000 | 17.468 | 5.034 | 0.699 | 0.000 |
| Graph-MB | 0.637 | 3.945 | 423 | 0.0016 | 21.806 | 5.313 | 0.911 | 0.000 |

H-GAT-GT 在仿真环境中取得最优表现，MOTA 达到 0.781，OSPA 和 RMSE 均为最低。值得注意的是，H-GAT-GT 是唯一取得正 G-IoU（0.601）的方法，表明其聚类边界与真实群组有显著重叠。

---

### ETH 场景对比结果

| 算法 | MOTA ↑ | MOTP ↓ | OSPA ↓ | RMSE ↓ |
|------|--------|--------|--------|--------|
| **H-GAT-GT (Ours)** | **0.957** | **3.746** | **4.073** | **4.235** |
| Social-STGCNN | 0.920 | 4.336 | 8.709 | 5.229 |
| GM-CPHD | 0.847 | 4.461 | 9.063 | 5.548 |
| CB-MeMBer | 0.823 | 4.502 | 10.142 | 5.601 |
| Graph-MB | 0.801 | 4.578 | 12.354 | 5.612 |
| Baseline | 0.782 | 4.615 | 15.810 | 5.669 |

H-GAT-GT 在所有核心指标上均取得最优表现，MOTA 达到 0.957，比最强基线 Social-STGCNN 高出 3.7 个百分点。

---

## 文件说明速查

| 文件路径 | 功能描述 |
|---------|---------|
| `metrics.py` | 通用跟踪评估指标类 |
| `sim_env/config.py` | 全局配置文件 (数据、模型、训练参数) |
| `sim_env/model.py` | GNNGroupTracker 模型 (仿真版, 2D) |
| `sim_env/dataset.py` | RadarFileDataset 数据集 (仿真版) |
| `sim_env/generate_data.py` | ActiveInteractionScenarioEngine 数据生成 |
| `sim_env/train_sim.py` | 仿真数据训练入口 |
| `sim_env/evaluate.py` | 仿真环境 5 算法对比评估 |
| `sim_env/track_gif_gen.py` | 跟踪结果 GIF 动画生成 |
| `sim_env/run_ablation/ablation_model.py` | 消融实验可配置模型 |
| `sim_env/run_ablation/train_ablation.py` | 消融实验训练 |
| `sim_env/run_ablation/run_comparison.py` | 消融结果对比 |
| `ewap_env/config.py` | EWAP 环境配置 |
| `ewap_env/model.py` | GNNGroupTracker 模型 (EWAP 版, 4D) |
| `ewap_env/dataset.py` | RadarFileDataset 数据集 (EWAP 版, 含速度) |
| `ewap_env/prepare_ewap.py` | EWAP 原始数据格式转换 |
| `ewap_env/prepare_pseudo_data.py` | 伪标签生成 (物理规则同伴检测) |
| `ewap_env/train_ewap.py` | EWAP 数据训练入口 |
| `ewap_env/evaluate_ewap.py` | EWAP 环境 6 算法对比评估 |
| `ewap_env/visualize_eth_on_video.py` | ETH 视频叠加可视化 |
| `trackers/baseline.py` | DBSCAN + 卡尔曼滤波 基线跟踪器 |
| `trackers/gm_cphd.py` | GM-CPHD 滤波器 |
| `trackers/gm_phd.py` | GM-PHD 滤波器 |
| `trackers/cbmember.py` | CB-MeMBer 滤波器 |
| `trackers/graph_mb.py` | 图辅助多伯努利跟踪器 |
| `trackers/gnn_processor.py` | GNN 后处理器 (级联匈牙利关联) |
| `trackers/social_stgcnn_tracker.py` | Social-STGCNN 跟踪器 |
| `trackers/kalman_box.py` | 卡尔曼滤波辅助类 |
