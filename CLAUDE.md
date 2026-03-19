# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 先读

- 项目各个文件含义可参考 `项目结构.md`，更完整的实验说明在根目录 `README.md`。
- 项目主要分两种场景：仿真 `sim_env`，真实 EWAP/ETH 数据集 `ewap_env`。
- H-GAT-GT 的主线实现是 **场景模型 (`sim_env/model.py` 或 `ewap_env/model.py`) + `trackers/gnn_processor.py`**。
- 各种对比算法都在 `trackers/` 下，场景级评估入口分别是 `sim_env/evaluate.py` 和 `ewap_env/evaluate_ewap.py`。
- 写好代码后**不要在本地执行训练/评估/可视化命令**；本地环境不可用，由用户上传到服务器运行。
- 仓库里**没有** repo 级统一构建/测试配置；不要臆造 lint/test 命令。
- 若打算了解各个版本的修改信息，请参考 `各个版本修改信息/` 。

## 架构总览

### 统一主线

两套场景都遵循同一条主线：

1. 原始 episode / 轨迹数据转成 `.npy`
2. `RadarFileDataset` 将每一帧转成 PyTorch Geometric 图数据
3. `GNNGroupTracker` 做两项预测：
   - 边分类：两个点是否属于同一群
   - 偏移回归：点到群中心的偏移
4. 用预测偏移修正点位置，再做聚类/连通分量恢复群组
5. `trackers/gnn_processor.py` 负责跨帧跟踪与 ID 关联
6. `metrics.py` 统一计算 MOTA、MOTP、OSPA、Purity、G-IoU 等指标

### `sim_env/`：仿真雷达场景

- `sim_env/generate_data.py` 生成合成群目标场景（汇聚 / 分裂 / 混合）。
- `sim_env/dataset.py` 把仿真 episode 构造成图：节点通常是 2D 位置，边特征通常是 3D 相对几何量。
- `sim_env/model.py` 是仿真版 H-GAT-GT 模型。
- `sim_env/train_sim.py` 训练模型，`sim_env/evaluate.py` 做 5 种算法对比。
- `sim_env/run_ablation/` 是仿真消融实验子模块。

### `ewap_env/`：EWAP / ETH 真实行人场景

- `ewap_env/prepare_ewap.py` 把 `datasets/ewap_dataset/` 下的原始 `obsmat.txt` 转成模型使用的 `.npy` episode。
- `ewap_env/prepare_pseudo_data.py` 用物理规则（距离、速度方向、速度差、共现帧数）生成伪群组标签，作为训练数据来源。
- `ewap_env/dataset.py` 构图时使用 4D 节点特征（位置+速度）和 6D 边特征。
- `ewap_env/model.py` 是 EWAP 版 H-GAT-GT 模型。
- `ewap_env/evaluate_ewap.py` 做 6 种算法对比；比仿真场景多了 `Social-STGCNN` 基线。

### `trackers/`：基线与后处理

- `trackers/gnn_processor.py` 是 H-GAT-GT 推理后的核心后处理器，负责卡尔曼滤波、级联匈牙利匹配、轨迹维护。
- `trackers/baseline.py`、`gm_phd.py`、`gm_cphd.py`、`cbmember.py`、`graph_mb.py`、`social_stgcnn_tracker.py` 是对比算法实现或包装器。
- 如果修改某个基线，通常要同时检查对应场景的 `evaluate*.py` 调用逻辑。

### `Social-STGCNN-master/`

- 第三方基线项目，不是本仓库核心代码。
- 它通过 `trackers/social_stgcnn_tracker.py` 被 `ewap_env/evaluate_ewap.py` 评估脚本调用。

## 关键文件

- `metrics.py`：统一评估指标实现。
- `sim_env/config.py`、`ewap_env/config.py`：路径、checkpoint、设备、特征维度等核心配置。
- `sim_env/model.py`、`ewap_env/model.py`：两套场景各自的 GNN 模型定义。
- `sim_env/dataset.py`、`ewap_env/dataset.py`：图构建入口。
- `trackers/gnn_processor.py`：H-GAT-GT 的跟踪后处理核心。
- `sim_env/evaluate.py`、`ewap_env/evaluate_ewap.py`：主评估入口。

## 修改时的注意点

- 大量脚本依赖相对路径，**尽量假设命令从仓库根目录发起**。这对 `ewap_env` 尤其重要。
- `sim_env/generate_data.py` 重新生成数据时会删除现有 `./data/train`、`./data/val`、`./data/test` 目录再重建。
- `ewap_env/prepare_ewap.py` 当前是把每个行人先当作独立目标做测试数据转换；真正用于训练的群组标签来自 `prepare_pseudo_data.py` 生成的伪标签，而不是直接依赖原始 `groups.txt`。
