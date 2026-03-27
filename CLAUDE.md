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
