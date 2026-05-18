# GA-DAA-Bitcoin-Difficulty-Adjustment
Bitcoin blockchain difficulty adjustment based on NSGA-II genetic algorithm
# GA-DAA: Bitcoin Difficulty Adjustment Based on NSGA-II

基于NSGA-II多目标遗传算法的比特币区块链难度调整方案

## 项目简介

本项目实现了GA-DAA方案，将NSGA-II多目标遗传算法引入比特币PoW难度调整，
设计了基于偏差阈值的GA触发机制，实现双变量（目标出块时间、调整间隔）
双目标（出块时间标准差、难度标准差）的自适应优化。

## 文件结构

- `blockchain_simulator.py`：区块链事件驱动仿真框架 + Bitcoin DAA实现
- `ga_daa.py`：NSGA-II完整实现 + GA-DAA逻辑
- `run_experiments_final.py`：5种哈希率场景实验组织与可视化

## 运行环境

- Python 3.12
- NumPy 1.26
- Matplotlib 3.8

## 快速运行

pip install numpy matplotlib

python run_experiments_final.py

## 作者

张俊林，暨南大学网络空间安全学院

指导教师：吴永东 教授
