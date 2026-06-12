# 代码说明

## 环境配置

| 依赖 | 版本 |
|------|------|
| Python | >=3.10, <3.13 |
| LightGBM | >=4.0 |
| scikit-learn | >=1.3 |
| scipy | >=1.10 |
| pandas | >=2.0 |
| numpy | >=1.24 |
| TA-Lib | >=0.6 |
| joblib | >=1.3 |
| tqdm | >=4.65 |

Docker 镜像基于 `python:3.12-slim-bookworm`，TA-Lib 从源码编译安装（0.4.0），Python 依赖通过 `uv` 管理。

## 数据

使用比赛官方提供的沪深300指数成分股日线数据，包含开盘价、收盘价、最高价、最低价、成交量、成交额等基础行情字段。数据文件为 `data/train.csv`（训练集）和 `data/test.csv`（测试集）。

## 预训练模型

未使用外部预训练模型，所有模型均从随机初始化开始训练。

## 算法

### 整体思路介绍

将股票选择建模为 **Learning to Rank（排序学习）** 问题。对每个交易日，将所有沪深300成分股的特征输入模型，输出每只股票的排序分数，通过组合优化选取得分最高的 5 只股票，按温度软最大化分配权重。

核心流程：
1. **特征工程（102 维）**：60 日窗口 Shape 特征（斜率/自相关/FFT/上涨比率）+ Shock/Event 特征（波动率冲击/缺口/异常收益）+ Market Regime 特征（趋势/波动率体制/市场广度）+ Cross-stock Interaction 特征（Beta/行业排名/龙头相关性）
2. **标签构建（Phase 4）**：Multi-horizon ensemble label = 0.2 × rank(T+3) + 0.5 × rank(T+5) + 0.3 × rank(T+10)，尾部分布放大离散化
3. **Regime-Split 建模（Phase 6B）**：按 bull_strength = sigmoid(ret20/vol20) 分为 Bull/NonBull 两个市场状态，分别训练 LGB Ranker，推理时 soft blend
4. **Meta Ranker（Phase 8）**：Ridge 回归在 Phase 6B 和独立 LGB 模型的候选池上做二阶集成
5. **组合优化（Phase 5）**：贪心选股（同行业 ≤2 只、相关系数惩罚）、温度软最大化权重（T=0.5）

### 方法的创新点

- **Multi-Horizon Label Ensemble**：融合 T+3/T+5/T+10 三个期限的排序信号，平滑单期限噪声窗口，配合尾部分布放大提升 Top-5 区分度
- **Regime-Split 双模型**：按市场牛市强度（bull_strength）将数据分为 Bull/NonBull，分别训练排序模型，推理时按牛市强度连续软混合。解决了单一模型在牛市/震荡市中学到不同 Alpha 结构的问题
- **Shape 特征体系**：用 60 日窗口的 10 种统计形态（线性斜率、自相关 lag1/5/10、FFT 主导频率、上涨比率）描述 6 个核心因子的轨迹形态，替代传统的大量窗口统计特征
- **Meta Ranker 二阶集成**：在 Phase 6B 和独立模型的候选池上，用 Ridge 回归学习微调权重，交互项（p6b_rank × s1_rank）捕捉两模型共识信号
- **共识奖励机制**：若股票同时进入两模型 Top-20，给予 +0.05 的得分奖励，强化高置信度选股

### 网络结构

本方案使用 **LightGBM LambdaRank** 作为主力模型（Phase 6B），配合 Ridge 回归做二阶集成，不使用深度学习网络。

**Phase 6B Bull/NonBull Ranker**：
```
输入 (N, 102)
  → LightGBM Lambdarank (num_leaves=255, lr=0.03)
  → 输出 (N,) 每只股票的排序分数
```

**Meta Ranker (Ridge)**：
```
输入 (M, ~35)  — M 为候选池股票数（~55 只）
  → 特征：p6b_rank, s1_rank, 交互项, 偏离度, 市场环境, 结构特征
  → Ridge Regression (alpha=1.0)
  → 输出 (M,) 微调后的排序分数
```

### 损失函数

- **Phase 6B**：LightGBM `lambdarank` 目标函数，评估指标 `ndcg@5`。Label 通过 tail-amplified discretization 转为整数 relevance（top 10% → 20, top 10-20% → 10, top 20-30% → 5, 其余 0-4）
- **Meta Ranker**：Ridge 回归最小化 MSE，目标为连续 ensemble label

### 数据扩增

未使用数据扩增。

### 模型集成

两阶段集成策略：

1. **Phase 6B Soft Blend**：`score = bull_strength × bull_model + (1-bull_strength) × nonbull_model`，两个模型分数均先做 cross-sectional rank normalization
2. **Meta Ranker 微调**：在候选池上对 Phase 6B 和独立模型的分数进行 Ridge 加权微调，权重通过历史数据学习

### 算法的其他细节

- **Bull Strength**：`sigmoid(idx_ret20 / idx_vol20)`，将市场状态映射为连续值 (0, 1)，替代离散的三分类
- **Cross-sectional Rank Normalization**：所有模型分数在混合前先做日内 cross-sectional rank (pct=True)，确保不同模型分数可比
- **前向收益计算**：label = (Open_T+5 - Open_T+1) / Open_T+1，严格匹配比赛评分公式
- **组合约束**：同板块 ≤2 只、相关系数 >0.8 时线性惩罚、得分低于 Top-1 的 70% 不纳入
