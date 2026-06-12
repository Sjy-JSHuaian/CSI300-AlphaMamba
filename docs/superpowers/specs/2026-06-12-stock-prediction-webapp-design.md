# CSI300 AlphaMamba — 股票预测交互分析工作台 设计文档

**日期**: 2026-06-12
**状态**: Approved
**版本**: 1.0

---

## 1. 概述

将现有 CSI300 AlphaMamba 命令行预测系统改造为交互式 Web 分析工作台。用户可通过浏览器选择日期、切换预测模式、调整参数，实时获取 Top-5 股票推荐及可视化分析。

### 1.1 目标

- 将 CLI 预测脚本包装为 HTTP API
- 提供 React 前端实现交互式预测体验
- 支持 4 种预测模式切换和参数实时调整
- 可视化展示预测结果、模型对比、市场状态

### 1.2 非目标

- 不做实时行情接入（使用历史数据）
- 不做用户认证系统
- 不做数据库持久化（直接读 CSV/模型文件）

---

## 2. 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 后端框架 | FastAPI | 异步 Python Web 框架，自动 Swagger 文档 |
| 后端服务器 | Uvicorn | ASGI 服务器 |
| 前端框架 | React 18 + TypeScript | 组件化 UI |
| UI 组件库 | Ant Design 5 | 日期选择、表格、滑块、标签页等 |
| 图表库 | ECharts 5 (echarts-for-react) | 折线图、饼图、散点图、仪表盘 |
| 构建工具 | Vite | 快速开发服务器 + 生产构建 |
| 包管理 | pnpm | 高效的前端依赖管理 |

---

## 3. 系统架构

```
React 前端 (localhost:5173)
    │
    │ REST API (JSON)
    ▼
FastAPI 后端 (localhost:8000)
    │
    ├── 模型加载层 (model/regime6b/*.txt, *.pkl)
    ├── 特征工程层 (features_phase6b.py)
    ├── 预测引擎层 (predict_phase6b, consensus, meta_ranker, phase7)
    └── 组合优化层 (portfolio_optimizer.py)
```

### 3.1 API 端点设计

| Method | Path | 描述 |
|--------|------|------|
| GET | `/api/market/dates` | 获取可用预测日期列表 |
| GET | `/api/market/status?date=YYYY-MM-DD` | 获取某日市场状态 (bull_strength, regime) |
| POST | `/api/predict` | 执行单模式预测 |
| POST | `/api/predict/compare` | 执行多模式并行预测对比 |
| POST | `/api/backtest` | 获取回测数据 (收益曲线、指标) |
| GET | `/api/history` | 获取历史预测记录 |

### 3.2 POST `/api/predict` 请求体

```json
{
  "date": "2025-06-12",
  "mode": "phase6b",
  "params": {
    "temperature": 0.5,
    "max_per_industry": 2,
    "corr_threshold": 0.8,
    "min_score_ratio": 0.7,
    "top_n": 10,
    "select_k": 5,
    "bull_gate": 0.6,
    "turnover_penalty": 0.02,
    "persistence_bonus": 0.01
  }
}
```

### 3.3 POST `/api/predict` 响应体

```json
{
  "date": "2025-06-12",
  "mode": "phase6b",
  "bull_strength": 0.82,
  "regime_label": "STRONG_BULL",
  "portfolio": [
    {"stock_id": "600519", "weight": 0.285, "score": 0.92, "sector_id": 0},
    {"stock_id": "000858", "weight": 0.231, "score": 0.87, "sector_id": 0},
    {"stock_id": "601318", "weight": 0.194, "score": 0.81, "sector_id": 1},
    {"stock_id": "000333", "weight": 0.162, "score": 0.76, "sector_id": 2},
    {"stock_id": "600036", "weight": 0.128, "score": 0.71, "sector_id": 1}
  ],
  "bull_top5": ["600519", "000858", "601318", "000333", "600036"],
  "nonbull_top5": ["600519", "000858", "600036", "601318", "000725"],
  "consensus_overlap": 4,
  "score_distribution": {
    "min": 0.12, "max": 0.92, "mean": 0.45, "std": 0.18
  },
  "sector_distribution": {
    "白酒": 2, "金融": 1, "家电": 1, "银行": 1
  }
}
```

---

## 4. 前端设计

### 4.1 页面结构

```
┌─────────────────────────────────────────────┐
│  Header: CSI300 AlphaMamba · 分析工作台      │
├─────────────────────────────────────────────┤
│  Tab: 📊预测 | 📈回测 | 🔬对比 | ⚙️调参 | 📋历史 │
├─────────────────────────────────────────────┤
│                                              │
│  当前标签页内容区                              │
│                                              │
└─────────────────────────────────────────────┘
```

### 4.2 标签页详细设计

#### 📊 预测 (PredictTab)

- **上半部分**：控制栏
  - 日期选择器 (DatePicker, 默认最新日期)
  - 预测模式下拉框 (Select: Phase6B / Consensus / MetaRanker / Phase7 Safe)
  - "运行预测" 按钮 (Button, loading 状态)
  - 市场状态卡片：Bull Strength 进度条 + regime 标签

- **下半部分**：结果展示
  - 5 张股票卡片 (Card)，每张显示：
    - 排名数字 (🥇🥈🥉)
    - 股票代码 + 名称
    - 权重 (Progress 进度条)
    - 得分
  - 行业分布饼图 (ECharts Pie)
  - 得分分布柱状图 (ECharts Bar, Top-20)

#### 📈 回测 (BacktestTab)

- 日期范围选择器 (RangePicker)
- 指标卡片行：累计收益 / 夏普比率 / 最大回撤 / 胜率
- 累计收益曲线 (ECharts Line, 含基准对比)
- 月度收益热力图 (ECharts Heatmap)

#### 🔬 对比 (CompareTab)

- 日期选择器
- "并行对比" 按钮
- 四列并排结果表 (Table: stock_id × 4 模式得分)
- Top-5 重叠矩阵图
- 得分分布小提琴图或箱线图

#### ⚙️ 调参 (TuningTab)

- 日期选择器 + 模式选择器
- 参数滑块组 (Slider):
  - temperature: 0.1 ~ 2.0, step 0.1
  - max_per_industry: 1 ~ 5, step 1
  - corr_threshold: 0.5 ~ 1.0, step 0.05
  - min_score_ratio: 0.3 ~ 0.9, step 0.05
  - bull_gate: 0.0 ~ 1.0, step 0.05
  - top_n: 5 ~ 30, step 1
- "应用参数重跑" 按钮
- 实时结果更新

#### 📋 历史 (HistoryTab)

- 历史预测日期列表 (Table)
- 点击展开每次预测的详细持仓
- 换手率趋势折线图
- 导出 CSV 按钮

### 4.3 组件树

```
App
├── AppHeader (标题栏)
├── AppTabs (标签导航)
│   ├── PredictTab
│   │   ├── DateSelector
│   │   ├── ModeSelector
│   │   ├── MarketStatusCard
│   │   ├── StockCardGrid (×5)
│   │   ├── SectorPieChart
│   │   └── ScoreBarChart
│   ├── BacktestTab
│   │   ├── DateRangeSelector
│   │   ├── MetricsCardRow
│   │   ├── CumulativeReturnChart
│   │   └── MonthlyHeatmap
│   ├── CompareTab
│   │   ├── DateSelector
│   │   ├── MultiModeTable
│   │   ├── OverlapMatrix
│   │   └── ScoreDistributionChart
│   ├── TuningTab
│   │   ├── DateSelector
│   │   ├── ParamSliders
│   │   └── ResultPreview
│   └── HistoryTab
│       ├── HistoryTable
│       ├── TurnoverTrendChart
│       └── ExportButton
```

---

## 5. 后端设计

### 5.1 项目结构

```
backend/
├── main.py              # FastAPI 应用入口
├── api/
│   ├── __init__.py
│   ├── market.py         # 市场状态接口
│   ├── predict.py        # 预测接口
│   ├── backtest.py       # 回测接口
│   └── history.py        # 历史记录接口
├── services/
│   ├── __init__.py
│   ├── predictor.py      # 预测逻辑封装（复用现有模块）
│   └── backtest.py       # 回测计算
├── schemas/
│   ├── __init__.py
│   └── models.py         # Pydantic 数据模型
├── requirements.txt
└── start.sh
```

### 5.2 关键实现细节

- **模型加载**：应用启动时一次性加载所有模型到内存（lgb.Booster, scaler, feature_cols），避免每次请求重新加载
- **特征工程**：`data/train.csv` 在启动时加载并做特征工程，预测时按日期切片
- **并发**：FastAPI 的 async 特性保证 I/O 不阻塞，CPU 密集型预测操作使用 `run_in_executor`
- **CORS**：允许 `localhost:5173` 跨域请求
- **错误处理**：统一异常处理，返回 `{"error": "message", "detail": "..."}`

### 5.3 依赖

新增依赖（在 pyproject.toml 中添加）：
- `fastapi>=0.115`
- `uvicorn>=0.30`
- `pydantic>=2.0`

现有依赖全部复用：
- `lightgbm`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `joblib`

---

## 6. 数据流

```
用户选择日期 + 模式 → 点击"运行预测"
  → POST /api/predict {date, mode, params}
    → 后端按日期切片特征数据
    → 按模式执行预测流程
    → 组合优化
    → 返回 JSON
  → 前端更新 StockCardGrid + Charts
```

---

## 7. 错误处理

| 场景 | 处理 |
|------|------|
| 日期无数据 | 返回 400 + "选定日期无可用股票数据" |
| 日期股票不足5只 | 返回 400 + "可用股票不足5只" |
| Bull Gate 触发 (Phase7) | 返回空组合 + "bull_gate 触发，建议空仓" |
| 模型文件缺失 | 启动时报错退出，返回 500 |
| 参数越界 | Pydantic 自动校验，返回 422 |

---

## 8. 实现优先级

### P0 (第一阶段 — 核心交付)
1. FastAPI 后端 + 模型加载
2. `/api/market/dates`, `/api/market/status`, `/api/predict` 三个端点
3. React 脚手架 + PredictTab（日期选择 + 模式切换 + 结果展示）
4. TuningTab（参数调整）

### P1 (第二阶段 — 增强)
5. CompareTab（多模式对比）
6. BacktestTab（回测曲线）
7. HistoryTab（历史记录）

---

## 9. 文件变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `backend/main.py` | FastAPI 应用入口 |
| `backend/api/__init__.py` | |
| `backend/api/market.py` | 市场状态 API |
| `backend/api/predict.py` | 预测 API |
| `backend/api/backtest.py` | 回测 API |
| `backend/api/history.py` | 历史记录 API |
| `backend/services/__init__.py` | |
| `backend/services/predictor.py` | 预测服务层 |
| `backend/services/backtest.py` | 回测计算服务 |
| `backend/schemas/__init__.py` | |
| `backend/schemas/models.py` | Pydantic Schema |
| `backend/start.sh` | 后端启动脚本 |
| `frontend/` | React 项目 (Vite + React + TS) |

### 修改文件

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 添加 fastapi, uvicorn 依赖 |

---

## 10. 自审清单

- [x] 无 placeholder / TODO
- [x] API 请求/响应格式完整定义
- [x] 组件树和数据流清晰
- [x] 错误处理场景覆盖
- [x] 实现优先级明确
- [x] 与现有代码的集成路径清晰
