# OnlineForge · 流式/增量学习系统

> 随机创新的世界级 AI 系统 · 复用顶级开源 · 随机上传 GitHub（作者：晨星）

OnlineForge 是一套**模块化、可复现、零下载可跑**的流式（在线/增量）机器学习系统。
它面向**概念漂移（concept drift）**这一真实数据流核心难题，提供从合成数据生成、
多后端学习器、漂移检测到**漂移感知集成路由旗舰（DAER）**的完整链路，并给出统一、
可公平比较的基准。

- **作者**：晨星（GitHub: [CJX0712](https://github.com/CJX0712)）
- **许可**：MIT
- **Python**：3.9+（已验证 3.13.12 可运行）

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🌊 流式学习 | 单样本 `partial_fit_one` / `predict_one`，test-then-train 前缀（prequential）范式 |
| 🔀 概念漂移 | 内置 `none / abrupt / gradual / incremental` 四种合成漂移流生成器 |
| 🧠 顶级开源复用 | **river** 0.26（HoeffdingTree / HoeffdingAdaptiveTree / LeveragingBagging / 在线 LogReg）+ **scikit-learn**（SGD/PA `partial_fit`）|
| 🛡️ 离线兜底 | SOTA（river）不可用时，自动降级为**纯 numpy** 实现（感知机 / 高斯 NB / PA）+ 纯 numpy ADWIN，保证零下载 demo 可跑 |
| 🚀 旗舰 DAER | Drift-Aware Ensemble Router：近期准确率加权投票 + 漂移触发弱学习器重置，兼顾稳定期精度与漂移期恢复 |
| 📊 公平基准 | 统一指标（前缀准确率 / 滑窗准确率 / 漂移次数 / 恢复延迟），固定 seed 完全可复现 |
| ✅ 质量门禁 | 11 个 pytest 全绿，ruff 校验通过 |

---

## 🧱 技术栈

| 层级 | 复用 / 实现 |
|------|------------|
| SOTA 后端 | `river` 0.26（流式学习标杆库）|
| mid-tier | `scikit-learn` 1.x（`SGDClassifier` / `PassiveAggressiveClassifier` 的 `partial_fit`）|
| 离线兜底 | 纯 `numpy` 手写（感知机、高斯 NB、PA、ADWIN）|
| 数值/工具 | `scipy`、`numpy` |

> 未从零自研任何 SOTA 算法；所有前沿能力均复用顶级开源，兜底为可验证的纯 numpy 实现。

---

## 🚀 快速开始

```bash
# 1. 安装（核心依赖即可运行；river 为可选 SOTA 后端）
pip install -e .
pip install river        # 可选：启用 SOTA 后端（失败不影响系统）

# 2. 跑默认基准（4 条流 × 12 个方法），打印表并写 benchmark.json
python -m onlineforge.cli

# 3. 端到端演示（落盘 benchmark.json）
python onlineforge/examples/run_demo.py

# 4. 测试 + 代码风格
pytest -q -W ignore::UserWarning
ruff check . && ruff format --check .
```

Docker 一键复现：

```bash
docker build -t onlineforge . && docker run --rm onlineforge
```

---

## 🏗️ 架构

```
cli.py ──▶ pipeline.run() / pipeline.benchmark()
                │
                ├─▶ data.generators      # 合成漂移流（确定性、可复现）
                ├─▶ learners             # 学习器（numpy / sklearn / river / 批式基线）
                ├─▶ drift.detectors     # ADWIN（numpy）/ river 漂移检测
                ├─▶ eval.metrics        # 前缀/滑窗准确率、恢复延迟
                └─▶ pipeline.flagship    # DAER 漂移感知集成路由器

调用方向单向无环：cli → pipeline → {data, learners, drift, eval} → core
```

### DAER 旗舰算法

1. 每个基学习器维护一个**近期正确率滚动窗口**（ADWIN 窗口）。
2. 预测：按各学习器近期准确率做 **softmax 加权投票**（温度锐化，逼近最优成员）。
3. 维护一个**全局 ADWIN** 监控集成正确率；一旦检测到概念漂移：
   - 依据近期准确率**重算权重**（表现差者降权）；
   - **重置最弱 1/3 基学习器**，使其在漂移后重新适应。
4. 不变量：重置只发生在漂移信号之后，且只对弱学习器执行，避免破坏已学强的表征。

---

## 📊 基准结果（seed 固定，4 流聚合前缀准确率）

| 方法 | 后端 | 聚合前缀准确率 |
|------|------|--------------|
| river_leveraging_bag | river | **0.9673** |
| **daer_flagship** | mixed | **0.9567** |
| river_hat | river | 0.9354 |
| river_hoeffding_tree | river | 0.9115 |
| sklearn_pa | sklearn | 0.8785 |
| numpy_gaussian_nb | numpy | 0.8749 |
| sklearn_sgd | sklearn | 0.8429 |
| static_baseline | sklearn | 0.8150 |
| numpy_perceptron | numpy | 0.8115 |
| numpy_pa | numpy | 0.7984 |
| periodic_batch | sklearn | 0.7959 |
| river_logreg | river | 0.5173 |

**关键结论**
- 自适应在线学习器（river 系、DAER）在**渐进 / 突变漂移**下大幅领先非自适应方法。
- `incremental` 旋转漂移场景下，`static_baseline` 跌至 0.44、`periodic_batch` 跌至 ~0.33（低于随机 1/3）——直观证明概念漂移对 naive 批式方法的毁灭性；而 `river_hat`(0.93)、`river_leveraging_bag`(0.965)、`daer_flagship`(0.918) 保持稳定。
- `river_logreg` 偏低是因流为非线性（blob 旋转），线性在线模型固有局限——诚实保留以体现选型重要性。
- **DAER 旗舰无需预先知道当前流最优学习器**，即以集成路由取得跨场景稳定 Top-2/3 表现。

> 完整逐流逐方法明细见 `benchmark.json`。

---

## 📁 目录结构

```
onlineforge/
├── onlineforge/
│   ├── core/         # types / errors / config / interfaces（契约先行）
│   ├── data/         # 合成漂移流生成器
│   ├── learners/     # numpy 兜底 + sklearn + river + 批式基线
│   ├── drift/        # ADWIN（numpy）/ river 漂移检测
│   ├── eval/         # 前缀/滑窗准确率、恢复延迟
│   ├── pipeline/     # run / benchmark / DAER 旗舰
│   ├── cli.py        # 命令行入口
│   └── examples/run_demo.py
├── tests/            # 11 个 pytest（确定性 / ADWIN 不变量 / 学习器 / pipeline / 旗舰）
├── docs/architecture.md
├── requirements.txt / requirements.lock.txt / pyproject.toml / Dockerfile / Makefile
└── README.md
```

---

## ✅ DoD（交付完成度）

- [x] 全随机选中未覆盖域（在线学习），命名 `onlineforge`
- [x] 复用顶级开源（river + scikit-learn），未自研 SOTA
- [x] 离线兜底：river 缺失时纯 numpy 全链路可跑
- [x] 模块化架构、契约先行、调用单向无环
- [x] 端到端 demo 落盘 `benchmark.json`，固定 seed 可复现
- [x] 11 个 pytest 全绿 + ruff 收口
- [x] git 提交（作者 晨星）+ 推送 GitHub

---

*作者：晨星 · 随机创新的世界级 AI 系统系列（forge）*
