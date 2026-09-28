# OnlineForge 架构文档

> 作者：晨星 · 随机创新的世界级 AI 系统（forge 系列）

本文档描述 OnlineForge 的模块职责、接口契约、数据流、旗舰算法与可复现性设计。

---

## 1. 设计原则

1. **契约先行**：所有跨模块交互通过 `core/interfaces.py` 的 `Protocol`（`StreamLearner` / `DriftDetector` / `StreamMetric`）约束，调用方向单向无环。
2. **单一职责**：每个模块只做一件事。
3. **离线可运行**：SOTA 后端（river）通过 `try/except ImportError` + `available_*()` 探测；缺失时纯 numpy 兜底链路保证零下载 demo 可跑。
4. **可复现**：所有随机性由固定 `seed` 控制（`Config.random_seed` / 各生成器 `seed`）。

```
cli → pipeline → {data, learners, drift, eval} → core
```

---

## 2. 模块职责

| 模块 | 职责 |
|------|------|
| `core/types` | `StreamExample` / `DriftSignal` / `EvalRecord` / `BenchmarkResult` / `Benchmark` |
| `core/errors` | `ForgeError` 及 E1xx–E5xx 错误码 |
| `core/config` | `Config`，支持 `ONLINEFORGE_*` 环境变量覆盖 |
| `core/interfaces` | `StreamLearner` / `DriftDetector` / `StreamMetric` Protocol |
| `data/generators` | 合成漂移流：`none / abrupt / gradual / incremental` + sklearn 数据集转流 |
| `learners/numpy_backends` | 纯 numpy 在线学习器：感知机 / 高斯 NB / PA（离线兜底）|
| `learners/sklearn_backends` | `SGDClassifier` / `PassiveAggressiveClassifier` 的 `partial_fit` 包装 |
| `learners/river_backends` | river 0.26：HoeffdingTree / HoeffdingAdaptiveTree / LeveragingBagging / 在线 LogReg |
| `learners/batch_baseline` | 批式基线：`StaticBaseline`（fit-once）、`PeriodicBatch`（周期重拟合）|
| `learners/registry` | `build_learners(n_classes, config)`：按配置组装学习器清单 |
| `drift/detectors` | `ADWIN`（纯 numpy）+ river 漂移检测器（ADWIN/DDM/KSWIN/PageHinkley）包装 |
| `eval/metrics` | `PrequentialAccuracy` / `RollingAccuracy` / `cohen_kappa` / `recovery_lag` |
| `pipeline/streaming_pipeline` | `run()` 单学习器跑流；`benchmark()` 多学习器×多流编排 |
| `pipeline/flagship` | `DAERFlagship` 漂移感知集成路由器 |
| `cli` | 命令行入口，打印基准表 + 写 `benchmark.json` |
| `examples/run_demo` | 端到端演示，落盘 `benchmark.json` |

---

## 3. 接口契约（核心）

```python
class StreamLearner(Protocol):
    name: str
    backend: str
    def partial_fit_one(self, x: List[float], y) -> None: ...
    def predict_one(self, x: List[float]) -> Any: ...
    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]: ...  # 行和=1
    def reset(self) -> None: ...
    def classes(self) -> List[Any]: ...
```

所有学习器（numpy / sklearn / river / 批式 / 旗舰）均实现该契约，因此 `pipeline.run()` 可无差别评测任意学习器。

---

## 4. 合成漂移流生成器

对 `n_classes` 个高斯簇中心 `C_k`，按漂移类型随时间演化：

- `none`：`C_k` 固定。
- `abrupt`：中点处 `C_k → C_k + Δ`（突变漂移）。
- `gradual`：中点附近窗口内 `C_k` 线性插值过渡。
- `incremental`：`C_k` 随时间绕原点缓慢旋转（渐进漂移）。

**可验证不变量（单测交叉验证）**
- 同一 `(seed, drift_type)` 生成的流完全确定（逐元素一致）。
- `abrupt` 流在过中点后与 `none` 流逐元素不同（漂移确实存在）。
- 同分布下自适应在线学习器准确率 ≥ 静态基线；漂移下静态基线明显下滑。

---

## 5. 漂移检测：ADWIN

纯 numpy 实现自适应窗口（ADWIN）近似：

对当前窗口 `W`（长度 `n`，上限 `adwin_max_len`），枚举分割点 `k`：

```
n0=k, n1=n-k
mean0 = Σ_{i<k} W_i / n0
mean1 = Σ_{i>=k} W_i / n1
eps(n0,n1) = sqrt( (1/(2δ)) · (1/n0 + 1/n1) · ln(2/δ) )
if |mean0 - mean1| > eps: 检测到漂移
```

输入为该学习器逐步**正确率（1/0）**；当正确率分布发生结构性变化（概念漂移）时触发。

**不变量（单测）**
- 正确率由 0.9 突变为 0.1 的合成序列：检测器在有限步内 firing。
- 平稳 0.9 正确率序列：不误报。

---

## 6. 旗舰：DAER（Drift-Aware Ensemble Router）

输入：一组基学习器 `L_1..L_m`（含 numpy / sklearn / river 在线学习器）。

```
初始化: weights = [1/m]; 每个学习器一个近期正确率滚动窗口(长度=window); 全局 ADWIN

predict_one(x):
    votes = Σ_i weights[i] · onehot(L_i.predict_one(x))
    return argmax(votes)

partial_fit_one(x, y):
    preds_i = L_i.predict_one(x)              # test-then-train
    for i: 记录 preds_i==y 到滚动窗口; L_i.partial_fit_one(x, y)
    ens = vote(preds)
    if 全局ADWIN.update(ens==y):              # 检测到概念漂移
        reweight()                            # 按近期准确率 softmax（温度锐化）重算权重
        reset_weakest()                        # 重置最弱 1/3 学习器
```

**设计要点**
- 温度锐化（`temperature=4.0`）放大高准确率成员优势，使旗舰逼近最优成员（非劣），同时保留集成鲁棒性。
- 重置只发生在漂移信号之后，且只对弱学习器执行，避免破坏已学强的表征。

---

## 7. 基准方法论

`pipeline.benchmark(streams, learners, config)`：

- 每条流对每个学习器从**全新状态**开始（`reset()`），保证公平。
- 指标：前缀（prequential）准确率、滑窗（最近 `window_size` 步）准确率、`drifts_detected`（该学习器误差流上的 ADWIN 触发次数）、`recovery_lag`（漂移后恢复到漂移前水平的平均步数）、`runtime_s`。
- 固定 `seed`，结果可复现（见 `benchmark.json`）。

---

## 8. 离线兜底链路

当且仅当 `river` 不可用时：

- 学习器回退到 `numpy_perceptron` / `numpy_gaussian_nb` / `numpy_pa`（全部纯 numpy，零外部依赖）。
- 漂移检测回退到纯 numpy `ADWIN`。
- 批式基线仍由 scikit-learn 提供（sklearn 为硬依赖）。

即：仅安装 `numpy + scikit-learn` 即可完整运行 demo 与基准（`make_stream` 也仅依赖 numpy）。

---

## 9. 复现命令

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install numpy scipy scikit-learn pytest river
.venv/Scripts/python.exe -m pytest -q -W ignore::UserWarning   # 11 测试全绿
.venv/Scripts/python.exe -m onlineforge.cli --n 2500           # 写 benchmark.json
```

镜像：`docker build -t onlineforge . && docker run --rm onlineforge`
