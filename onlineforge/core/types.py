"""核心数据类型（dataclass）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StreamExample:
    """单条流式样本。x 为定长特征向量，y 为类别标签。"""

    x: List[float]
    y: Any


@dataclass
class DriftSignal:
    """漂移检测结果。fired=True 表示检测到概念漂移。"""

    fired: bool
    step: int
    window_mean_old: float = 0.0
    window_mean_new: float = 0.0
    n_window: int = 0


@dataclass
class EvalRecord:
    """每个时间步的评测记录。"""

    step: int
    y_true: Any
    y_pred: Any
    correct: int
    prequential_acc: float
    drift: bool


@dataclass
class DatasetSpec:
    """数据集/流的描述。"""

    name: str
    n_samples: int
    n_features: int
    n_classes: int
    drift_type: str  # none | abrupt | gradual | incremental
    generator: str = ""


@dataclass
class BenchmarkResult:
    """单个 learner 在单个 stream 上的基准结果。"""

    dataset: str
    learner: str
    prequential_acc: float
    windowed_acc: float
    drifts_detected: int
    recovery_lag: float
    n_classes: int
    n_samples: int
    runtime_s: float
    backend: str  # numpy | sklearn | river | mixed
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "learner": self.learner,
            "prequential_acc": round(self.prequential_acc, 5),
            "windowed_acc": round(self.windowed_acc, 5),
            "drifts_detected": self.drifts_detected,
            "recovery_lag": round(self.recovery_lag, 3),
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
            "runtime_s": round(self.runtime_s, 4),
            "backend": self.backend,
            "notes": self.notes,
        }


@dataclass
class Benchmark:
    """整场基准的集合。"""

    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, r: BenchmarkResult) -> None:
        self.results.append(r)

    def as_dict(self) -> Dict[str, Any]:
        return {"results": [r.as_dict() for r in self.results]}
