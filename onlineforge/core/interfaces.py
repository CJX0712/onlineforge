"""接口契约（Protocol）。调用方向单向无环：

cli -> pipeline -> {data, learners, drift, eval} -> core
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class StreamLearner(Protocol):
    """流式学习器：逐步预测 + 增量训练。"""

    name: str
    backend: str

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        """用单条样本增量训练。"""
        ...

    def predict_one(self, x: List[float]) -> Any:
        """预测单条样本标签。"""
        ...

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        """返回类别 -> 概率。行和为 1。"""
        ...

    def reset(self) -> None:
        """丢弃已学状态（漂移触发时调用）。"""
        ...

    def classes(self) -> List[Any]:
        """返回已知类别列表。"""
        ...


@runtime_checkable
class DriftDetector(Protocol):
    """漂移检测器：输入标量（如预测正确与否）返回是否漂移。"""

    name: str

    def update(self, value: float) -> bool:
        """返回 True 表示本步检测到漂移。"""
        ...

    def reset(self) -> None: ...


@runtime_checkable
class StreamMetric(Protocol):
    """流式指标。"""

    name: str

    def update(self, correct: int) -> None: ...

    def get(self) -> float: ...
