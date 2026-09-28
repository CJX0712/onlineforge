"""DAER — Drift-Aware Ensemble Router（漂移感知集成路由器）。

创新点：
- 用每个基学习器近期（ADWIN 窗口）准确率做 softmax 加权投票；
- 维护全局 ADWIN 监控集成正确率，一旦检测到概念漂移：
    1) 依据各基学习器近期准确率重算权重（表现差的降权）；
    2) 重置最弱的 1/3 基学习器，使其在漂移后重新适应。
从而兼顾稳定期的高精度与漂移期的快速恢复。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List

from ..drift.detectors import ADWIN


class DAERFlagship:
    name = "daer_flagship"
    backend = "mixed"

    def __init__(
        self, learners: List, delta: float = 0.05, window: int = 200, temperature: float = 4.0
    ) -> None:
        if not learners:
            raise ValueError("DAER requires at least one base learner")
        self.learners = list(learners)
        self.delta = delta
        self.window = window
        self.temperature = temperature
        self.weights: List[float] = [1.0 / len(self.learners)] * len(self.learners)
        self._roll: List[Deque[int]] = [deque(maxlen=window) for _ in self.learners]
        self._global_adwin = ADWIN(delta)

    def _vote(self, preds: List[Any]) -> Any:
        votes: Dict[Any, float] = {}
        for w, p in zip(self.weights, preds):
            votes[p] = votes.get(p, 0.0) + w
        return max(votes, key=votes.get)

    def predict_one(self, x: List[float]) -> Any:
        return self._vote([learner.predict_one(x) for learner in self.learners])

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        preds = [learner.predict_one(x) for learner in self.learners]
        for i, learner in enumerate(self.learners):
            correct = 1 if preds[i] == y else 0
            self._roll[i].append(correct)
            learner.partial_fit_one(x, y)
        ens = self._vote(preds)
        gcorrect = 1 if ens == y else 0
        if self._global_adwin.update(gcorrect):
            self._reweight()
            self._reset_weakest()

    def _reweight(self) -> None:
        means = [(sum(d) / len(d) if d else 0.5) for d in self._roll]
        mx = max(means)
        # 温度锐化：放大高准确率成员优势，使旗舰逼近最优成员（非劣），同时保留集成鲁棒性
        exps = [math.exp((m - mx) * self.temperature) for m in means]
        s = sum(exps) or 1.0
        self.weights = [e / s for e in exps]

    def _reset_weakest(self) -> None:
        k = max(1, len(self.learners) // 3)
        order = sorted(
            range(len(self.learners)),
            key=lambda i: sum(self._roll[i]) / max(1, len(self._roll[i])),
        )
        for i in order[:k]:
            self.learners[i].reset()
            self._roll[i].clear()

    def classes(self) -> List[Any]:
        s = set()
        for learner in self.learners:
            s.update(learner.classes())
        return list(s)

    def reset(self) -> None:
        for learner in self.learners:
            learner.reset()
        self.weights = [1.0 / len(self.learners)] * len(self.learners)
        self._roll = [deque(maxlen=self.window) for _ in self.learners]
        self._global_adwin.reset()
