"""批式基线学习器（离线对照）。

- StaticBaseline：仅用前 warm_start 个样本训练一次后冻结（模拟不自适应系统）。
- PeriodicBatch：每 refit_every 步用全部历史重拟合（强离线参照，但代价高）。

均实现 StreamLearner 契约，便于在同一基准里公平比较。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression


class _BatchBase:
    backend = "sklearn"
    _model = None

    def __init__(self, n_classes: int, warm_start: int = 100, refit_every: int = 500) -> None:
        self._classes = list(range(n_classes))
        self._warm = max(1, warm_start)
        self._refit_every = refit_every
        self._X: List[List[float]] = []
        self._y: List[Any] = []
        self._step = 0
        self._model = None

    def _build(self):
        return LogisticRegression(max_iter=300, C=1.0)

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        self._step += 1
        self._X.append(list(x))
        self._y.append(y)
        self._maybe_refit()

    def _maybe_refit(self) -> None:
        raise NotImplementedError

    def predict_one(self, x: List[float]) -> Any:
        if self._model is None:
            return self._classes[0]
        X = np.asarray([x], dtype=float)
        return int(self._model.predict(X)[0])

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        if self._model is None:
            return {c: 1.0 / len(self._classes) for c in self._classes}
        X = np.asarray([x], dtype=float)
        proba = np.asarray(self._model.predict_proba(X)[0], dtype=float)
        proba = proba / proba.sum()
        return {c: float(proba[i]) for i, c in enumerate(self._classes)}

    def classes(self) -> List[Any]:
        return list(self._classes)

    def reset(self) -> None:
        self._X = []
        self._y = []
        self._step = 0
        self._model = None


class StaticBaseline(_BatchBase):
    name = "static_baseline"

    def _maybe_refit(self) -> None:
        if self._model is None and len(self._X) >= self._warm:
            self._model = self._build()
            self._model.fit(np.asarray(self._X, dtype=float), np.asarray(self._y))


class PeriodicBatch(_BatchBase):
    name = "periodic_batch"

    def _maybe_refit(self) -> None:
        if self._model is None and len(self._X) >= self._warm:
            self._model = self._build()
            self._model.fit(np.asarray(self._X, dtype=float), np.asarray(self._y))
        elif self._model is not None and self._step % self._refit_every == 0:
            self._model = self._build()
            self._model.fit(np.asarray(self._X, dtype=float), np.asarray(self._y))
