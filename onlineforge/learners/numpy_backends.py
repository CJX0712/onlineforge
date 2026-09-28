"""纯 numpy 在线学习器（离线兜底，零外部依赖）。

实现：多类感知机 (Crammer-Singer)、在线高斯朴素贝叶斯、被动-攻击 (PA-I)。
均支持逐步 partial_fit_one 与 predict_one/predict_proba_one。
未见过任何样本时 predict 返回默认类 0，保证 test-then-train 范式首步不崩溃。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np


class OnlinePerceptron:
    """多类感知机（Crammer-Singer）。"""

    name = "numpy_perceptron"
    backend = "numpy"

    def __init__(self) -> None:
        self.W: Dict[Any, np.ndarray] = {}
        self._order: List[Any] = []

    def _ensure_class(self, y: Any, n: int) -> None:
        if y not in self.W:
            self.W[y] = np.zeros(n)
            self._order.append(y)

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        xv = np.asarray(x, dtype=float)
        self._ensure_class(y, len(xv))
        scores = {c: float(self.W[c] @ xv) for c in self._order}
        y_hat = max(scores, key=scores.get)
        if y_hat != y:
            self.W[y] += xv
            self.W[y_hat] -= xv

    def predict_one(self, x: List[float]) -> Any:
        if not self._order:
            return 0
        xv = np.asarray(x, dtype=float)
        scores = {c: float(self.W[c] @ xv) for c in self._order}
        return max(scores, key=scores.get)

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        if not self._order:
            return {0: 1.0}
        xv = np.asarray(x, dtype=float)
        scores = np.array([float(self.W[c] @ xv) for c in self._order], dtype=float)
        scores -= scores.max()
        e = np.exp(scores)
        p = e / e.sum()
        return {c: float(p[i]) for i, c in enumerate(self._order)}

    def classes(self) -> List[Any]:
        return list(self._order)

    def reset(self) -> None:
        self.W = {}
        self._order = []


class OnlineGaussianNB:
    """在线高斯朴素贝叶斯（Welford 风格增量统计）。"""

    name = "numpy_gaussian_nb"
    backend = "numpy"

    def __init__(self, var_smoothing: float = 1e-3) -> None:
        self.var_smoothing = var_smoothing
        self._order: List[Any] = []
        self.n: Dict[Any, int] = {}
        self.sum: Dict[Any, np.ndarray] = {}
        self.sumsq: Dict[Any, np.ndarray] = {}

    def _ensure_class(self, y: Any, n: int) -> None:
        if y not in self._order:
            self._order.append(y)
            self.n[y] = 0
            self.sum[y] = np.zeros(n)
            self.sumsq[y] = np.zeros(n)

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        xv = np.asarray(x, dtype=float)
        self._ensure_class(y, len(xv))
        self.n[y] += 1
        self.sum[y] += xv
        self.sumsq[y] += xv * xv

    def _stats(self, y: Any):
        n = self.n[y]
        mean = self.sum[y] / n
        var = self.sumsq[y] / n - mean * mean
        var = np.maximum(var, self.var_smoothing)
        return mean, var

    def predict_one(self, x: List[float]) -> Any:
        if not self._order:
            return 0
        xv = np.asarray(x, dtype=float)
        best, best_ll = None, -1e18
        total = sum(self.n.values())
        for c in self._order:
            mean, var = self._stats(c)
            ll = math.log(self.n[c] / total) - 0.5 * np.sum(np.log(2 * math.pi * var))
            ll += -0.5 * np.sum((xv - mean) ** 2 / var)
            if ll > best_ll:
                best_ll, best = ll, c
        return best

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        if not self._order:
            return {0: 1.0}
        xv = np.asarray(x, dtype=float)
        total = sum(self.n.values())
        lls = []
        for c in self._order:
            mean, var = self._stats(c)
            ll = math.log(self.n[c] / total) - 0.5 * np.sum(np.log(2 * math.pi * var))
            ll += -0.5 * np.sum((xv - mean) ** 2 / var)
            lls.append(ll)
        lls = np.array(lls, dtype=float)
        lls -= lls.max()
        e = np.exp(lls)
        p = e / e.sum()
        return {c: float(p[i]) for i, c in enumerate(self._order)}

    def classes(self) -> List[Any]:
        return list(self._order)

    def reset(self) -> None:
        self._order = []
        self.n = {}
        self.sum = {}
        self.sumsq = {}


class OnlinePA:
    """多类被动-攻击学习器 (PA-I, Crammer-Singer)。"""

    name = "numpy_pa"
    backend = "numpy"

    def __init__(self, C: float = 1.0) -> None:
        self.C = C
        self.W: Dict[Any, np.ndarray] = {}
        self._order: List[Any] = []

    def _ensure_class(self, y: Any, n: int) -> None:
        if y not in self.W:
            self.W[y] = np.zeros(n)
            self._order.append(y)

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        xv = np.asarray(x, dtype=float)
        self._ensure_class(y, len(xv))
        scores = {c: float(self.W[c] @ xv) for c in self._order}
        y_hat = max(scores, key=scores.get)
        loss = max(0.0, 1.0 - (scores[y] - scores[y_hat]))
        if loss > 0:
            norm2 = float(xv @ xv) + 1e-12
            tau = loss / (2.0 * norm2)
            self.W[y] += tau * xv
            self.W[y_hat] -= tau * xv

    def predict_one(self, x: List[float]) -> Any:
        if not self._order:
            return 0
        xv = np.asarray(x, dtype=float)
        scores = {c: float(self.W[c] @ xv) for c in self._order}
        return max(scores, key=scores.get)

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        if not self._order:
            return {0: 1.0}
        xv = np.asarray(x, dtype=float)
        scores = np.array([float(self.W[c] @ xv) for c in self._order], dtype=float)
        scores -= scores.max()
        e = np.exp(scores)
        p = e / e.sum()
        return {c: float(p[i]) for i, c in enumerate(self._order)}

    def classes(self) -> List[Any]:
        return list(self._order)

    def reset(self) -> None:
        self.W = {}
        self._order = []
