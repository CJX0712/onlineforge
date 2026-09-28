"""scikit-learn 在线学习器（partial_fit）。

sklearn 1.9 中 SGDClassifier / PassiveAggressiveClassifier 仍支持 partial_fit，
是工业界主流的在线/增量学习器，作为 mid-tier 后端。
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier


class _SklearnPartial:
    """包装 sklearn 的 partial_fit 分类器。"""

    backend = "sklearn"

    def __init__(self, n_classes: int, **kw) -> None:
        self._classes = list(range(n_classes))
        self._model = self._build(**kw)
        self._fitted = False

    def _build(self, **kw):
        raise NotImplementedError

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        X = np.asarray([x], dtype=float)
        Y = np.asarray([y], dtype=int)
        if not self._fitted:
            self._model.partial_fit(X, Y, classes=self._classes)
            self._fitted = True
        else:
            self._model.partial_fit(X, Y)

    def predict_one(self, x: List[float]) -> Any:
        if not self._fitted:
            return self._classes[0]
        X = np.asarray([x], dtype=float)
        return int(self._model.predict(X)[0])

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        if not self._fitted:
            return {c: 1.0 / len(self._classes) for c in self._classes}
        X = np.asarray([x], dtype=float)
        proba = self._model.predict_proba(X)[0]
        proba = np.asarray(proba, dtype=float)
        if proba.sum() == 0:
            proba = np.full(len(self._classes), 1.0 / len(self._classes))
        proba = proba / proba.sum()
        return {c: float(proba[i]) for i, c in enumerate(self._classes)}

    def classes(self) -> List[Any]:
        return list(self._classes)

    def reset(self) -> None:
        self._model = self._build()
        self._fitted = False


class SklearnSGD(_SklearnPartial):
    name = "sklearn_sgd"

    def _build(self, **kw):
        return SGDClassifier(loss="log_loss", random_state=42, **kw)


class SklearnPA(_SklearnPartial):
    name = "sklearn_pa"

    def _build(self, **kw):
        return PassiveAggressiveClassifier(C=1.0, **kw)
