"""学习器模块聚合 + 注册器。"""

from __future__ import annotations

from typing import List

from ..core.config import Config
from .batch_baseline import PeriodicBatch, StaticBaseline
from .numpy_backends import OnlineGaussianNB, OnlinePA, OnlinePerceptron
from .sklearn_backends import SklearnPA, SklearnSGD

try:
    from .river_backends import RIVER_AVAILABLE, build_river_learners
except Exception:  # pragma: no cover
    RIVER_AVAILABLE = False

    def build_river_learners():
        return []


__all__ = [
    "OnlinePerceptron",
    "OnlineGaussianNB",
    "OnlinePA",
    "SklearnSGD",
    "SklearnPA",
    "StaticBaseline",
    "PeriodicBatch",
    "RIVER_AVAILABLE",
    "build_river_learners",
    "build_learners",
]


def build_learners(n_classes: int, config: Config | None = None) -> List:
    """按配置组装默认学习器清单（numpy 兜底 + sklearn + river SOTA）。"""
    cfg = config or Config()
    learners = [
        OnlinePerceptron(),
        OnlineGaussianNB(),
        OnlinePA(),
        SklearnSGD(n_classes),
        SklearnPA(n_classes),
        StaticBaseline(n_classes, warm_start=cfg.warm_start),
        PeriodicBatch(n_classes, warm_start=cfg.warm_start, refit_every=cfg.refit_every),
    ]
    if cfg.prefer_river and RIVER_AVAILABLE:
        learners.extend(build_river_learners())
    return learners
