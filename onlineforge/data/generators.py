"""合成流生成器：支持 none / abrupt / gradual / incremental 概念漂移。

不变量（可被单测交叉验证）：
- 同分布 stationary 流下，自适应在线学习器应显著优于 "fit-once" 静态基线不成立
  （二者接近）；但在 abrupt/gradual/incremental 漂移下，静态基线准确率应明显下降，
  自适应学习器能够恢复。
- 同一 (seed, drift_type) 生成的流完全确定可复现。
"""
from __future__ import annotations

import math
from typing import Callable, Iterator, List

import numpy as np

from ..core.types import StreamExample


def _rotation_matrix(angle: float, n: int) -> np.ndarray:
    """构造前 2 维旋转、其余维恒等的旋转矩阵。"""
    r = np.eye(n)
    ca, sa = math.cos(angle), math.sin(angle)
    r[0, 0], r[0, 1] = ca, -sa
    r[1, 0], r[1, 1] = sa, ca
    return r


def _base_centers(n_features: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 5.0, size=(n_classes, n_features))


def make_stream(
    drift_type: str = "abrupt",
    n_samples: int = 4000,
    n_features: int = 2,
    n_classes: int = 3,
    seed: int = 42,
    noise: float = 1.2,
    drift_magnitude: float = 8.0,
    name: str = "",
) -> List[StreamExample]:
    """生成一条确定性流。

    drift_type:
      none         - 中心固定
      abrupt       - 中点处中心整体跳变（突变漂移）
      gradual      - 中点附近窗口内线性过渡
      incremental  - 中心随时间缓慢旋转（渐进漂移）
    """
    if drift_type not in ("none", "abrupt", "gradual", "incremental"):
        raise ValueError(f"unknown drift_type: {drift_type}")

    rng = np.random.default_rng(seed)
    base = _base_centers(n_features, n_classes, seed)
    target = base + rng.normal(0.0, drift_magnitude, size=(n_classes, n_features))

    examples: List[StreamExample] = []
    mid = n_samples // 2
    g_start, g_end = int(n_samples * 0.4), int(n_samples * 0.6)

    for t in range(n_samples):
        frac = t / max(1, n_samples - 1)
        if drift_type == "none":
            centers = base
        elif drift_type == "abrupt":
            centers = base if t < mid else target
        elif drift_type == "gradual":
            alpha = min(1.0, max(0.0, (t - g_start) / max(1, g_end - g_start)))
            centers = (1.0 - alpha) * base + alpha * target
        else:  # incremental
            rot = _rotation_matrix(2 * math.pi * frac, n_features)
            centers = base @ rot.T
        k = int(rng.integers(0, n_classes))
        x = centers[k] + rng.normal(0.0, noise, size=n_features)
        examples.append(StreamExample(x=x.tolist(), y=k))
    return examples


# 真实/准真实静态流（用于对照，无漂移）
_SKLEARN_LOADERS: dict[str, Callable[[], tuple]] = {}


def _lazy_load_sklearn(name: str):
    from sklearn import datasets

    if name == "digits":
        data = datasets.load_digits()
        X, y = data.data, data.target
    elif name == "iris":
        data = datasets.load_iris()
        X, y = data.data, data.target
    else:
        raise ValueError(f"unknown sklearn dataset: {name}")
    return X, y


def from_sklearn(name: str, n_samples: int | None = None, seed: int = 0, shuffle: bool = True) -> List[StreamExample]:
    """把 sklearn 数据集转成一条（无漂移）流。"""
    X, y = _lazy_load_sklearn(name)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if n_samples is None:
        n_samples = len(X)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    idx = idx[:n_samples]
    out = [StreamExample(x=X[i].tolist(), y=int(y[i])) for i in idx]
    return out


def iter_batches(stream: List[StreamExample], batch_size: int = 1) -> Iterator[List[StreamExample]]:
    """按批次产出（默认单样本在线）。"""
    for i in range(0, len(stream), batch_size):
        yield stream[i : i + batch_size]
