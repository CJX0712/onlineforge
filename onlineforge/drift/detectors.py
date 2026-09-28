"""概念漂移检测器。

- ADWIN（纯 numpy 实现，离线兜底）：基于滑动窗口内两段子窗均值的 Hoeffding 界差异。
- river 检测器（可选 SOTA）：ADWIN / DDM / KSWIN / PageHinkley。
"""
from __future__ import annotations

import math
from typing import List

RIVER_DRIFT_AVAILABLE = False
try:
    from river import drift as _rd

    RIVER_DRIFT_AVAILABLE = True
except Exception:  # pragma: no cover
    RIVER_DRIFT_AVAILABLE = False


def available_river_drift() -> bool:
    return RIVER_DRIFT_AVAILABLE


class ADWIN:
    """纯 numpy ADWIN 近似实现。

    不变量（可被单测交叉验证）：在错误率由 0.1 突变为 0.9 的合成序列上，
    检测器应在有限步内 firing（检测到漂移）。
    """

    name = "adwin_numpy"

    def __init__(self, delta: float = 0.05, max_len: int = 2000) -> None:
        self.delta = delta
        self.max_len = max_len
        self.window: List[float] = []

    def update(self, value: float) -> bool:
        self.window.append(float(value))
        if len(self.window) > self.max_len:
            self.window.pop(0)
        return self._detect()

    def _detect(self) -> bool:
        w = self.window
        n = len(w)
        if n < 10:
            return False
        prefix = [0.0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + w[i]
        delta = self.delta
        for k in range(1, n):
            n0, n1 = k, n - k
            mean0 = prefix[k] / n0
            mean1 = (prefix[n] - prefix[k]) / n1
            eps = math.sqrt(
                (1.0 / (2.0 * delta)) * (1.0 / n0 + 1.0 / n1) * math.log(2.0 / delta)
            )
            if abs(mean0 - mean1) > eps:
                return True
        return False

    def recent_mean(self) -> float:
        if not self.window:
            return 0.5
        return sum(self.window) / len(self.window)

    def reset(self) -> None:
        self.window = []


class RiverDrift:
    """统一包装 river 的漂移检测器（update 返回是否漂移）。"""

    backend = "river"

    def __init__(self, kind: str = "adwin", **kw) -> None:
        self.name = f"river_{kind}"
        if kind == "adwin":
            self._det = _rd.ADWIN(**kw)
        elif kind == "ddm":
            self._det = _rd.DDM(**kw)
        elif kind == "kswin":
            self._det = _rd.KSWIN(**kw)
        elif kind == "pagehinkley":
            self._det = _rd.PageHinkley(**kw)
        else:
            raise ValueError(f"unknown river drift kind: {kind}")
        self._fired = False

    def update(self, value: float) -> bool:
        # river 的 update 返回 bool 表示本步是否漂移
        self._fired = bool(self._det.update(value))
        return self._fired

    def reset(self) -> None:
        # 重建检测器以清空状态
        kind = self.name.split("_", 1)[1]
        self.__init__(kind)  # type: ignore[call-arg]


def build_river_drifts(delta: float = 0.05) -> List:
    if not RIVER_DRIFT_AVAILABLE:
        return []
    return [
        RiverDrift("adwin", delta=delta),
        RiverDrift("ddm"),
        RiverDrift("kswin"),
        RiverDrift("pagehinkley"),
    ]
