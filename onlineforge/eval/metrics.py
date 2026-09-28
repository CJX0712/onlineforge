"""流式评测指标。"""
from __future__ import annotations

from collections import deque
from typing import Deque, List


class PrequentialAccuracy:
    """累计前缀（prequential）准确率。"""

    name = "prequential_acc"

    def __init__(self) -> None:
        self._correct = 0
        self._total = 0

    def update(self, correct: int) -> None:
        self._correct += correct
        self._total += 1

    def get(self) -> float:
        return self._correct / self._total if self._total else 0.0


class RollingAccuracy:
    """最近 W 步滑窗准确率。"""

    name = "rolling_acc"

    def __init__(self, window: int = 200) -> None:
        self._window: Deque[int] = deque(maxlen=window)
        self._sum = 0

    def update(self, correct: int) -> None:
        if len(self._window) == self._window.maxlen:
            self._sum -= self._window[0]
        self._window.append(correct)
        self._sum += correct

    def get(self) -> float:
        n = len(self._window)
        return self._sum / n if n else 0.0


def cohen_kappa(y_true: List, y_pred: List) -> float:
    """Cohen's kappa（对比多数类基线）。"""
    if not y_true:
        return 0.0
    from collections import Counter

    n = len(y_true)
    maj = Counter(y_true).most_common(1)[0][0]
    po = sum(1 for a, b in zip(y_true, y_pred) if a == b) / n
    pe = (sum(1 for a in y_true if a == maj) / n) * (sum(1 for b in y_pred if b == maj) / n)
    return (po - pe) / (1 - pe) if (1 - pe) else 0.0


def recovery_lag(correct_flags: List[int], drift_steps: List[int], window: int = 400) -> float:
    """漂移后恢复到漂移前水平所需的平均步数。

    correct_flags: 每个时间步是否正确(0/1)。
    drift_steps:   ADWIN 触发步索引。
    对每次漂移，以其前 window/2 步的准确率作为基线，向后找首次恢复（连续 window/2 步
    不低于基线）的步距；找不到则记 window。
    """
    if not drift_steps:
        return 0.0
    n = len(correct_flags)
    half = max(5, window // 2)
    lags: List[float] = []
    for d in drift_steps:
        if d < half:
            lags.append(float(window))
            continue
        base = sum(correct_flags[d - half : d]) / half
        # 向后扫描
        recovered_at = None
        for s in range(d, min(n - half, d + window)):
            if sum(correct_flags[s : s + half]) / half >= base - 1e-6:
                recovered_at = s
                break
        lags.append(float((recovered_at - d) if recovered_at is not None else window))
    return sum(lags) / len(lags)
