"""river SOTA 在线学习器（可选后端）。

river 0.26 中：
- 决策树为 river.tree.HoeffdingTreeClassifier（无 seed 参数）。
- 自适应树（内置漂移检测）为 river.tree.HoeffdingAdaptiveTreeClassifier。
- 集成替代原 AdaptiveRandomForest：river.ensemble.LeveragingBaggingClassifier（默认基学习器即 HoeffdingTree）。
- 在线线性模型：river.linear_model.LogisticRegression。

本模块在 river 不可用时整体跳过，不影响系统其它部分。
"""

from __future__ import annotations

from typing import Any, Dict, List

RIVER_AVAILABLE = False
try:
    from river import ensemble, linear_model, tree

    RIVER_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖缺失时静默降级
    RIVER_AVAILABLE = False


def available_river() -> bool:
    return RIVER_AVAILABLE


def _to_dict(x: List[float]) -> Dict[str, float]:
    return {f"f{i}": float(v) for i, v in enumerate(x)}


class _RiverWrapper:
    """统一包装 river 的 learn_one / predict_one / predict_proba_one。"""

    backend = "river"

    def __init__(self, model) -> None:
        self._model = model
        self._classes: List[Any] = []

    def partial_fit_one(self, x: List[float], y: Any) -> None:
        self._model.learn_one(_to_dict(x), y)
        if y not in self._classes:
            self._classes.append(y)

    def predict_one(self, x: List[float]) -> Any:
        p = self._model.predict_one(_to_dict(x))
        if p is None:
            return self._classes[0] if self._classes else 0
        return p

    def predict_proba_one(self, x: List[float]) -> Dict[Any, float]:
        proba = self._model.predict_proba_one(_to_dict(x))
        if not proba:
            proba = {c: 1.0 / max(1, len(self._classes)) for c in self._classes}
        s = sum(proba.values()) or 1.0
        return {c: float(v) / s for c, v in proba.items()}

    def classes(self) -> List[Any]:
        return list(self._classes)

    def reset(self) -> None:
        self._model = self._fresh()
        self._classes = []


class RiverHoeffdingTree(_RiverWrapper):
    name = "river_hoeffding_tree"

    def _fresh(self):
        return tree.HoeffdingTreeClassifier()


class RiverHAT(_RiverWrapper):
    """Hoeffding Adaptive Tree：内置 ADWIN 漂移检测，对概念漂移鲁棒。"""

    name = "river_hat"

    def _fresh(self):
        return tree.HoeffdingAdaptiveTreeClassifier()


class RiverLeveragingBag(_RiverWrapper):
    """Leveraging Bagging 集成（ARF 思路的 river 实现）。"""

    name = "river_leveraging_bag"

    def _fresh(self):
        return ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier())


class RiverLogReg(_RiverWrapper):
    name = "river_logreg"

    def _fresh(self):
        return linear_model.LogisticRegression()


def build_river_learners() -> List[_RiverWrapper]:
    if not RIVER_AVAILABLE:
        return []
    return [
        RiverHoeffdingTree(tree.HoeffdingTreeClassifier()),
        RiverHAT(tree.HoeffdingAdaptiveTreeClassifier()),
        RiverLeveragingBag(
            ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier())
        ),
        RiverLogReg(linear_model.LogisticRegression()),
    ]
