"""全局配置：支持 ENV_ONLINEFORGE_* 环境变量覆盖。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Config:
    """系统级配置。

    环境变量覆盖规则：ENV_ONLINEFORGE_<KEY>=value 会覆盖同名字段。
    例如 ONLINEFORGE_DRIFT_DELTA=0.03 覆盖 drift_delta。
    """

    random_seed: int = 42
    drift_delta: float = 0.05
    warm_start: int = 100
    window_size: int = 200  # 滑窗评测窗口
    recovery_window: int = 400  # 漂移恢复判定窗口
    adwin_max_len: int = 1000  # ADWIN 内部窗口上限
    refit_every: int = 500  # PeriodicBatch 重拟合间隔
    prefer_river: bool = True  # 探测到 river 时优先使用

    _ENV_PREFIX = "ONLINEFORGE_"

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls()
        for key in list(cfg.__dataclass_fields__.keys()):  # type: ignore[attr-defined]
            env_key = cls._ENV_PREFIX + key.upper()
            val = os.environ.get(env_key)
            if val is None:
                continue
            target = cfg.__dataclass_fields__[key]  # type: ignore[attr-defined]
            cfg.__dict__[key] = _coerce(val, target.type)
        return cfg

    def as_dict(self) -> Dict[str, Any]:
        return {
            "random_seed": self.random_seed,
            "drift_delta": self.drift_delta,
            "warm_start": self.warm_start,
            "window_size": self.window_size,
            "recovery_window": self.recovery_window,
            "adwin_max_len": self.adwin_max_len,
            "refit_every": self.refit_every,
            "prefer_river": self.prefer_river,
        }


def _coerce(val: str, typ: Any) -> Any:
    if typ is bool or typ is type(True):
        return val.strip().lower() in ("1", "true", "yes", "on")
    if typ is int:
        return int(val)
    if typ is float:
        return float(val)
    return val
