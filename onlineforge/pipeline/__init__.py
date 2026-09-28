"""pipeline 模块聚合。"""

from .flagship import DAERFlagship
from .streaming_pipeline import benchmark, run

__all__ = ["DAERFlagship", "run", "benchmark"]
