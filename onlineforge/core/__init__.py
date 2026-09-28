"""OnlineForge core：类型、错误码、配置、接口契约。"""

from .config import Config
from .errors import E100_BAD_INPUT, E200_FIT, E300_PREDICT, E400_DRIFT, E500_CONFIG, ForgeError
from .interfaces import DriftDetector, StreamLearner, StreamMetric
from .types import BenchmarkResult, DatasetSpec, DriftSignal, EvalRecord, StreamExample

__all__ = [
    "StreamExample",
    "DriftSignal",
    "EvalRecord",
    "BenchmarkResult",
    "DatasetSpec",
    "ForgeError",
    "E100_BAD_INPUT",
    "E200_FIT",
    "E300_PREDICT",
    "E400_DRIFT",
    "E500_CONFIG",
    "Config",
    "StreamLearner",
    "DriftDetector",
    "StreamMetric",
]
