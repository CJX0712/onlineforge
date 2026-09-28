"""OnlineForge core：类型、错误码、配置、接口契约。"""
from .types import StreamExample, DriftSignal, EvalRecord, BenchmarkResult, DatasetSpec
from .errors import ForgeError, E100_BAD_INPUT, E200_FIT, E300_PREDICT, E400_DRIFT, E500_CONFIG
from .config import Config
from .interfaces import StreamLearner, DriftDetector, StreamMetric

__all__ = [
    "StreamExample", "DriftSignal", "EvalRecord", "BenchmarkResult", "DatasetSpec",
    "ForgeError", "E100_BAD_INPUT", "E200_FIT", "E300_PREDICT", "E400_DRIFT", "E500_CONFIG",
    "Config", "StreamLearner", "DriftDetector", "StreamMetric",
]
