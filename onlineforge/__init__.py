"""OnlineForge — 流式/增量学习系统。

复用顶级开源：river（HoeffdingTree / AdaptiveRandomForest / ADWIN）+ scikit-learn
(partial_fit 在线学习器)。离线兜底为纯 numpy 实现（Perceptron / GaussianNB / PA）
与纯 numpy ADWIN 漂移检测器，保证零下载可运行。

作者：晨星 (CJX0712)
"""

from . import core, data, drift, learners, pipeline
from . import eval as eval_mod

__version__ = "0.1.0"
__author__ = "晨星"

__all__ = ["core", "data", "learners", "drift", "eval_mod", "pipeline"]
