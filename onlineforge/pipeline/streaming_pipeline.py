"""流式评测 pipeline：单学习器跑完整条流 + 多学习器/多流基准编排。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..core.config import Config
from ..core.types import Benchmark, BenchmarkResult, StreamExample
from ..drift.detectors import ADWIN
from ..eval.metrics import PrequentialAccuracy, RollingAccuracy, recovery_lag


def run(
    stream: List[StreamExample],
    learner,
    drift_detector=None,
    config: Optional[Config] = None,
) -> Dict[str, Any]:
    """在单条流上运行单学习器（test-then-train 前缀范式）。

    返回聚合指标 + 明细（drift_steps / correct_flags）。
    注意：pipeline 不主动重置学习器；自适应由学习器自身（或旗舰内部）负责。
    drift_detector 仅用于统一统计该学习器误差流上的漂移次数。
    """
    cfg = config or Config()
    if drift_detector is None:
        drift_detector = ADWIN(delta=cfg.drift_delta, max_len=cfg.adwin_max_len)

    preq = PrequentialAccuracy()
    roll = RollingAccuracy(cfg.window_size)
    correct_flags: List[int] = []
    drift_steps: List[int] = []

    t0 = time.time()
    for step, ex in enumerate(stream):
        pred = learner.predict_one(ex.x)
        correct = 1 if pred == ex.y else 0
        preq.update(correct)
        roll.update(correct)
        correct_flags.append(correct)
        if drift_detector.update(correct):
            drift_steps.append(step)
        learner.partial_fit_one(ex.x, ex.y)
    runtime = time.time() - t0

    return {
        "prequential_acc": preq.get(),
        "windowed_acc": roll.get(),
        "drifts_detected": len(drift_steps),
        "recovery_lag": recovery_lag(correct_flags, drift_steps, cfg.recovery_window),
        "runtime_s": runtime,
        "drifts_detected_steps": drift_steps,
        "correct_flags": correct_flags,
    }


def benchmark(
    streams: Dict[str, List[StreamExample]],
    learners: List,
    config: Optional[Config] = None,
) -> Benchmark:
    """在多条流上跑多个学习器，产出 Benchmark。"""
    cfg = config or Config()
    bench = Benchmark()
    for sname, stream in streams.items():
        n = len(stream)
        n_classes = len({e.y for e in stream})
        for learner in learners:
            # 每条流都从全新状态开始
            learner.reset()
            res = run(stream, learner, config=cfg)
            bench.add(
                BenchmarkResult(
                    dataset=sname,
                    learner=learner.name,
                    prequential_acc=res["prequential_acc"],
                    windowed_acc=res["windowed_acc"],
                    drifts_detected=res["drifts_detected"],
                    recovery_lag=res["recovery_lag"],
                    n_classes=n_classes,
                    n_samples=n,
                    runtime_s=res["runtime_s"],
                    backend=getattr(learner, "backend", "unknown"),
                )
            )
    return bench
