"""OnlineForge CLI 入口。

用法：
    python -m onlineforge.cli            # 跑默认基准，打印表 + 写 benchmark.json
    python -m onlineforge.cli --n 4000   # 指定流长度
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from .core.config import Config
from .data.generators import make_stream
from .learners import RIVER_AVAILABLE, build_learners
from .learners.numpy_backends import OnlineGaussianNB, OnlinePA, OnlinePerceptron
from .learners.sklearn_backends import SklearnPA, SklearnSGD
from .pipeline import DAERFlagship, benchmark


def build_default_experiment(config: Config | None = None, n_samples: int = 4000):
    cfg = config or Config()
    streams: Dict[str, List] = {
        "stationary": make_stream(
            "none", n_samples=n_samples, n_features=2, n_classes=3, seed=1
        ),
        "abrupt": make_stream("abrupt", n_samples=n_samples, n_features=2, n_classes=3, seed=2),
        "gradual": make_stream(
            "gradual", n_samples=n_samples, n_features=2, n_classes=3, seed=3
        ),
        "incremental": make_stream(
            "incremental", n_samples=n_samples, n_features=2, n_classes=3, seed=4
        ),
    }
    n_classes = 3
    learners = build_learners(n_classes, cfg)
    # 旗舰 DAER 包裹真正的在线学习器（不含批式基线）
    online = [
        OnlinePerceptron(),
        OnlineGaussianNB(),
        OnlinePA(),
        SklearnSGD(n_classes),
        SklearnPA(n_classes),
    ]
    if RIVER_AVAILABLE:
        from .learners.river_backends import build_river_learners

        online.extend(build_river_learners())
    flagship = DAERFlagship(online, delta=cfg.drift_delta, window=cfg.window_size)
    all_learners = learners + [flagship]
    return streams, all_learners, flagship


def _print_table(bench) -> None:
    rows = bench.results
    print(
        f"\n{'learner':<20}{'dataset':<14}{'preq_acc':>9}{'win_acc':>9}{'drift':>7}{'reclag':>8}{'rt_s':>8}{'backend':>10}"
    )
    print("-" * 85)
    for r in rows:
        print(
            f"{r.learner:<20}{r.dataset:<14}{r.prequential_acc:>9.4f}{r.windowed_acc:>9.4f}"
            f"{r.drifts_detected:>7}{r.recovery_lag:>8.1f}{r.runtime_s:>8.2f}{r.backend:>10}"
        )


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="onlineforge", description="OnlineForge 流式学习基准")
    p.add_argument("--n", type=int, default=2500, help="每条流样本数")
    p.add_argument("--out", type=str, default="benchmark.json", help="输出 JSON 路径")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    cfg.random_seed = args.seed
    streams, all_learners, flagship = build_default_experiment(cfg, n_samples=args.n)

    print(
        "OnlineForge 基准开始 | 流: stationary/abrupt/gradual/incremental | "
        f"n_classes=3 | n_samples/stream={args.n}"
    )
    print(f"river SOTA backend: {'启用' if RIVER_AVAILABLE else '降级(纯 numpy/sklearn)'}")

    bench = benchmark(streams, all_learners, cfg)
    _print_table(bench)

    # 旗舰 vs 各基线：聚合前缀准确率
    agg: Dict[str, float] = {}
    for r in bench.results:
        agg.setdefault(r.learner, []).append(r.prequential_acc)
    print("\n聚合前缀准确率 (4 流均值):")
    for name, vals in sorted(agg.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {name:<20} {sum(vals) / len(vals):.4f}")

    out_path = Path(args.out)
    payload = {
        "config": cfg.as_dict(),
        "river_available": RIVER_AVAILABLE,
        "benchmark": bench.as_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入 {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
