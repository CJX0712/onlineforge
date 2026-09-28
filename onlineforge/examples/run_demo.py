"""端到端演示：在合成漂移流上对比在线学习器与 DAER 旗舰，落盘 benchmark.json。

运行：
    python onlineforge/examples/run_demo.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..cli import build_default_experiment
from ..core.config import Config
from ..pipeline import benchmark


def main() -> dict:
    cfg = Config.from_env()
    streams, all_learners, flagship = build_default_experiment(cfg, n_samples=2500)
    t0 = time.time()
    bench = benchmark(streams, all_learners, cfg)
    elapsed = time.time() - t0

    here = Path(__file__).resolve().parent
    out = here / "benchmark.json"
    payload = {
        "system": "OnlineForge",
        "version": "0.1.0",
        "author": "晨星",
        "config": cfg.as_dict(),
        "elapsed_s": round(elapsed, 3),
        "benchmark": bench.as_dict(),
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[demo] benchmark 完成，用时 {elapsed:.2f}s，结果已写入 {out}")
    return payload


if __name__ == "__main__":
    main()
