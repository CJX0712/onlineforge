import random

from onlineforge.drift.detectors import ADWIN


def test_adwin_fires_on_concept_jump():
    d = ADWIN(delta=0.05, max_len=500)
    rng = random.Random(0)
    # 平稳期：正确率 0.9
    for _ in range(120):
        d.update(1.0 if rng.random() < 0.9 else 0.0)
    # 概念漂移：正确率跌到 0.1
    fired = False
    for _ in range(300):
        if d.update(1.0 if rng.random() < 0.1 else 0.0):
            fired = True
            break
    assert fired, "ADWIN 应在错误率突变后 firing"


def test_adwin_quiet_on_stationary():
    d = ADWIN()
    rng = random.Random(1)
    fired = False
    for _ in range(300):
        if d.update(1.0 if rng.random() < 0.9 else 0.0):
            fired = True
    # 平稳流不应触发漂移
    assert not fired
    assert 0.0 <= d.recent_mean() <= 1.0
