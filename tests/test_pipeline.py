from onlineforge.core.config import Config
from onlineforge.data.generators import make_stream
from onlineforge.learners import build_learners
from onlineforge.learners.numpy_backends import OnlineGaussianNB, OnlinePA, OnlinePerceptron
from onlineforge.learners.sklearn_backends import SklearnPA, SklearnSGD
from onlineforge.pipeline import DAERFlagship, benchmark, run


def test_run_stationary_high():
    s = make_stream("none", 600, 2, 3, seed=9, noise=0.6)
    L = OnlineGaussianNB()
    L.reset()
    r = run(s, L)
    assert r["prequential_acc"] > 0.6
    assert r["drifts_detected"] >= 0


def test_benchmark_runs_all_learners():
    cfg = Config()
    streams = {
        "ab": make_stream("abrupt", 400, 2, 3, seed=2),
        "st": make_stream("none", 400, 2, 3, seed=1),
    }
    learners = build_learners(3, cfg)
    online = [OnlinePerceptron(), OnlineGaussianNB(), OnlinePA(), SklearnSGD(3), SklearnPA(3)]
    flag = DAERFlagship(online)
    b = benchmark(streams, learners + [flag], cfg)
    assert len(b.results) == len(streams) * (len(learners) + 1)


def test_flagship_predict_consistent_classes():
    online = [OnlinePerceptron(), OnlineGaussianNB(), OnlinePA()]
    flag = DAERFlagship(online)
    s = make_stream("abrupt", 400, 2, 3, seed=2)
    for e in s[:50]:
        flag.partial_fit_one(e.x, e.y)
    pred = flag.predict_one(s[60].x)
    assert pred in {0, 1, 2}
