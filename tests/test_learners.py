import numpy as np

from onlineforge.data.generators import make_stream
from onlineforge.learners.numpy_backends import OnlineGaussianNB, OnlinePA, OnlinePerceptron
from onlineforge.learners.sklearn_backends import SklearnPA, SklearnSGD


def _train_then_test(L, s):
    L.reset()
    mid = len(s) // 2
    for e in s[:mid]:
        L.partial_fit_one(e.x, e.y)
    correct = sum(1 for e in s[mid:] if L.predict_one(e.x) == e.y)
    return correct / len(s[mid:])


def _easy(seed=11):
    return make_stream("none", 600, 2, 3, seed=seed, noise=0.6)


def test_numpy_learners_learn_stationary():
    for L in [OnlinePerceptron(), OnlineGaussianNB(), OnlinePA()]:
        acc = _train_then_test(L, _easy())
        assert acc > 0.55, f"{L.name} acc={acc}"


def test_predict_proba_sums_to_one():
    L = OnlineGaussianNB()
    s = _easy()
    L.reset()
    for e in s[:50]:
        L.partial_fit_one(e.x, e.y)
    p = L.predict_proba_one(s[60].x)
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_sklearn_partial_fit():
    for L in [SklearnSGD(3), SklearnPA(3)]:
        acc = _train_then_test(L, _easy(seed=5))
        assert acc > 0.55, f"{L.name} acc={acc}"
