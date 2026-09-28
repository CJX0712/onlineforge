from onlineforge.data.generators import make_stream


def test_deterministic_same_seed():
    a = make_stream("abrupt", 500, 2, 3, seed=7)
    b = make_stream("abrupt", 500, 2, 3, seed=7)
    assert [e.y for e in a] == [e.y for e in b]
    assert all(abs(a[i].x[j] - b[i].x[j]) < 1e-12 for i in range(500) for j in range(2))


def test_abrupt_differs_from_stationary():
    ab = make_stream("abrupt", 500, 2, 3, seed=7)
    st = make_stream("none", 500, 2, 3, seed=7)
    assert any(abs(ab[i].x[0] - st[i].x[0]) > 1e-6 for i in range(250, 500))


def test_drift_types_have_same_length_and_classes():
    for dt in ("none", "abrupt", "gradual", "incremental"):
        s = make_stream(dt, 300, 2, 3, seed=3)
        assert len(s) == 300
        assert set(e.y for e in s) == {0, 1, 2}
