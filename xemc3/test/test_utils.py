import hypothesis.strategies as strat
import numpy as np
import xarray as xr
from hypothesis import assume, given

from xemc3.core import utils


@given(strat.lists(strat.integers(min_value=1, max_value=10000)))
def test_rrange(shape):
    assume(utils.prod(shape) < 1e5)
    dat = np.zeros(shape, dtype=int)
    for ijk in utils.rrange(shape):
        dat[ijk] += 1
    assert (dat == 1).all()


@given(strat.lists(strat.integers(min_value=1, max_value=100), min_size=0))
def test_rrange2(shape):
    assume(utils.prod(shape) < 1e4)
    dat = np.zeros(shape, dtype=int)
    print(dat)
    for ijk in utils.rrange2(shape):
        dat[ijk] += 1
    assert (dat == 1).all()


def test_merge_indexers_raise():
    a = "a"
    b = "b"
    v1 = {a: 1, b: 2}
    v2 = {a: 2}
    try:
        utils.merge_indexers(v1, v2)
    except AssertionError:
        pass
    else:
        raise AssertionError


def test_merge_indexers_posion():
    a = "a"
    b = "b"
    v1 = {a: 1, b: 2}

    assert v1 == utils.merge_indexers(v1, {})
    assert v1 == utils.merge_indexers(None, v1)


def test_merge_indexers_combine():
    a = "a"
    b = "b"
    v1 = {a: 1}
    v2 = {b: 2}
    both = {**v1, **v2}
    assert both == utils.merge_indexers(v1, v2)
    assert both == utils.merge_indexers(v2, v1)


def test_interval_round_1d():
    a = xr.DataArray(np.random.random(5), dims="a")
    b = utils.from_interval(utils.to_interval(a))
    assert all(a == b)


def test_interval_round_mismatch():
    a = xr.DataArray(np.random.random(5), dims="a")
    b = utils.to_interval(a)
    b.data[1, 1] += 1
    raised = False
    try:
        utils.from_interval(b)
    except:
        raised = True
    assert raised


def test_timit():
    with utils.timeit():
        pass

    with utils.timeit("pass takes %f seconds"):
        pass


def test_timit2():
    with utils.timeit2():
        pass
