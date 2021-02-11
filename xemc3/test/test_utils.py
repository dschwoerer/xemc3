from xemc3.core import utils
import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as strat


@given(strat.lists(strat.integers(min_value=1, max_value=1e4)))
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
