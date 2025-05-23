import itertools
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import xarray as xr
import warnings

_open_formats: List[Tuple[Callable, str]] = []
try:
    import lzma

    _open_formats += [
        (lzma.open, ".xz"),
        (lzma.open, ".lzma"),
    ]
except ImportError:
    pass
try:
    import gzip

    _open_formats += [(gzip.open, ".gz")]
except ImportError:
    pass


def format_time(secs):
    ret = ""
    out = False
    for name, val in [("m", 60), ("h", 3600), ("d", 60 * 60 * 24)][::-1]:
        if secs >= val:
            out = True
        if out:
            cur = int(secs // val)
            ret += f"{cur:02d}{name}"
            secs -= cur * val
    ret += f"{secs:02.0f}s"
    return ret


class rrange2(object):
    def __init__(self, args, *, update=None):
        if update is not None:
            # This is no forther needed, the update is now based on time.
            warnings.warn(
                "passing the `update` keyword to rrange2 is deprecateed.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        self.args = [int(a) for a in args]
        self.total = 1
        self.t0 = time.time()
        self.tnext = self.t0 + 1
        for a in self.args:
            self.total *= a
        self._has_returned = 0
        return None

    def __iter__(self):
        self.state = [0 for a in self.args]
        if len(self.state) > 0:
            self.state[0] = -1
        self.tstate = -1
        return self

    def __next__(self):
        self.tstate += 1
        self.tnow = time.time()
        if self.tnow >= self.tnext:
            self.tnext = self.tnow + 1
            needed = self.tnow - self.t0
            prog = self.tstate / self.total
            print(
                f"{prog * 100:6.2f} % ... {format_time(needed)}>{format_time(needed / (prog) - needed) if prog else 0}",
                end="\r",
            )
        try:
            self.state[0] += 1
        except IndexError:
            if self.tstate == 0:
                return tuple()
            raise StopIteration
        for i in range(len(self.args)):
            if self.state[i] < self.args[i]:
                return tuple(self.state)
            else:
                self.state[i] = 0
                try:
                    self.state[i + 1] += 1
                except IndexError:
                    print()
                    raise StopIteration


def rrange(args):
    ranges = [range(a) for a in args]
    return itertools.product(*ranges)


def prod(args):
    ret = 1
    for arg in args:
        ret *= arg
    return ret


raise_issue = " If you think the input is valid, please open a bug at https://github.com/dschwoerer/xemc3/issues"


def to_interval(dims, data=None) -> xr.DataArray:
    """Transforms a N-D i_1+1 x ... x i_N+1 mesh to an
    i_1 x ... x i_N x 2 x ... x 2 mesh of quads
    """
    if isinstance(dims, tuple) and data is None:
        dims, data = dims

    if data is None:
        data = dims
        in_dims = data.dims
        attrs = data.attrs
        data = data.data
    else:
        in_dims = dims
        attrs = {}
    dims = len(data.shape)
    assert dims == len(
        in_dims
    ), f"Data mismatch - {in_dims} as dimensions given, but data is shape {data.shape} (len{len(data.shape)})"
    # Once we drop python3.8, replace by removesuffix
    in_dims = [x[: -len("_plus1")] if x.endswith("_plus1") else x for x in in_dims]
    out_dims = [d for d in in_dims] + ["delta_" + d for d in in_dims]
    ret = np.empty([i - 1 for i in data.shape] + [2] * dims, dtype=data.dtype)
    for ijk in itertools.product(*[range(a - 1) for a in data.shape]):
        tmp = np.array(data[tuple([slice(i, i + 2) for i in ijk])])
        ret[ijk] = tmp
    return xr.DataArray(data=ret, dims=out_dims, attrs=attrs)


def from_interval(data, check=True):
    dims = len(data.shape) // 2
    out_dims = data.dims[:dims]
    attrs = data.attrs
    data = data.data
    ret = np.empty([i + 1 for i in data.shape[:dims]])
    # if dims == 1:
    #     ret[:-1] = data[:, 0]
    #     ret[-1] = data[-1, 1]
    # elif dims == 2:
    #     ret[:-1, :-1] = data[:, :, 0, 0]
    #     ret[-1, :-1] = data[-1, :, 1, 0]
    #     ret[:-1, -1] = data[:, -1, 0, 1]
    #     ret[-1, -1] = data[-1, -1, 1, 1]
    for ijk in itertools.product(*[[0, 1]] * dims):
        first = tuple([[slice(None, -1), -1][i] for i in ijk])
        second = tuple([[slice(None), -1][i] for i in ijk] + [i for i in ijk])
        ret[first] = data[second]
    if check:
        for ijk in itertools.product(*[[0, 1]] * dims):
            first = tuple([[slice(None, -1), slice(1, None)][i] for i in ijk])
            second = tuple([slice(None)] * dims + [i for i in ijk])
            assert (ret[first] == data[second]).all()
    return xr.DataArray(data=ret, dims=out_dims, attrs=attrs)


class timeit:
    def __init__(self, info="%f"):
        self.info = info

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        print(self.info % (time.time() - self.t0))


T0 = time.time()
last = time.time()


class timeit2:
    def __init__(self, info="%f to %f - %f vs %f"):
        self.info = info

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, *args):
        t0 = self.t0
        t1 = time.time()
        global last
        print(self.info % (t0 - T0, t1 - T0, t1 - t0, t0 - last))
        last = t1


def merge_indexers(
    indexers: Optional[Mapping[str, Any]], indexers_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if indexers is None:
        return indexers_kwargs.copy()
    for k in indexers.keys():
        if k in indexers_kwargs:
            raise AssertionError(
                f"{k} is provided as kwargs and in indexers - only provide the key once"
            )
    indexers_kwargs = indexers_kwargs.copy()
    indexers_kwargs.update(indexers)
    return indexers_kwargs


def _fft(data):
    from scipy.fftpack import fft  # type: ignore

    fr = fft(data)
    import matplotlib.pyplot as plt  # type: ignore

    plt.plot(np.abs(fr))
    # print(abs(fr.)
    plt.show()
    exit(1)


_open_formats = []
_org_open = open


def open(fn, mode="rt", *args):
    if fn[0] == "~":
        fn = os.environ["HOME"] + fn[1:]
    if mode in ["r", "rb", "rt"]:
        try:
            return _org_open(fn, mode, *args)
        except FileNotFoundError as e:
            for func, ext in _open_formats:
                try:
                    return func(fn + ext, mode, *args)
                except FileNotFoundError:
                    pass
            raise e
    return _org_open(fn, mode, *args)
