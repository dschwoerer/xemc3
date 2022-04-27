import copy
import os
import runpy
import sys
import tempfile
import warnings

import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

from .. import write
from . import gen_ds as g
from .test_write_load import assert_ds_are_equal


def call(cmd: str) -> None:
    _argv = copy.deepcopy(sys.argv)
    args = cmd.split()
    sys.argv = args
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "'xemc3.cli.append_time' found in sys.modules after import of package 'xemc3.cli', but prior to execut",
        )
        runpy.run_module("xemc3.cli." + args[0], run_name="__main__")
    sys.argv = _argv


@settings(**g.setting)  # type: ignore
@given(
    g.hypo_shape(10),
    g.hypo_vars12(),
    st.integers(min_value=1, max_value=3),
)
def test_append_ds(shape, v12, rep):
    do_test_append_ds(shape, v12, rep)


def do_test_append_ds(shape, v12, rep):
    v1, v2 = v12
    assert len(v2)
    orgs = [g.gen_rand(shape, v1)]
    out = "temp"
    for i in range(rep):
        orgs.append(g.gen_updated(orgs[0], v2))
    with tempfile.TemporaryDirectory() as dir:
        dir = dir + "/" + out
        os.mkdir(dir)
        for org in orgs:
            write.fortran(org, dir)
            call("append_time " + dir + "///")
        nc = dir + ".nc"
        with xr.open_dataset(nc) as read:
            if "time" not in read.dims:
                assert_ds_are_equal(orgs[0], read, True, 1e-2, 1e-2)
            else:
                for i, org in enumerate(orgs):
                    assert_ds_are_equal(org, read.isel(time=i), True, 1e-2, 1e-2)


@settings(**g.setting)  # type: ignore
@given(
    g.hypo_shape(30),
    g.hypo_vars(),
    st.integers(min_value=1, max_value=3),
)
def test_to_netcdf_ds(shape, var, rep):
    do_to_netcdf_ds(shape, var, rep)


def do_to_netcdf_ds(shape, var, rep):
    out = "test"
    with tempfile.TemporaryDirectory() as dir:
        dir = dir + "/" + out
        os.mkdir(dir)
        for i in range(rep):
            org = g.gen_rand(shape, var)
            write.fortran(org, dir)
            call("to_netcdf " + dir + "///")
            nc = dir + ".nc"
            read = xr.open_dataset(nc)
            assert_ds_are_equal(org, read, True, 1e-2, 1e-2)
            del read
        os.remove(nc)


if __name__ == "__main__":
    do_test_append_ds((2, 2, 2), ([["ENERGY_INFO"]], [["ENERGY_INFO"]]), 1)
    # do_to_netcdf_ds((1, 1, 1), [["ENERGY_INFO"]], 1)
