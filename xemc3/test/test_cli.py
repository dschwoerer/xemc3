import xemc3
import gen_ds as g
import tempfile
from hypothesis import given, settings, strategies as st
from test_write_load import assert_ds_are_equal
from subprocess import call as call_unsafe
import xarray as xr
import os


def call(cmd: str) -> None:
    out = call_unsafe("python3 xemc3/cli/" + cmd, shell=True)
    if out:
        raise RuntimeError(f"Nonzero return code {out}")


@settings(**g.setting)
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
            xemc3.write.fortran(org, dir)
            call("append_time.py " + dir)
        nc = dir + ".nc"
        read = xr.open_dataset(nc)
        for i, org in enumerate(orgs):
            assert_ds_are_equal(org, read.isel(time=i), True, 1e-2, 1e-2)
        del read


@settings(**g.setting)
@given(
    g.hypo_shape(30),
    g.hypo_vars(),
    st.integers(min_value=1, max_value=3),
)
def test_to_netcdf_ds(shape, var, rep):
    out = "test"
    with tempfile.TemporaryDirectory() as dir:
        dir = dir + "/" + out
        os.mkdir(dir)
        for i in range(rep):
            org = g.gen_rand(shape, var)
            xemc3.write.fortran(org, dir)
            call("to_netcdf.py " + dir)
            nc = dir + ".nc"
            read = xr.open_dataset(nc)
            assert_ds_are_equal(org, read, True, 1e-2, 1e-2)
            del read
        os.remove(nc)


if __name__ == "__main__":
    do_test_append_ds((2, 2, 2), ([["LG_CELL", 1]], [["LG_CELL", 1]]), 1)
