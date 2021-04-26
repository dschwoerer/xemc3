import xemc3
import os
import pytest
from xarray.testing import assert_identical
import xarray as xr


def get_data():
    basedir = "xemc3/test/testdata/"
    if not os.path.isdir(basedir):
        pytest.skip("create {basedir} to enable testing on real data")
        return
    if not os.path.isdir(basedir + ".git"):
        os.system(
            f"git clone https://gitlab.mpcdf.mpg.de/dave/xemc3-data/ {basedir} --depth 1"
        )
    else:
        os.system(f"cd {basedir}; git fetch origin main --depth 1")
        os.system(f"cd {basedir}; git checkout origin next")
    return basedir + "emc3_example"


# simple regression test
def test_load_all():
    bd = get_data()
    result = xemc3.load.all(bd)
    expected = xr.open_dataset(bd + ".nc")
    assert_identical(result, expected)


# simple regression test
def test_load_plates():
    bd = get_data()
    result = xemc3.load.plates(bd, cache=False)
    expected = xemc3.load.plates(bd, cache=True)
    assert_identical(result, expected)


def test_iter_plates():
    bd = get_data()
    ds = xemc3.load.plates(bd)
    for _ in ds.emc3.iter_plates():
        pass


def test_get_element():
    bd = get_data()
    ds = xemc3.load.plates(bd)
    fe = ds.emc3["f_E"]
    assert isinstance(fe, list)
    for fel in fe:
        assert isinstance(fel, xr.DataArray)
        assert fel.attrs == ds["f_E"].attrs
