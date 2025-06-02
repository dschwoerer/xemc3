import os


def get_data(force=False):
    basedir = "./example-data/"
    if not os.path.isdir(basedir) and not force:
        pytest.skip("create {basedir} to enable testing on real data")
        return
    if not os.path.isdir(basedir + ".git"):
        os.system(
            f"git clone https://oauth2:glpat-gS1qiEX7Ncoys5xH3CfS@gitlab.mpcdf.mpg.de/dave/xemc3-data/ {basedir} --depth 1"
        )
    else:
        os.system(f"cd {basedir}; git fetch origin main --depth 1")
        os.system(f"cd {basedir}; git checkout origin main")
    return basedir + "emc3_example"


if __name__ == "__main__":
    import sys

    get_data(force=True)
    sys.exit(0)


import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_identical

import xemc3


# simple regression test
def test_load_all():
    bd = get_data()
    result = xemc3.load.all(bd)
    expected = xr.open_dataset(bd + ".nc")

    # Remove changing attributes
    changing = "software_version", "date_created", "id"
    for a in changing:
        assert a in result.attrs
        result.attrs.pop(a, None)
        expected.attrs.pop(a, None)

    # Remove new attributes, so we don't have to regenerate the data that often
    for k in list(result) + list(result.coords):
        if k not in expected:
            del result[k]
            continue
        for a in list(result[k].attrs):
            if a not in expected[k].attrs:
                del result[k].attrs[a]
    for a in list(result.attrs):
        if a not in expected.attrs:
            del result.attrs[a]

    assert_identical(result, expected)


def test_iter_plates():
    bd = get_data()
    ds = xemc3.load.plates(bd)
    c = 0
    for x in ds.emc3.iter_plates():
        c += 1
        assert np.all(np.isfinite(x.emc3["f_E"].data))
    assert c == 22
    c = 0
    for _ in ds.emc3.iter_plates(symmetry=True, segments=5):
        c += 1
    assert c == 22 * 5 * 2


def test_get_element():
    bd = get_data()
    ds = xemc3.load.plates(bd)
    fe = ds.emc3["f_E"]
    assert isinstance(fe, list)
    for fel in fe:
        assert isinstance(fel, xr.DataArray)
        assert fel.attrs == ds["f_E"].attrs
