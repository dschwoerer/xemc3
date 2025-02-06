import tempfile

import numpy as np
from hypothesis import given, settings

import xemc3

from . import gen_ds


def assert_ds_are_equal(d1, d2, check_attrs=True, rtol=1e-2, atol=1e-6):
    """
    d1 : old dataset
    d2 : new dataset
    """
    d1k = [k for k in d1] + [k for k in d1.coords]
    d2k = [k for k in d2] + [k for k in d2.coords]
    if not set(d1k) == set(d2k):
        raise AssertionError(f"{d1.keys()} != {d2.keys()}")
    for k in d1k:
        tatol = atol
        if k.endswith("_change"):
            tatol = 1e-3
        if k == "ne" or k.startswith("nZ"):
            tatol = 1e2
        assert_da_are_equal(d1[k], d2[k], k, check_attrs, rtol, tatol)


def assert_da_are_equal(d1, d2, k, check_attrs, rtol, atol):
    if d1.dtype == float:
        slc = np.isfinite(d1.data)
    else:
        slc = slice(None)
    if (
        d1.dtype == float
        and (
            d1.shape != d2.shape
            or not np.isclose(d1.data[slc], d2.data[slc], rtol=rtol, atol=atol).all()
        )
    ) or (d1.dtype != float and (d1.shape != d2.shape or np.any(d1.data != d2.data))):
        raise AssertionError(
            f"""var {k} is changed.

Before: {d1.shape}: {d1.data.flatten()}

After: {d2.shape}: {d2.data.flatten()}

np.isclose: {np.isclose(d1, d2 ,rtol=rtol).flatten()}"""
        )
    if check_attrs:
        d1a = d1.attrs.copy()
        key = "xemc3_type"
        if key not in d1a and key in d2.attrs:
            d1a[key] = d2.attrs[key]

        assert d1a == d2.attrs, f"attributes changed for {k}: {d1a} != {d2.attrs}"


setting = gen_ds.setting


@settings(**setting)  # type: ignore
@given(gen_ds.hypo_shape())
def test_write_load_simple(shape):
    ds = gen_ds.gen_ds(shape)
    with tempfile.TemporaryDirectory() as dir:
        # print(ds)
        # print(ds["_plasma_map"])
        xemc3.write.fortran(ds, dir)
        dl = xemc3.load(dir)
        assert_ds_are_equal(ds, dl, False, 1e-2, 1e-2)
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran.all(dl, dir)
        dn = xemc3.load.all(dir)
        da = xemc3.load.mapped_raw(dir + "/DENSITY_A", dn, kinetic=True)
        assert not isinstance(da, list), "test with squeeze failed"
        assert np.allclose(da, dn["nH"] / 1e6), "reading mapped_raw gives wrong data"
        da = xemc3.load.mapped_raw(dir + "/DENSITY_A", dn, kinetic=True, squeeze=False)
        assert isinstance(da, list), "squeeze=False gives list"
        assert np.allclose(da[0], dn["nH"] / 1e6), "mapped_raw gives wrong data"
        for var, fn in ("nH", "DENSITY_A"), ("bf_bounds", "BFIELD_STRENGTH"):
            for withmaps in True, False:
                da = xemc3.load.var(dir, var, dn["_plasma_map"] if withmaps else None)
                assert np.allclose(da[var], dn[var]), (
                    "load.var fails with" + ("" if withmaps else "out") + " map fails"
                )
                da = xemc3.load.file(
                    f"{dir}/{fn}", dn["_plasma_map"] if withmaps else None
                )
                assert np.allclose(da[var], dn[var]), (
                    "load.file fails with" + ("" if withmaps else "out") + " map fails"
                )

        # print(xemc3.core.load.files)
        assert_ds_are_equal(dl, dn, True, 1e-4)


@settings(**setting)  # type: ignore
@given(gen_ds.hypo_shape(200))
def test_write_load_full(shape):
    ds = gen_ds.gen_full(shape)
    # if True:
    #    dir = "xemc3/test/test"
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran(ds, dir)
        dl = xemc3.load(dir)
        assert_ds_are_equal(ds, dl, True, 1e-2, 1e-2)
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran.all(dl, dir)
        dn = xemc3.load.all(dir)
        # print(xemc3.core.load.files)
        assert_ds_are_equal(dl, dn, True, 1e-4)


@settings(**setting)  # type: ignore
@given(gen_ds.hypo_shape(200), gen_ds.hypo_vars())
def test_write_load_some(shape, vars):
    ds = gen_ds.gen_rand(shape, vars)
    # if True:
    #    dir = "xemc3/test/test"
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran(ds, dir)
        dl = xemc3.load(dir)
        assert_ds_are_equal(ds, dl, True, 1e-2, 1e-2)
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran.all(dl, dir)
        dn = xemc3.load.all(dir)
        # print(xemc3.core.load.files)
        assert_ds_are_equal(dl, dn, True, 1e-4)


if __name__ == "__main__":
    test_write_load_simple()
    test_write_load_full()
