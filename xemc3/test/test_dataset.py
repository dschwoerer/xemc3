from xemc3.core import dataset
import xarray as xr
import numpy as np


def test_sel():
    ds = xr.Dataset()

    ds.emc3["r_corners"] = "r", np.linspace(0, 4, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3, 4)

    for i in range(5):
        dss = ds.emc3.sel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == min(i, 3)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.sel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == int(i)


def test_isel():
    ds = xr.Dataset()
    ds.emc3["r_corners"] = "r", np.linspace(0, 4, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3, 4)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == min(i, 3)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == int(i)


def test_isel2():
    ds = xr.Dataset()
    rf = 2
    xf = 3
    ds.emc3["r_corners"] = "r", np.linspace(0, 4 * rf, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i * rf
        assert dss["x"] == min(i, 3) * xf
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i * rf
        assert dss["x"] == int(i) * xf


def test_isel_multi():
    ds = xr.Dataset()
    rf = 2
    zf = 5
    xf = 3
    rr = np.linspace(0, 4 * rf, 5)
    zr = np.linspace(0, 7 * zf, 8)
    ds.emc3["r_corners"] = ("r", "z"), (rr[:, None] + zr[None, :])
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)
    ds.emc3["y"] = "z", np.linspace(0, 6 * xf, 7)
    # ds.emc3["k"] = ("r","z"), np.linspace(0, 6 * xf, 7)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert all(dss.emc3["r_corners"] == i * rf + zr)
        assert dss["x"] == min(i, 3) * xf
        assert all(dss["y"] == ds["y"])
    for j in range(8):
        dss = ds.emc3.isel(z=j)
        assert all(dss.emc3["r_corners"] == j * zf + rr)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert np.allclose(dss.emc3["r_corners"], i * rf + zr)
        assert dss["x"] == int(i) * xf


if __name__ == "__main__":
    test_isel_multi()
