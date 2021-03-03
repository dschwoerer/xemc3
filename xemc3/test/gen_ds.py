#!/usr/bin/env python3
import xarray as xr
import numpy as np
from xemc3.core import utils, load

from hypothesis import assume, strategies as st

dims = "r", "theta", "phi"

setting = dict(deadline=None, max_examples=10)


@st.composite
def hypo_shape(draw, max=1000):
    s = []
    for i in range(3):
        s.append(draw(st.integers(min_value=1, max_value=int(max))))
        max /= s[-1]
    return tuple(s)


@st.composite
def hypo_vars(draw):
    files = []
    for v in load.files:
        if v in ("BFIELD_STRENGTH", "fort.70"):
            continue
        if draw(st.booleans()):
            t = [v]
            for k in load.files[v]["vars"]:
                if "%" in k:
                    t.append(draw(st.integers(min_value=1, max_value=20)))
            files.append(t)
    return files


@st.composite
def hypo_vars12(draw):
    files = draw(hypo_vars())
    # Skip some random files
    # Also always skip PLATES_MAG:
    #   * it shouldn't change in a simulation
    #   * it is likely to be the same twice, thus resulting in a test failure
    files2 = [f for f in files if draw(st.booleans()) and f[0] != "PLATES_MAG"]
    assume(len(files2))
    return files, files2


def gen_ds(shape):
    ds = xr.Dataset()

    ds["_plasma_map"] = gen_mapping(shape)
    ds.emc3["bf_corners"] = gen_bf(shape)
    ds.emc3["R_corners"] = gen_bf(shape)
    ds.emc3["z_corners"] = gen_bf(shape)
    ds.emc3["phi_corners"] = ("phi",), np.random.random(shape[2] + 1)
    ds["ne"] = gen_mapped(ds)
    ds["ne"].attrs["print_before"] = "   1\n"
    # ds["nZ1"] = gen_mapped(ds)
    ds["Te"] = gen_mapped(ds)
    ds["Ti"] = gen_mapped(ds)
    ds["nH"] = gen_mapped(ds, True)

    return ds


def gen_full(shape):
    return gen_rand(shape, None)


def gen_rand(shape, files):
    ds = xr.Dataset()

    ds["_plasma_map"] = gen_mapping(shape)
    ds.emc3["bf_corners"] = gen_bf(shape)
    ds["bf_bounds"].attrs = {"units": "T", "long_name": "Magnetic field strength"}
    ds.emc3["R_corners"] = gen_bf(shape)
    ds.emc3["z_corners"] = gen_bf(shape)
    for k in "R_bounds", "z_bounds":
        ds[k].attrs["units"] = "m"
    ds.emc3["phi_corners"] = ("phi",), np.random.random(shape[2] + 1)
    for f in load.files:
        ids = 3
        if files is not None:
            for f2 in files:
                if f2[0] == f:
                    try:
                        ids = f2[1]
                    except IndexError:
                        ids = 3
                    break
            else:
                continue
        i = 0
        vs = load.files[f]["vars"]
        for v in vs:
            if v in ds:
                if "attrs" in vs[v]:
                    ds[v].attrs.update(vs[v]["attrs"])
                continue
            genf = {"mapped": gen_mapped, "full": gen_bf, "plates_mag": gen_plates_mag}[
                load.files[f].get("type", "mapped")
            ]
            if load.files[f].get("kinetic", False):
                assert genf == gen_mapped
                genf = gen_kinetic
            pre = load.files[f].get("skip_first", 0)
            dtype = load.files[f].get("dtype", float)

            if "%" in v:
                for i in range(i, i + ids):
                    ds[v % i] = genf(ds)
                    if dtype != float:
                        ds[v % i] = genf(ds)[0], np.round(genf(ds)[1] * 20)
                    if "attrs" in vs[v]:
                        ds[v % i].attrs = vs[v]["attrs"].copy()
                    if pre:
                        ds[v % i].attrs["print_before"] = "   %d\n" % i
            else:
                ds[v] = genf(ds)
                if "attrs" in vs[v]:
                    ds[v].attrs = vs[v]["attrs"].copy()
                if pre:
                    ds[v].attrs["print_before"] = "   %d\n" % i
            i += 1
    return ds


def gen_bf(shape):
    if isinstance(shape, xr.Dataset):
        shape = shape["_plasma_map"].data.shape
    return dims, 0.5 + 2 * np.random.random([i + 1 for i in shape])


def gen_plates_mag(shape):
    if isinstance(shape, xr.Dataset):
        shape = shape["_plasma_map"].data.shape
    return dims, (0.5 + 1 * np.random.random(shape) > 1)


def gen_kinetic(ds):
    return gen_mapped(ds, True)


def gen_mapped(ds, kinetic=False):
    key = "other" if kinetic else "plasmacells"
    map = ds["_plasma_map"]
    shape = map.shape
    max = map.attrs[key]
    dat = np.random.random(max)
    ret = np.zeros(shape) * np.nan
    mapdat = map.values
    for ijk in utils.rrange(shape):
        i = mapdat[ijk]
        if i < max:
            ret[ijk] = dat[i]
    return map.dims, ret


def gen_updated(org: xr.Dataset, var) -> xr.Dataset:
    shape = org["_plasma_map"].shape
    dt = gen_rand(shape, var)
    d2 = org.copy()
    for k in dt:
        if k in (
            "_plasma_map",
            "bf_bounds",
            "R_bounds",
            "z_bounds",
            "phi_bounds",
        ):
            continue
        d2[k] = dt[k]
    return d2


def gen_mapping(shape):
    dat = np.zeros(shape, dtype=int)
    i = 0
    for ijk in utils.rrange(shape):
        dat[ijk] = i // 3
        i += 1
    da = xr.DataArray(dat, dims=dims)
    pc = i // 6
    if pc == 0:
        pc = 1
    da.attrs = dict(numcells=i, plasmacells=pc, other=np.max(dat) + 1)
    return da


if __name__ == "__main__":
    print(load.files["LG_CELL"])
    ds = gen_full((3, 4, 5))
    print(load.files["LG_CELL"])
    print(ds)
