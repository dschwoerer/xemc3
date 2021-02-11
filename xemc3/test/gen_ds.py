#!/usr/bin/env python3
import xarray as xr
import numpy as np
from xemc3.core import utils, load

dims = "r", "theta", "phi"


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
    ds = xr.Dataset()

    ds["_plasma_map"] = gen_mapping(shape)
    ds.emc3["bf_corners"] = gen_bf(shape)
    ds.emc3["R_corners"] = gen_bf(shape)
    ds.emc3["z_corners"] = gen_bf(shape)
    for k in "R_bounds", "z_bounds":
        ds[k].attrs["units"] = "m"
    ds.emc3["phi_corners"] = ("phi",), np.random.random(shape[2] + 1)
    for f in load.files:
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
                for i in range(i, i + 3):
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
