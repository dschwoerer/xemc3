#!/usr/bin/env python3
import typing

import numpy as np
import xarray as xr
from hypothesis import assume
from hypothesis import strategies as st

from ..core import dataset, load, utils

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
def hypo_vars(draw, skip=[]):
    files = []
    for v in load.files:
        if v in (
            "BFIELD_STRENGTH",
            "fort.70",
            "ADD_SF_N0",
            "TARGET_PROFILES",
        ):
            continue
        if v in skip:
            continue
        if draw(st.booleans()):
            t = [v]
            for k in load.files[v]["vars"]:
                if "%" in k:
                    t.append(draw(st.integers(min_value=1, max_value=7)))
            files.append(t)
    return files


@st.composite
def hypo_vars12(draw, skip_info=False):
    skip = []
    if skip_info:
        skip += [
            "NEUTRAL_INFO",
            "ENERGY_INFO",
            "IMPURITY_INFO",
            "STREAMING_INFO",
            # Merging of INFO files isn't supported yet, we just keep
            # the last one ...
        ]
    files = draw(hypo_vars(skip=skip))
    # Skip some random files
    skip += [
        # Also always skip PLATES_MAG:
        #   * it shouldn't change in a simulation
        #   * it is likely to be the same twice, thus resulting in
        #     a test failure.
        "PLATES_MAG",
        # Also always skip LG_CELL:
        #   * it is likely to be the same twice, thus resulting in
        #     a test failure.
        "LG_CELL",
    ]
    files2 = [f for f in files if draw(st.booleans()) and f[0] not in skip]
    assume(len(files2))
    return files, files2


def gen_ds(shape):
    ds = xr.Dataset()

    ds["_plasma_map"] = gen_mapping(shape)
    ds.emc3["bf_corners"] = gen_bf(shape)
    ds.emc3["R_corners"] = gen_bf(shape)
    ds.emc3["z_corners"] = gen_bf(shape)
    ds.emc3["phi_corners"] = ("phi",), np.random.random(shape[2] + 1)
    ds["ne"] = gen_mapped(ds, scale=1e6)
    ds["ne"].attrs["print_before"] = "   1\n"
    # ds["nZ1"] = gen_mapped(ds)
    ds["Te"] = gen_mapped(ds)
    ds["Ti"] = gen_mapped(ds)
    ds["nH"] = gen_mapped(ds, True)

    return ds


def gen_full(shape, index=None):
    return gen_rand(shape, None)


def gen_rand(shape, files):
    ds = xr.Dataset()

    ds["_plasma_map"] = gen_mapping(shape)
    ds.emc3["bf_corners"] = gen_bf(shape)
    ds["bf_bounds"].attrs = {
        "units": "T",
        "long_name": "Magnetic field strength",
        "parallel_flux": 0,
    }
    ds.emc3["R_corners"] = gen_bf(shape)
    ds.emc3["z_corners"] = gen_bf(shape)
    for k in "R_bounds", "z_bounds":
        ds[k].attrs["units"] = "m"
    ds.emc3["phi_corners"] = ("phi",), np.random.random(shape[2] + 1)
    ds["phi_bounds"].attrs["units"] = "radian"

    def get_attrs(vsv):
        out = {}
        if "attrs" in vsv:
            out.update(vsv["attrs"])
        for attr in "long_name", "units", "notes":
            if attr in vsv:
                out[attr] = vs[v][attr]
        if not v in ["_plasma_map", "bf_bounds", "R_bounds", "phi_bounds", "z_bounds"]:
            k = "parallel_flux"
            out[k] = vs[v].get(k, 0)
        return out

    for f in load.files:
        if f in ["ADD_SF_N0", "TARGET_PROFILES"]:
            continue
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
        if f in ("PARTICLE_DEPO", "ENERGY_DEPO"):
            ids = 6
        i = 0
        vs = load.files[f]["vars"]
        for v in vs:
            if v in ds:
                ds[v].attrs.update(get_attrs(vs[v]))
                continue

            genf = {
                "mapped": gen_mapped,
                "full": gen_bf,
                "plates_mag": gen_plates_mag,
                "info": gen_info,
                "raw": gen_raw,
                "depo": gen_depo,
            }[load.files[f].get("type", "mapped")]
            if load.files[f].get("kinetic", False):
                assert genf == gen_mapped
                genf = gen_kinetic
            pre = load.files[f].get("skip_first", 0)
            scale = load.files[f].get("scale", None)
            dtype = load.files[f].get("dtype", float)

            def add_pre(ds, k, pre, i):
                if pre:
                    if isinstance(pre, int):
                        pret = pre
                    else:
                        pret = pre[min(len(pre) - 1, i)]
                    ds[k].attrs["print_before"] = ("   %d\n" % i) * pret
                return ds

            if "%" in v:
                for i in range(i, i + ids):
                    dat = genf(ds, index=i, scale=scale)
                    ds[v % i] = dat
                    if dtype != float and genf != gen_depo:
                        ds[v % i] = genf(ds)[0], np.round(genf(ds)[1] * 20)
                    ds[v % i].attrs.update(get_attrs(vs[v]))
                    ds = add_pre(ds, v % i, pre, i)
            else:
                ds[v] = genf(ds, index=i)
                ds[v].attrs.update(get_attrs(vs[v]))
                ds = add_pre(ds, v, pre, i)
                if genf == gen_depo and i == 0:
                    ds[v].attrs["description"] = "True means +1, False means -1"

            i += 1
    return ds


def gen_bf(shape, index=None, scale=None):
    if isinstance(shape, xr.Dataset):
        shape = shape["_plasma_map"].data.shape
    return dims, 0.5 + 2 * np.random.random([i + 1 for i in shape])


def gen_plates_mag(shape, index=None, scale=None):
    if isinstance(shape, xr.Dataset):
        shape = shape["_plasma_map"].data.shape
    return dims, (0.5 + 1 * np.random.random(shape) > 1)


def gen_kinetic(ds, index=None, scale=None):
    return gen_mapped(ds, True, scale)


def gen_mapped(ds, kinetic=False, index=None, scale=None):
    key = "other" if kinetic else "plasmacells"
    map = ds["_plasma_map"]
    shape = map.shape
    max = map.attrs[key]
    dat = np.random.random(max)
    if scale:
        dat *= scale
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


def gen_mapping(shape, index=None, scale=None):
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


def gen_info(ds: xr.Dataset, index=None, scale=None) -> xr.DataArray:
    index = "iteration"
    length = np.random.randint(2, 6)
    dat = np.empty(1000)
    dat[:] = np.nan
    dat[-length:] = np.random.random(length)
    coords: typing.Mapping[typing.Hashable, typing.Any] = {index: np.arange(-999, 1)}
    return xr.DataArray(dat, dims=index, coords=coords)


def gen_raw(ds: xr.Dataset, index=None) -> xr.DataArray:
    text = """Some text.
Sooo random.
With linebreaks 🎉
"""
    return xr.DataArray(text)


gen_depo_map = None


def gen_depo(ds: xr.Dataset, index=None, scale=None):
    shape = ds["_plasma_map"].data.shape
    shape = [x + 1 for x in shape]
    dtype = bool if index in (0, 6) else float
    out = np.empty(shape, dtype=dtype)
    fill = np.nan
    if index == 0:
        fill = True
    if index == 6:
        fill = False
    out[:] = fill

    global gen_depo_map

    if index == 0 or index == 6:
        gen_depo_map = np.random.random(shape) > 0.8

    rand = np.random.random(shape)[gen_depo_map]
    if index == 0:
        out[gen_depo_map] = rand > 0.5
    elif index == 6:
        out[gen_depo_map] = True
    else:
        out[gen_depo_map] = rand

    return tuple([f"{x}_plus1" for x in ds["_plasma_map"].dims]), out


class rotating_circle(object):
    def __init__(self, period, iota=1, sym=False):
        self.period = period
        self.iota = iota
        self.sym = sym
        self.R = 5
        self.r = 1

    def gends(self, shape):
        phi = np.linspace(
            0, np.pi * (1 if self.sym else 2) / self.period, shape[2] + 1
        )  # [None, None, :]
        theta = np.linspace(0, np.pi * 2, shape[1] + 1)[None, :, None] - (
            np.pi / shape[1]
        )

        r = np.linspace(0, self.r, shape[0] + 1)[:, None, None]
        r, p, z = self.rpt_to_rpz(r, phi, theta)

        ds = xr.Dataset()
        ds["_plasma_map"] = gen_mapping(shape)
        ds.emc3["bf_corners"] = gen_bf(shape)
        ds.emc3["R_corners"] = dims, r
        ds.emc3["z_corners"] = dims, z
        ds.emc3["phi_corners"] = ("phi",), p
        return ds

    def rpt_to_rpz(self, r, p, t):
        angle = t + self.iota * p
        r_corner = self.R + r * np.cos(angle)
        z_corner = r * np.sin(angle)
        return r_corner, p, z_corner

    def rpz_to_xyz(self, r, p, z):
        x = r * np.cos(p)
        y = r * np.sin(p)
        return x, y, z


if __name__ == "__main__":
    print(load.files["LG_CELL"])
    ds = gen_full((3, 4, 5))
    print(load.files["LG_CELL"])
    print(ds)
