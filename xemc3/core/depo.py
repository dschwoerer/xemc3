import xarray as xr
import numpy as np
import re
import typing

from .utils import rrange, raise_issue


if 0:
    import sparse  # type: ignore
else:
    sparse = None

# #################################################
# Abstractions between sparse and full data:
# #################################################


def newarray(shape, dtype=None, fill_value=np.nan):
    if dtype == bool and fill_value is np.nan:
        fill_value = False
    if sparse:
        if dtype:
            return sparse.DOK(shape=shape, dtype=dtype, fill_value=fill_value)
        return sparse.DOK(shape=shape, fill_value=fill_value)
    if dtype:
        ret = np.empty(shape, dtype=dtype)
    else:
        ret = np.empty(shape)
    ret[:] = fill_value
    return ret


def tocoo(data):
    if sparse:
        return data.to_coo()
    return data


def nnz(data):
    if sparse:
        return data.nnz
    return np.sum(np.isfinite(data))


def keys(data):
    if hasattr(data, "coords"):
        for ijk in data.coords.T:
            yield ijk
        return
    try:
        for ijk in data.keys():
            yield ijk
        return
    except AttributeError:
        for ijk in rrange(data.shape):
            if np.isfinite(data[ijk]):
                yield ijk


def sparseCOO(locs, data, shape, fill_value):
    if sparse:
        return sparse.COO(locs, data, shape, fill_value)
    ret = newarray(shape, data.dtype, fill_value)
    retflat = ret.reshape(-1)
    for ijk, d in zip(locs, data):
        retflat[ijk] = d
    return ret


# #################################################
# The "main" code:
# #################################################


def read_depo_raw(ds: xr.Dataset, fn: str) -> typing.List[xr.DataArray]:
    bad = re.compile(r"(\d)([+-]\d)")
    # print(ds)

    shape = tuple([x + 1 for x in [len(ds.r), len(ds.theta), len(ds.phi)]])
    dims = tuple([x + "_plus1" for x in ["r", "theta", "phi"]])
    haszone = "zone" in ds.dims
    if haszone:
        shape = len(ds.zone), *shape
        dims = "zone", *dims

    with open(fn) as f:
        line = f.readline().split()
        assert len(line) == 2, (
            f"Unexpected: `{line}` at start of `{fn}` - expected 2 elements"
            + raise_issue
        )
        blockasize = int(line[0])
        # Lines look like this:
        # Index
        # 3
        # type of the surface
        # index of zone
        # index of surface in radial
        #                  in poloidal
        #                  in toroidal direction
        # flux
        out = {
            "surftype": newarray(shape=shape, dtype=bool, fill_value=True),
            "flux": newarray(shape=shape, fill_value=np.nan),
            "other": [newarray(shape=shape, fill_value=np.nan) for _ in range(4)],
        }
        varnames = "surftype", "flux", "other"

        hasother = hasother2 = False
        last = None
        for i in range(blockasize):
            line = bad.sub(r"\1E\2", f.readline()).split()
            ints = [int(x) for x in line[:7]]
            floats = [float(x) for x in line[7:]]
            assert ints[0] == i + 1, (
                f"Expected first index to be contigous, thus expected {i+1} but got {ints[0]} while reading {fn}"
                + raise_issue
            )
            assert ints[1] == 3, (
                f"Expected 3 but got {ints[1]} while reading `{fn}`." + raise_issue
            )
            # print(ints[3:])
            if haszone:
                slc = tuple(ints[3:])
            else:
                assert ints[3] == 0, (
                    f"Expected zoneid=0 but got {ints[3]}" + raise_issue
                )
                slc = tuple(ints[4:])

            for name, val in zip(varnames, (ints[2] == 1, floats[0])):
                out[name][slc] = val
            if last:
                assert not out["surftype"][last], (
                    f"Unexpected input in `{fn}`." + raise_issue
                )
            if ints[2] != 1:
                last = slc
            # print("a", out["surftype"][slc], ints[2] == 1)

            if len(floats) > 1:
                hasother = True
                for j in range(4):
                    out["other"][j][slc] = floats[j + 1]

        _ = f.readline()

        out2 = {
            "surftype": newarray(shape=shape, dtype=bool, fill_value=False),
            "flux": newarray(shape=shape, fill_value=np.nan),
            "other": [newarray(shape=shape, fill_value=np.nan) for _ in range(4)],
        }

        while True:
            line = bad.sub(r"\1E\2", f.readline()).split()
            if line == []:
                break
            ints = [int(x) for x in line[:7]]
            floats = [float(x) for x in line[7:]]
            assert ints[1] == 3, f"Expected 3 but got {ints[1]}" + raise_issue
            if haszone:
                slc = tuple(ints[3:])
            else:
                assert ints[3] == 0, f"Unexpected input in {fn} `line`." + raise_issue
                slc = tuple(ints[4:])
            for name, val in zip(varnames, (ints[2] == 1, floats[0])):
                out2[name][slc] = val
            # print("b", out["surftype"][slc], ints[2] == 1)

            if len(floats) > 1:
                hasother2 = True
                for j in range(4):
                    out2["other"][j][slc] = floats[j + 1]

    if sparse:
        assert any([d.nnz for d in out["other"]]) == hasother, raise_issue
        assert any([d.nnz for d in out2["other"]]) == hasother2, raise_issue
    # if not hasother:
    # out["other"] = []
    assert not hasother2, raise_issue
    out2["other"] = []

    ret = [
        xr.DataArray(data=tocoo(d), dims=dims)
        for d in [out["surftype"], out["flux"], *out["other"]]
    ], [
        xr.DataArray(data=tocoo(d), dims=dims)
        for d in [out2["surftype"], out2["flux"], *out2["other"]]
    ]
    assert len(ret[0]) == 6, f"Expected 6 items but got {len(ret[0])}." + raise_issue
    assert len(ret[1]) == 2, raise_issue
    return ret[0] + ret[1]


def write_depo_raw_part(datas, f, i):
    surf = datas[0].data
    dats = [d.data for d in datas[1:]]
    zone = [] if "zone" in datas[0].dims else [0]
    off = 1 if i == 0 else 0
    # print(keys(dats[0]))
    for ijk in keys(dats[0]):
        i += off
        ijk = tuple(ijk)
        s = 1 if surf[ijk] else -1

        zijk = *zone, *ijk
        f.write(
            f"{i:6d} 3 {s:2d} {zijk[0]:2d} "
            + (
                " ".join(
                    [f"{d:4d}" for d in zijk[1:]] + [f"{d[ijk]:11.4E}" for d in dats]
                )
            )
            + "\n"
        )
    return i


def write_depo_raw(datas, fn):
    assert len(datas) == 8, (
        f"Expected 8 data entries, but got { len(datas) }" + raise_issue
    )
    datas = datas[:6], datas[6:]
    i = 0
    with open(fn, "w") as f:
        f.write(
            f"      {nnz(datas[0][1].data):6d} "
            + (
                f"{nnz(datas[1][1].data):11d}\n"
                if "PARTICLE_DEPO" in fn
                else "TARGET\n"
            )
        )
        # for k in datas[0]:
        #     print()
        #     print(k)
        i = write_depo_raw_part(datas[0], f, i)
        f.write(" MAPPING\n")
        write_depo_raw_part(datas[1], f, i + 1)


# #################################################
# Code for writing sparse files. Currently not used
# #################################################


def da_to_netcdf(da, fn):
    if hasattr(da.data, "nnz"):
        ds = xr.Dataset()
        ds["var"] = da
        return ds_to_netcdf(ds, fn)
    da.to_netcdf()


def ds_to_netcdf(ds, fn):
    dsorg = ds
    ds = dsorg.copy()
    for v in ds:
        if hasattr(ds[v].data, "nnz") and (
            hasattr(ds[v].data, "to_coo") or hasattr(ds[v].data, "linear_loc")
        ):
            coord = f"_{v}_xarray_index_"
            assert coord not in ds
            data = ds[v].data
            if hasattr(data, "to_coo"):
                data = data.to_coo()
            ds[coord] = coord, data.linear_loc()
            dims = ds[v].dims
            ds[coord].attrs["compress"] = " ".join(dims)
            at = ds[v].attrs
            ds[v] = coord, data.data
            ds[v].attrs = at
            ds[v].attrs["_fill_value"] = str(data.fill_value)
            for d in dims:
                if d not in ds:
                    ds[f"_len_{d}"] = len(dsorg[d])

    # print(ds)
    ds.to_netcdf(fn)


def xr_open_dataset(fn):
    ds = xr.open_dataset(fn)

    def fromflat(shape, i):
        index = []
        for fac in shape[::-1]:
            index.append(i % fac)
            i //= fac
        return tuple(index[::-1])

    for c in ds.coords:
        if "compress" in ds[c].attrs:
            vs = c.split("_")
            if len(vs) < 5:
                continue
            if vs[-1] != "" or vs[-2] != "index" or vs[-3] != "xarray":
                continue
            v = "_".join(vs[1:-3])
            # at = ds[v].attrs
            dat = ds[v].data
            fill = ds[v].attrs.pop("_fill_value", None)
            if fill:
                knownfails = {"nan": np.nan, "False": False, "True": True}
                if fill in knownfails:
                    fill = knownfails[fill]
                else:
                    fill = np.fromstring(fill, dtype=dat.dtype)
            dims = ds[c].attrs["compress"].split()
            shape = []
            for d in dims:
                try:
                    shape.append(len(ds[d]))
                except KeyError:
                    shape.append(int(ds[f"_len_{d}"].data))
                    ds = ds.drop_vars(f"_len_{d}")

            locs = fromflat(shape, ds[c].data)
            data = sparseCOO(locs, ds[v].data, shape, fill_value=fill)
            ds[v] = dims, data, ds[v].attrs, ds[v].encoding
    # print(ds)
    return ds


# #################################################
# Code for manual testing ...
# #################################################


def get_ds():
    cache = "this.nc"
    try:
        return xr.open_dataset(cache)
    except FileNotFoundError:
        pass
    import xemc3

    ds = xemc3.load(".")
    ds.to_netcdf(cache)
    return ds


def get_p(ds):
    cache = "test_p.nc"
    try:
        ds = xr_open_dataset(cache)
    except FileNotFoundError:
        pass
    else:
        x = [[ds[f"var{j}_{i}"] for i in range(6 - j * 4)] for j in range(2)]
        return x

    x = read_depo_raw(ds, "PARTICLE_DEPO")
    ds = xr.Dataset()
    for i, t in enumerate(x):
        for j, v in enumerate(t):
            ds[f"var{i}_{j}"] = v
    ds_to_netcdf(ds, cache)
    return x


def get_e(ds):
    cache = "test_e.nc"
    try:
        ds = xr_open_dataset(cache)
    except FileNotFoundError:
        pass
    else:
        x = [[ds[f"var{j}_{i}"] for i in range(6 - j * 4)] for j in range(2)]
        return x

    x = read_depo_raw(ds, "ENERGY_DEPO")
    ds = xr.Dataset()
    for i, t in enumerate(x):
        for j, v in enumerate(t):
            ds[f"var{i}_{j}"] = v
    ds_to_netcdf(ds, cache)
    return x


if __name__ == "__main__":
    ds = get_ds()

    x = get_p(ds)
    t = x[0][1:]
    # print(len(t))
    # print(xr.combine_nested(t, "time"))

    write_depo_raw(x, "PARTICLE_DEPO_NEW")
    y = get_e(ds)

    # read_depo_raw(ds, "ENERGY_DEPO")
    write_depo_raw(y, "ENERGY_DEPO_NEW")
