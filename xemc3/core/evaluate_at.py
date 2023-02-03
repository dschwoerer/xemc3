import warnings
from typing import Dict

import numpy as np
import xarray as xr
from eudist import PolyMesh  # type: ignore

from . import utils


def _evaluate_get_keys(ds, r, phi, z, periodicity, updownsym, delta_phi, progress):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value encountered in remainder")
        phi = phi % (np.pi * 2 / periodicity)
    # Get raw data
    dims, shape, coords, (r, phi, z) = get_out_shape(r, phi, z)
    pln = xr.Dataset()
    pln = pln.assign_coords({k: ds[k] for k in ["z_bounds", "R_bounds", "phi_bounds"]})
    lr = len(ds.r)
    lt = len(ds.theta)
    lp = len(ds.phi)
    pln["phi_index"] = "phi", np.arange(lp, dtype=int)
    pln["r_index"] = ("r", "theta"), np.zeros((lr, lt), dtype=int) + np.arange(
        lr, dtype=int
    )[:, None]
    pln["theta_index"] = ("r", "theta"), np.zeros((lr, lt), dtype=int) + np.arange(
        lt, dtype=int
    )
    keys = ["phi_index", "r_index", "theta_index"]

    cache: Dict[int, PolyMesh] = {}
    scache: Dict[int, xr.Dataset] = {}

    n = len(pln.theta)
    outs = [np.empty(shape, dtype=pln[k].data.dtype) for k in keys]
    cid = -1
    assert "delta_phi" in pln.phi_bounds.dims
    assert "phi" in pln.phi_bounds.dims

    rrange = utils.rrange2 if progress else utils.rrange
    for ijk in rrange(shape):
        if not np.isfinite(phi[ijk]):
            for i in range(len(keys)):
                try:
                    outs[i][ijk] = phi[ijk]
                except ValueError:
                    try:
                        outs[i][ijk] = np.nan
                    except ValueError:  # cannot assign nan to integers
                        outs[i][ijk] = -1
            continue

        if updownsym and phi[ijk] > np.pi / periodicity:
            zc = -z[ijk]
            phic = (np.pi * 2 / periodicity) - phi[ijk]
        else:
            zc = z[ijk]
            phic = phi[ijk]

        j = -1
        if delta_phi:
            j = int(round((phic - delta_phi / 2) / delta_phi))

        try:
            mesh = cache[j]
            s = scache[j]
        except KeyError:
            if delta_phi:
                phic = (
                    round((phic - delta_phi / 2) / delta_phi) * delta_phi
                    + delta_phi / 2
                )
                if updownsym and phic > np.pi / periodicity:
                    zc = -zc
                    phic = (np.pi * 2 / periodicity) - phi[ijk]
            s = pln.emc3.sel(phi=phic)
            mesh = PolyMesh(s.emc3["R_corners"].data, s.emc3["z_corners"].data)
            if delta_phi:
                cache[j] = mesh
                scache[j] = s
        cid = mesh.find_cell(np.array([r[ijk], zc]), cid)
        if cid == -1:
            for i in range(len(keys)):
                outs[i][ijk] = -1
        else:
            ij = cid // n, cid % n
            for out, key in zip(outs, keys):
                if len(s[key].dims):
                    out[ijk] = s[key].data[ij]
                else:
                    out[ijk] = s[key].data
    keyout = [k.split("_")[0] for k in keys]

    ret = xr.Dataset(coords=coords)
    for i in range(len(dims)):
        d0 = dims[i]
        cnt = 0
        while dims[i] in keyout:
            dims[i] = f"{d0}_{cnt}"
        if dims[i] != d0 and d0 in ret:
            ret = ret.rename({d0: dims[i]})
    for out, k, ko in zip(outs, keys, keyout):
        ret[k.split("_")[0]] = xr.DataArray(out, dims=dims, attrs=pln[k].attrs)
    return ret


def get_out_shape(*data):
    """
    Convert xarray arrays and plain arrays to same shape and return
    dims, shape and the raw arrays
    """
    if any([isinstance(x, xr.DataArray) for x in data]):
        dims = []
        shape = []
        out = []
        coords = {}
        for d in data:
            if isinstance(d, xr.DataArray):
                for dim in d.dims:
                    if dim in dims:
                        assert len(d[dim]) == shape[dims.index(dim)]
                    else:
                        dims.append(dim)
                        shape.append(len(d[dim]))
                        coords[dim] = d.coords[dim]
            else:
                assert (
                    utils.prod(np.shape(d)) == 1
                ), "Cannot mix `xr.DataArray`s and `np.ndarray`s"
        outzero = xr.DataArray(np.zeros(shape), dims=dims, coords=coords)
        out = [outzero + d for d in data]
        # for o in out:
        #    assert all(o.dims == dims) if len(o.dims) > 1 else o.dims == dims
        out = [d.data for d in out]
        return dims, shape, coords, out
    outzero = np.zeros(1)
    for d in data:
        outzero = outzero * d
    out = [outzero + d for d in data]
    dims = [f"dim_{d}" for d in range(len(outzero.shape))]
    shape = outzero.shape
    return dims, shape, None, out
