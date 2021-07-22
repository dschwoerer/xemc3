import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import xarray as xr
from . import utils


def plot_rz(
    ds,
    key,
    phi,
    ax=None,
    Rmin=None,
    Rmax=None,
    zmin=None,
    zmax=None,
    aspect=True,
    figsize=None,
    **kwargs,
):
    phis = ds["phi_bounds"]
    if phi < np.min(phis.data) or phi > np.max(phis.data):
        raise RuntimeError(
            f"{phi} outside of bounds in dataset {np.min(phis)}:{np.max(phis)}"
        )
    for phi_i, phib in enumerate(phis):
        if phib[0] <= phi <= phib[1]:
            break
    else:
        raise RuntimeError(f"no suitable phi slice found for {phi} in {phis}")

    p = ((phi - phib[0]) / (phib[1] - phib[0])).data
    ds = ds.isel(phi=phi_i)
    das = [ds[k] for k in ["R_bounds", "z_bounds"]]
    if key:
        das.append(ds[key])
    pp = xr.DataArray(data=[(1 - p), p], dims="delta_phi")
    das = [
        (da * pp).sum(dim="delta_phi") if "delta_phi" in da.dims else da for da in das
    ]
    if key:
        data = das[2].data
        if "time" in ds[key].dims:
            raise ValueError(
                "Unexpected dimension `time` - animation is not yet supported!"
            )
        if len(ds[key].dims) != 2:
            raise ValueError(
                f"Expected 2 dimensions for R-z plot, but found {len(ds[key].dims)}: {ds[key].dims}!"
            )
    else:
        data = np.zeros(das[0].shape[:2]) * np.nan
        if "edgecolors" not in kwargs:
            kwargs["edgecolors"] = "k"
    r = utils.from_interval(das[0])
    z = utils.from_interval(das[1])
    if figsize is not None:
        assert (
            ax is None
        ), "Passing in an axes object and specifing the figure size cannot be combined"
        plt.figure(figsize=figsize)
    if ax is None:
        ax = plt.axes(label=np.random.bytes(20))
    p = ax.pcolormesh(r, z, data, **kwargs)
    # plt.xlabel(xr.plot.utils.label_from_attrs(r))
    if aspect:
        ax.set_aspect(1)
    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")
    plt.colorbar(p, ax=ax)
    if Rmin is not None or Rmax is not None:
        ax.set_xlim(Rmin, Rmax)
    if zmin is not None or zmax is not None:
        ax.set_ylim(zmin, zmax)
    p.colorbar.set_label(label=xr.plot.utils.label_from_attrs(das[-1]))
    return p
