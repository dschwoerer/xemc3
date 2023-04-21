import matplotlib.pyplot as plt  # type: ignore
import matplotlib as mpl  # type: ignore
import numpy as np
import xarray as xr

from . import utils
from .load import plate_prefix


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
    target=False,
    log=False,
    robust=False,
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
    norm = None
    if key:
        data = das[2].data
        if "time" in das[2].dims:
            raise ValueError(
                "Unexpected dimension `time` - animation is not yet supported!"
            )
        if len(das[2].dims) != 2:
            if das[2].dims == das[0].dims:
                data = utils.from_interval(das[2])
                if "shading" not in kwargs:
                    kwargs["shading"] = "gouraud"
            else:
                raise ValueError(
                    f"Expected 2 dimensions for R-z plot, but found {len(das[2].dims)}: {das[2].dims}!"
                )
        if robust:
            vmin, vmax = np.nanpercentile(data, [1, 99])
        else:
            vmin = np.nanmin(data)
            vmax = np.nanmax(data)
        vmin = kwargs.pop("vmin", vmin)
        vmax = kwargs.pop("vmax", vmax)
        if log and vmin <= 0:
            raise ValueError(f"vmin ({vmin}) is not positive but log plot requested!")
        assert vmin < vmax, f"vmin ({vmin}) is not smaller than vmax ({vmax})"
        norm = (mpl.colors.LogNorm if log else mpl.colors.Normalize)(
            vmin=vmin, vmax=vmax
        )
    else:
        data = np.zeros(das[0].shape[:2]) * np.nan
        if "edgecolors" not in kwargs:
            kwargs["edgecolors"] = "k"
    r = utils.from_interval(das[0])
    z = utils.from_interval(das[1])
    ax = _get_ax(figsize, ax)
    p = ax.pcolormesh(r, z, data, norm=norm, **kwargs)
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
    if target:
        plot_target(ds, phi, ax=ax, fmt="r-" if key is None else "k-", aspect=aspect)
    return p


def plot_target(ds, phi, fmt=None, ax=None, figsize=None, aspect=True):
    ax = _get_ax(figsize, ax)
    if aspect:
        ax.set_aspect(1)
    das = xr.Dataset()
    for k in "R", "phi", "z":
        k1 = f"{plate_prefix}{k}"
        das[k] = ds[k1]
    for k in ds:
        if k.endswith("_dims") and k.startswith(f"_{plate_prefix}"):
            das[k] = ds[k]
    for p in das.emc3.iter_plates(symmetry=True, segments=5):
        assert len(p.plate_phi.shape) == 1
        if not (p.plate_phi.min() <= phi <= p.plate_phi.max()):
            continue
        p["plate_phi_plus1"] = p.plate_phi
        p = p.interp(
            plate_phi_plus1=phi, assume_sorted=False, kwargs=dict(bounds_error=True)
        )
        # except ValueError as e:
        #    assert "value in x_new is" in e.args[0]
        #    continue

        assert np.isclose(p.plate_phi, phi), f"{phi} expected but got {p.plate_phi}"
        ax.plot(p.plate_R, p.plate_z, fmt or "k-")

    ax.set_xlabel("R [m]")
    ax.set_ylabel("z [m]")


def _get_ax(figsize, ax):
    if figsize is not None:
        assert (
            ax is None
        ), "Passing in an axes object and specifing the figure size cannot be combined"
        plt.figure(figsize=figsize)
    if ax is None:
        ax = plt.axes(label=np.random.bytes(20))
    return ax
