import xarray as xr
import numpy as np
from typing import Mapping, Hashable, Any, Dict, Optional

from . import utils
from . import load


def identity(x):
    return x


def from_interval_no_checks(x):
    return utils.from_interval(x, check=False)


@xr.register_dataset_accessor("emc3")
class EMC3DatasetAccessor:
    """Additional functions for working with EMC3 data."""

    def __init__(self, ds: xr.Dataset):
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  # None if just grid file

    def __str__(self):
        """
        Get a string representation of the EMC3Dataset.

        Accessed by e.g. print(ds.bout).
        """
        text = (
            "<xemc3.EMC3Dataset>\n"
            + "Contains:\n{}\n".format(str(self.data))
            + "Metadata:\n{}\n".format(str(self.metadata))
        )
        return text

    def _get(self, var):
        """Load a single var."""
        transform = identity
        try:
            dims = self.data[var].dims
        except KeyError as e:
            if var.endswith("_corners"):
                var_ = var[: -len("_corners")] + "_bounds"
                try:
                    dims = self.data[var_].dims
                except KeyError:
                    raise e
                var = var_
                transform = from_interval_no_checks
            else:
                raise
        if "plate_ind" in dims:
            crop = []
            for d in dims:
                if d == "plate_ind":
                    continue
                try:
                    crop.append(self.data[d + "_dims"].data)
                except KeyError:
                    crop.append(None)
            ret = []
            for i in range(len(self.data["plate_ind"])):
                slcr = tuple(
                    [slice(None) if j is None else slice(None, j[i]) for j in crop]
                )
                data = self.data.isel(plate_ind=i)
                # coords = {
                #     k: xr.DataArray(
                #         coord.data[slcr], dims=coord.dims, attrs=coord.attrs
                #     )
                #     for k, coord in data.coords.items()
                # }
                data = xr.DataArray(
                    data[var].data[slcr],
                    dims=data[var].dims,
                    attrs=data[var].attrs,
                    # coords=coords,
                )
                ret.append(transform(data))
            return ret
        crop = []
        for d in dims:
            try:
                crop.append(slice(None, self.data[d + "_dims"].data))
            except KeyError:
                crop.append(slice(None))
        data = self.data[var][tuple(crop)]
        return transform(data)

    def _set(self, var, data):
        """Set a single variable."""
        transform = identity
        if var.endswith("_corners"):
            var = var[: -len("_corners")] + "_bounds"
            transform = utils.to_interval
        # Maybe also do the cropping? See merge code somewhere
        self.data[var] = transform(data)
        return self

    def get(self, *args):
        """
        Get one or more variables of the dataset.

        The shapes are cropped to include only valid data if data for
        the divertor modules is returened.

        The code also transforms to ``*_corner`` cases, if
        ``*_bounds`` is present. The bounds version is an array of
        shape :math:`n\\times...\\times m\\times 2\\times...\\times 2`
        while the corner version is a :math:`n+1\\times...\\times m+1`
        dimensional array.

        Parameters
        ----------
        args : list of str
            The keys of the data to get

        Returns
        -------
        list of xr.DataArray
            The requested arrays from the dataset
        """
        return [self._get(i) for i in args]

    def __getitem__(self, var):
        """
        Get a variable from the dataset.

        See `get` for the transormations performed.

        Parameters
        ----------
        var : str
            The key to load

        Returns
        -------
        xr.DataArray
            The array from the dataset
        """
        return self._get(var)

    def __setitem__(self, var, data):
        """
        Set a variable in the dataset.

        In case a a var ending in ``*_corners`` is set, the data is
        transformed and the corresponding ``*_bounds`` var is set
        instead.

        Parameters
        ----------
        var : str
            The name of the variable
        data : any
            The data to be set. See the xarray documentation for what
            is accepted.
        """
        return self._set(var, data)

    def iter_plates(self, *, symmetry=False, segments=1):
        """
        Iterate over all plates.

        Repeat with stellerator symmetry if symmetry is given. If
        segments is given, repeat for all segments - assuming a 2*pi/n
        rotational transform symmetry.

        Parameters
        ----------
        symmetry: boolean (optional)
            Iterate over mirrored setup
        segments: int (optional)
            Iterate over all segments, assuming a symmetry of
            n=segments.

        Returns
        -------
        iterator of xr.Dataset
            The plates of the divertor plates
        """
        phis = []
        zs = []
        if segments != 1 or symmetry:
            for phioffset in np.linspace(0, 2 * np.pi, segments, endpoint=False):
                for sym in [False, True] if symmetry else [False]:
                    phi = self.data["phi_bounds"].copy()
                    z = self.data["z_bounds"].copy()
                    if sym:
                        phi *= -1
                        z *= -1
                    if phioffset != 0:
                        phi += phioffset
                    phis.append(phi)
                    zs.append(z)
        else:
            phis.append(self.data["phi_bounds"])
            zs.append(self.data["z_bounds"])
        for phi, z in zip(phis, zs):
            ds = self.data.copy()
            ds["phi_bounds"] = phi
            ds["z_bounds"] = z
            for i in range(ds.dims["plate_ind"]):
                yield ds.isel(plate_ind=i)

    def plot_div(self, index, **kwargs):
        """Plot divertor data."""
        from .plot_3d import divertor

        return divertor(self.data, index, **kwargs)

    def unit(self, var, value=None):
        """Get or set the units of a quantity"""
        if value is not None:
            self.data[var].attrs["units"] = value
        return self.data[var].attrs["units"]

    def to_netcdf(self, fn, complevel=1):
        """
        Write to netcdf file.

        Enables fast compression by default, which helps a lot with
        the sparse data.

        fn: string
            The path of the netcdf file to be written.

        complevel: integer (optional)
            The compression level in the range 1 to 9
        """
        self.data.to_netcdf(
            fn,
            encoding={
                i: {"zlib": True, "complevel": complevel}
                for i in [i for i in self.data] + [i for i in self.data.coords]
            },
        )

    def to_fort(self, keys, fn, kinetic=False):
        """
        Write to text file, using the mappings for plasma or kinetic data.
        """
        if not isinstance(keys, list):
            keys = [keys]
        return load.write_mapped(
            [self.data[k] for k in keys], self.data._plasma_map, fn, kinetic=kinetic
        )

    def from_fort(self, fn, skip_first=0, ignore_broken=False, kinetic=False):
        """
        Read from an EMC3 file, using the mappings for plasma or
        kinetic data.

        Parameters
        ----------
        fn : str
            The name of the file to be read
        skip_first : int (optional)
            For each datablock the first ``skip_first`` lines are
            ignored
        ignore_broken : boolean (optional)
            Incomplete datasets are ignored. This indicates that
            something is wrong, but is the default for EMC3
        kinetic : boolean (optional)
            If kinetic then the data is read using the mapping for
            neutal quantities, otherwise the mapping for plasma
            quantities is assumed

        Returns
        -------
        list of xr.DataArray
            The list of the read quantities
        """
        return load.read_mapped(
            fn, self.data._plasma_map, skip_first, ignore_broken, kinetic
        )

    def plot_Rz(self, key, phi, **kwargs):
        """
        Plot a R-z slice in lab coordinates.

        Parameters
        ----------
        key : str
            Index of the data to plot

        phi : float
            Angle at which to plot. As always in radian.

        ax : Axis object (optional)
            Axis object to be used for plotting

        Rmin : float (optional)
            left bound for plot

        Rmax : float (option)
            right bound for plot

        zmin : float (optional)
            lower bound for plot

        zmax : float (optional)
            upper bound for plot

        kwargs: Other arguments to be passed to matplotlib

        Returns
        -------
        matplotlib.collections.QuadMesh
            The return value from matplotlib.pyplot.pcolormesh is returned.
        """
        from . import plot_2d

        return plot_2d.plot_rz(self.data, key, phi, **kwargs)

    def plot(self, key, *args, **kw):
        """
        Plot some data.

        In case of data from data from the plates, a 3D plot of the divertor is returned.

        In case of 3D data a volume plot is returend.

        Otherwise xarray is used for plotting. Axes information might be missing.

        See also: plot_rz

        Parameters
        ----------
        key : str
            The key used for plotting

        Returns
        -------
        any
            Return value depends on the specific plotting routine used.
        """
        da = self.data[key]
        if len(da.dims) < 3:
            return da.plot(*args, **kw)
        # For 3D:
        from . import plot_3d

        if "plate_ind" in self.data.dims:
            # assert args == []
            return plot_3d.divertor(self.data, key, *args, **kw)

        init = {}
        for k in "updownsym", "periodicity":
            if k in kw:
                init[k] = kw.pop(k)
        vol = plot_3d.volume(self.data, **init)
        return vol.plot(key, *args, **kw)

    def load(self, path):
        """
        Load EMC3 simulation data from path

        Parameters
        ----------
        path : str
            The directory of the EMC3 simulation data

        Returns:
        xr.Dataset
            The xemc3 dataset with the simulation data
        """
        return load(path)

    def mean_time(self) -> xr.Dataset:
        """
        Average in time.

        Workaround for https://github.com/pydata/xarray/issues/4885
        """
        ds = self.data.copy()
        for k in ds:
            if "time" in ds[k].dims:
                attrs = ds[k].attrs
                ds[k] = ds[k].mean(dim="time")
                ds[k].attrs = attrs
        return ds

    def isel(
        self,
        indexers: Mapping[str, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> xr.Dataset:
        ds = self.data
        indexers = utils.merge_indexers(indexers, indexers_kwargs)
        mine = {}
        xas = {}
        for k, v in indexers.items():
            if "delta_" + k in ds.dims and k in ds.dims:
                mine[k] = v
            else:
                xas[k] = v
        ds = ds.isel(drop=drop, missing_dims=missing_dims, **xas)
        for k, v in mine.items():
            dk = "delta_" + k
            if v == len(ds[k]):
                ds = ds.isel({k: int(v) - 1, dk: 1})
            elif v == int(v):
                ds = ds.isel({k: int(v), dk: 0})
            else:
                vi = int(v)
                fac = v - vi
                ds_ = (ds.isel({k: vi}) * xr.DataArray([1 - fac, fac], dims=dk)).sum(
                    dim=dk
                )
                for co in ds.coords:
                    ds_[co] = (
                        ds[co].isel({k: vi}) * xr.DataArray([1 - fac, fac], dims=dk)
                    ).sum(dim=dk)
                ds = ds_
        return ds

    def sel(
        self,
        indexers: Mapping[str, Any] = None,
        drop: bool = False,
        missing_dims: str = "raise",
        **indexers_kwargs: Any,
    ) -> xr.Dataset:
        ds = self.data
        indexers = utils.merge_indexers(indexers, indexers_kwargs)
        forisel = {}
        for k in indexers.keys():
            val = indexers[k]
            if "delta_" + k in ds.dims and k + "_bounds" in ds:
                assert k in ds.dims
                assert (
                    len(ds[k + "_bounds"].dims) == 2
                ), "Only 1D bounds are currently supported. Maybe try isel."
                dat = ds[k + "_bounds"]
                if dat.dims == (k, "delta_" + k):
                    pass
                elif dat.T.dims == (k, "delta_" + k):
                    dat = dat.T
                else:
                    raise RuntimeError(
                        f"Unexpected dimensions for {k}_bounds - expected {k} and delta_{k} but got {dat.dims}"
                    )
                for i in range(len(dat)):
                    if dat[i, 0] <= val <= dat[i, 1]:
                        break
                else:
                    raise RuntimeError(f"Didn't find {val} in {k}_bounds")
                fac = (val - dat[i, 0]) / (dat[i, 1] - dat[i, 0])
                forisel[k] = i + fac
            else:
                forisel[k] = val
        ds = ds.emc3.isel(forisel)
        assert isinstance(ds, xr.Dataset)
        return ds

    def evaluate_at_xyz(self, x, y, z, *args, **kwargs):
        """
        See evaluate_at_rpz for options. Unlike evaluate_at_rpz the
        coordinates are given here in cartesian coordinates.
        """
        r = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return self.evaluate_at_rpz(r, phi, z, *args, **kwargs)

    def evaluate_at_rpz(
        self,
        r,
        phi,
        z,
        key=None,
        periodicity: int = 5,
        updownsym: bool = True,
        delta_phi: float = None,
    ):
        """
        Evaluate the field key in the dataset at the positions given by
        the array r, phi, z.  If key is None, return the indices to access
        the 3D field and get the appropriate values.

        Parameters
        ----------
        r : array-like
            The (major) radial coordinates to evaluate
        phi : array-like
            The toroidal coordinate
        z : array-like
            The z component
        key : None or str or sequence of str
            If None return the index-coordinates otherwise evaluate the
            specified field in the dataset
        periodicity : int
            The rotational symmetry in toroidal direction
        updownsym : bool
            Whether the data is additionally up-down symmetric with half
            the periodicity.
        delta_phi : None or float
            If not None, delta_phi gives the accuracy of the precision at
            which phi is evaluated. Giving a float enables caching of the
            phi slices, and can speed up computation. Note that it should
            be smaller then the toroidal resolution. For a grid with 1°
            resolution, delta_phi=2 * np.pi / 360 would be the upper
            bound.  None disables caching.
        """
        from eudist import PolyMesh  # type: ignore

        phi = phi % (np.pi * 2 / periodicity)
        # Get raw data
        dims, shape, coords, (r, phi, z) = get_out_shape(r, phi, z)
        pln = xr.Dataset()
        pln = pln.assign_coords(
            {k: self.data[k] for k in ["z_bounds", "R_bounds", "phi_bounds"]}
        )
        if key is None:
            lr = len(self.data.r)
            lt = len(self.data.theta)
            lp = len(self.data.phi)
            pln["phi_index"] = "phi", np.arange(lp, dtype=int)
            pln["r_index"] = ("r", "theta"), np.zeros((lr, lt), dtype=int) + np.arange(
                lr, dtype=int
            )[:, None]
            pln["theta_index"] = ("r", "theta"), np.zeros(
                (lr, lt), dtype=int
            ) + np.arange(lt, dtype=int)
            keys = ["phi_index", "r_index", "theta_index"]
        else:
            if isinstance(key, str):
                keys = [key]
            else:
                keys = key
            for k in keys:
                pln[k] = self.data[k]

        cache: Dict[int, PolyMesh] = {}
        scache: Dict[int, xr.Dataset] = {}

        n = len(pln.theta)
        outs = [np.empty(shape, dtype=pln[k].data.dtype) for k in keys]
        cid = -1
        assert "delta_phi" in pln.phi_bounds.dims
        assert "phi" in pln.phi_bounds.dims
        for ijk in utils.rrange(shape):
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
                s = pln.emc3.sel(phi=phic)
                mesh = PolyMesh(s.emc3["R_corners"].data, s.emc3["z_corners"].data)
                if delta_phi:
                    cache[j] = mesh
                    scache[j] = s
            cid = mesh.find_cell(np.array([r[ijk], zc]), cid)
            if cid == -1:
                for i in range(len(keys)):
                    try:
                        outs[i][ijk] = np.nan
                    except ValueError:  # cannot assign nan to integers
                        outs[i][ijk] = -1
            else:
                ij = cid // n, cid % n
                for i, key in enumerate(keys):
                    if len(s[key].dims):
                        outs[i][ijk] = s[key].data[ij]
                    else:
                        outs[i][ijk] = s[key].data
        ret = xr.Dataset(coords=coords)
        for i, k in enumerate(keys):
            ret[k] = dims, outs[i]
            ret[k].attrs = pln[k].attrs
        return ret

    # def evaluate_at_indices(self, indices:xr.Dataset, key: str) -> xr.DataArray:


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
    dims = [f"dim_{d}" for d in outzero.shape]
    shape = outzero.shape
    return dims, shape, None, out
