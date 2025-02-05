import warnings
from typing import Any, Mapping, Union, Optional
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal


import numpy as np
import xarray as xr

from . import load, utils


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

    def _get_crop(self, dims, getDA=False, skip=[]):
        for d in dims:
            if d in skip:
                continue
            try:
                if getDA:
                    yield self.data[f"_{d}_dims"]
                else:
                    yield self.data[f"_{d}_dims"].data
            except KeyError:
                try:
                    if getDA:
                        yield self.data[f"{d}_dims"]
                    else:
                        yield self.data[f"{d}_dims"].data
                except KeyError:
                    yield None

    def _get(self, var, crop=True, resolve=True):
        """
        Load a single var.

        crop : bool
            Remove nan data
        resolve : bool
            Check alternative names to get data
        """
        docrop = crop
        transform = identity
        try:
            dims = self.data[var].dims
        except KeyError as e:
            if not resolve:
                raise e
            if var.endswith("_corners"):
                var_ = var[: -len("_corners")] + "_bounds"
                try:
                    dims = self.data[var_].dims
                except KeyError:
                    raise e
                var = var_
                transform = from_interval_no_checks
            elif var.endswith("_bounds"):
                var = var[: -len("_bounds")]
                try:
                    dims = self.data[var + "_corners"].dims
                    var = var + "_corners"
                except KeyError:
                    try:
                        dims = self.data[var].dims
                    except KeyError:
                        raise e
                if not any(["_plus1" in x for x in dims]):
                    raise e
                transform = utils.to_interval
            else:
                raise
        if docrop and ("plate_ind" in dims or "zone" in dims):
            ind = "plate_ind"
            ind = ind if ind in dims else "zone"
            crop = list(self._get_crop(dims, skip=[ind]))
            ret = []
            for i in range(len(self.data[ind])):
                slcr = tuple(
                    [slice(None) if j is None else slice(None, j[i]) for j in crop]
                )
                data = self.data.isel(**{ind: i})
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
        if docrop:
            crop = [slice(None, x) for x in self._get_crop(dims)]
            data = self.data[var][tuple(crop)]
        else:
            data = self.data[var]
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

        See `get` for the transformations performed.

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

    def _get_alt_name(self, var_name, suffix=""):
        var_suffix = ["_bounds", "", "_plus1"]
        var_prefix = ["plate_", "_plate_", ""]
        for prefix in var_prefix:
            for extra_suffix in var_suffix:
                cur = prefix + var_name + extra_suffix + suffix
                if cur in self.data:
                    return cur
        assert False, f"Didn't find variable for {var_name} coordinate!"

    def iter_zones(self):
        """
        Iterate over all zones
        """
        if not "zone" in self.data.dims:
            yield self.data
        else:
            for i in range(len(self.data.zone)):
                yield self.isel(zone=i, drop=True)

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
        basevar = {k: self._get_alt_name(k) for k in ("phi", "z")}
        var_list = {k: [] for k in basevar.values()}

        if segments != 1 or symmetry:
            for phioffset in np.linspace(0, 2 * np.pi, segments, endpoint=False):
                for sym in [False, True] if symmetry else [False]:
                    for key in var_list:
                        tmp = self.data[key].copy()
                        if sym:
                            tmp *= -1
                        if phioffset != 0 and basevar["phi"] == key:
                            tmp += phioffset
                        var_list[key].append(tmp)
        else:
            for key in var_list:
                var_list[key].append(self.data[key])
        for i in range(len(next(iter(var_list.values())))):
            ds = self.data.copy()
            dims = ds.dims
            crop = self._get_crop(dims, True)
            crop = {
                d: c.data
                for d, c in zip(dims, crop)
                if c is not None and c.dims == ("plate_ind",)
            }
            phid = ds[self._get_alt_name("phi", "_dims")].data
            xd = ds[self._get_alt_name("x", "_dims")].data

            for key in var_list:
                ds[key] = var_list[key][i]
            if len(phid.shape) == 1:
                assert phid.shape == xd.shape, (
                    f"Expected phi.shape = {phid.shape} == x.shape = {xd.shape} to match."
                    + utils.raise_issue
                )
                for j in range(ds.dims["plate_ind"]):
                    yield ds.isel(
                        plate_ind=j, **{k: slice(None, v[j]) for k, v in crop.items()}
                    )
            else:
                for j in range(ds.dims["plate_ind"]):
                    yield ds.isel(plate_ind=j)

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
                for i in list(self.data) + list(self.data.coords)
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

    def plot_rz(self, key, phi, **kwargs):
        """
        Plot a R-z slice in lab coordinates.

        Parameters
        ----------
        key : str or None
            Index of the data to plot. Select None to plot the mesh
            instead.

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

    def plot_Rz(self, key, phi, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                "plot_Rz is deprecated. Please switch to plot_rz instead.",
                category=DeprecationWarning,
                stacklevel=2,
            )
        return self.plot_rz(key, phi, **kwargs)

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

        Returns
        -------
        xr.Dataset
            The xemc3 dataset with the simulation data
        """
        return load.read_fort_file_pub(path, self)

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
        indexers: Optional[Mapping[str, Any]] = None,
        drop: bool = False,
        missing_dims: Union[
            Literal["raise"], Literal["warn"], Literal["ignore"]
        ] = "raise",
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
                dsa = xr.Dataset({c: ds[c] for c in ds if dk in ds[c].dims})
                if len(list(dsa)):
                    ds_ = (
                        dsa.isel({k: vi}) * xr.DataArray([1 - fac, fac], dims=dk)
                    ).sum(dim=dk, skipna=False)
                else:
                    ds_ = dsa
                for co in ds:
                    if dk not in ds[co].dims:
                        ds_[co] = ds[co].isel({k: vi}, missing_dims="ignore")
                for co in ds.coords:
                    if dk in ds.coords[co].dims:
                        ds_[co] = (
                            ds[co].isel({k: vi}) * xr.DataArray([1 - fac, fac], dims=dk)
                        ).sum(dim=dk)
                    else:
                        ds_[co] = ds[co].isel({k: vi}, missing_dims="ignore")
                ds = ds_
        xas = {}
        for k in ds.dims:
            key = f"_{k}_dims"
            if key in ds and ds[key].dims == ():
                xas[k] = slice(None, ds[key].values)
        if xas:
            ds = ds.isel(xas)
        return ds

    def sel(
        self,
        indexers: Optional[Mapping[str, Any]] = None,
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
                assert (
                    k in ds.dims
                ), f"Expected {k} in {ds.dims} - maybe you already selected in {k} dim?"
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
                dat = dat.data
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
        assert isinstance(
            ds, xr.Dataset
        ), f"Expected to have an xr.Dataset, but instead got {type(ds)}"
        return ds

    def evaluate_at_xyz(self, x, y, z, *args, **kwargs):
        """
        See evaluate_at_rpz for options. Unlike evaluate_at_rpz the
        coordinates are given here in cartesian coordinates.
        """
        r = np.sqrt(x**2 + y**2)
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
        delta_phi: Optional[float] = None,
        fill_value=None,
        lazy=False,
        progress=False,
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
        fill_value : None or any
            If fill_value is None, missing data is initialised as
            np.nan, or as -1 for non-float datatypes. Otherwise
            fill_value is used to set missing data.
        lazy : bool
            Force the loading of the data for key. Defaults to False.
            Can significantly decrease performance, but can decrease
            memory usage.
        progress : bool
            Show the progress of the mapping. Defaults to False.
        """
        from .evaluate_at import _evaluate_get_keys

        at = _evaluate_get_keys(
            self.data, r, phi, z, periodicity, updownsym, delta_phi, progress
        )
        if key is None:
            return at
        if isinstance(key, str):
            keys = [key]
        else:
            keys = key
        ret = xr.Dataset(coords=at.coords)
        fill = at["phi"].data == -1
        filldims = at["phi"].dims
        dofill = np.any(fill)
        for k in keys:
            if not lazy:
                self.data[k].data
            ret[k] = self.data[k].isel(**at)
            if dofill:
                # fillthis = fill if filldims == ret[k].dims else ret[k].data
                assert (
                    ret[k].dims == filldims
                ), f"Got dimensions {ret[k].dims} but expected {filldims} for key {k}"
                if fill_value is None:
                    try:
                        ret[k].data[fill] = np.nan
                    except ValueError:
                        ret[k].data[fill] = -1
                else:
                    ret[k].data[fill] = fill_value
        return ret

    def evaluate_at_diagnostic(self, diag, key=None, num_points=100):
        ret = dict()
        for var in ("los",):
            if var in diag.__dict__:
                for i, dat in enumerate(diag.__dict__[var]):
                    ret[f"{var}_{i}"] = self.evaluate_at_xyz(
                        *dat.get_points(num_points)
                    )
        return ret
