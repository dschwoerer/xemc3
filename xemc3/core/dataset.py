import xarray as xr
import numpy as np  # type: ignore
from . import utils
from . import load


def identity(x):
    return x


def from_interval_no_checks(x):
    return utils.from_interval(x, check=False)


@xr.register_dataset_accessor("emc3")
class EMC3DatasetAccessor:
    """Additional functions for working with EMC3 data."""

    def __init__(self, ds):
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
            for i in range(dims["plate_ind"]):
                slcr = [slice(None) if j is None else slice(None, j[i]) for j in crop]
                data = self.data.isel(plate_ind=i).data[tuple(slcr)]
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
            [self.data[k] for k in keys], self.data._plasma_map, fn, kinetic
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
        vol = plot_3d.volume(self.data)
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

    def time_average(self) -> xr.Dataset:
        """
        Average in time.

        Workaround for https://github.com/pydata/xarray/issues/4885
        """
        ds = self.data.copy()
        for k in ds:
            if "time" in ds[k].dims:
                ds[k] = ds[k].mean(dim="time")
        return ds
