"""
A python library for reading EMC3 simulation as xarray data.

Additionally also provides some basic routines for plotting and
analysing the data. The data is mostly in SI units, with the exception
of temperatures that are in eV.
"""
import xarray as xr
import itertools
import os
import numpy as np

from .core import load as _load
from .core import utils as _utils


load = _load.load_all
for f in ["read_plates", "read_mappings", "read_mapped", "write_mapped"]:
    setattr(load, f, getattr(_load, f))
load.plates = _load.get_plates
load.by_var = _load.read_mapped_nice
assert callable(load)
assert callable(load.plates)


from . import write


# # should be deprecated
load.read_plate = _load.read_plate
# load.read_plate = _load.read_plate
# load.read_mappings = _load.read_mappings
# load.read_mapped = _load.read_mapped
# load.write_mapped = _load.write_mapped


@xr.register_dataset_accessor("emc3")
class EMC3DatasetAccessor:
    """Additional functions for working with EMC3 data."""

    def __init__(self, ds):
        self.data = ds
        self.metadata = ds.attrs.get("metadata")  # None if just grid file
        self.load = load

    def __str__(self):
        """
        Get a string representation of the EMC3Dataset.

        Accessed by e.g. print(ds.bout).
        """
        styled = partial(prettyformat, indent=4, compact=True)
        text = (
            "<xemc3.EMC3Dataset>\n"
            + "Contains:\n{}\n".format(str(self.data))
            + "Metadata:\n{}\n".format(styled(self.metadata))
        )
        return text

    def _get(self, var):
        """Load a single var."""
        transform = lambda x: x
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
                transform = lambda x: _utils.from_interval(x, check=False)
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
                slcr = [slice(None) if j == None else slice(None, j[i]) for j in crop]
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
        transform = lambda x: x
        if var.endswith("_corners"):
            var = var[: -len("_corners")] + "_bounds"
            transform = lambda x: _utils.to_interval(x)
        ## Maybe also do the cropping? See merge code somewhere
        self.data[var] = transform(data)
        return self

    def get(self, *args):
        """
        Get one or more variables of the dataset.

        The shapes are cropped to include only valid data if data for
        the divertor modules is returened.

        The code also transforms to *_corner cases, if *_bounds is
        present. The bounds version is an array of shape
        nx...xmx2x...x2 while the corner version is a n+1x...xm+1
        dimensional array.
        """
        return [self._get(i) for i in args]

    def __getitem__(self, var):
        return self._get(var)

    def __setitem__(self, var, data):
        return self._set(var, data)

    def iter_plates(self, *, symmetry=False, segments=1):
        """
        Iterate over all plates.

        Repeat with stellerator symmetry if symmetry is given. If
        segments is given, repeat for all segments - assuming a 2*pi/n
        rotational transform symmetry.

        symmetry: optional, bool
            Iterate over mirrored setup
        segments: optional, int
            Iterate over all segments, assuming a symmetry of
            n=segments.
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
        from .core.plot_3d import divertor

        plot_div.__doc__ = divertor.__doc__

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
        return _load.write_mapped(
            [self.data[k] for k in keys], self.data._plasma_map, fn, kinetic
        )

    def from_fort(self, fn, skip_first=0, ignore_broken=False, kinetic=False):
        """
        Read from text file, using the mappings for plasma or kinetic data.
        """
        return _load.read_mapped(
            fn, self.data._plasma_map, skip_first, ignore_broken, kinetic
        )

    def plot_Rz(self, key, phi, **kwargs):
        """
        Plot a R-z slice in lab coordinates.

        key: string
            Index of the data to plot

        phi: number
            Angle at which to plot. As always in radian.

        ax: Axis object (optional)
            Axis object to be used for plotting

        Rmin: number (optional)
            left bound for plot

        Rmax: number (option)
            right bound for plot

        zmin: number (option)
            lower bound for plot

        zmax: number (option)
            upper bound for plot

        kwargs: Other arguments to be passed to matplotlib
        """
        from .core import plot_2d

        plot_2d.plot_rz(self.data, key, phi, **kwargs)

    def plot(self, key, *args, **kw):
        """
        Plot some data.

        In case of data from data from the plates, a 3D plot of the divertor is returned.

        In case of 3D data a volume plot is returend.

        Otherwise xarray is used for plotting. Axes information might be missing.

        See also: plot_rz
        """
        da = self.data[key]
        if len(da.dims) < 3:
            return da.plot(*args, **kw)
        # For 3D:
        from core import plot_3d

        if "plate_ind" in self.data.dims:
            # assert args == []
            return _plot.divertor(self.data, key, *args, **kw)
        vol = _plot.volume(self.data)
        return vol.plot(key, *args, **kw)

    def load(self, path):
        """Load simulation from `path`"""
        return load(path)


# Without having the coordinates fixed, there is no point working on xr.da
# @xr.register_dataarray_accessor("emc3")
# class EMC3DataarrayAccessor:
#     def __init__(self, da):
#         self.data = da
#         self.metadata = da.attrs.get("metadata")  # None if just grid file
#         self.load = load
try:
    from importlib.metadata import version, PackageNotFoundError
except ModuleNotFoundError:
    from importlib_metadata import version, PackageNotFoundError
try:
    __version__ = version(__name__)
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)


if __name__ == "__main__":
    import sys

    plates = load_plates(sys.argv[1])
    print(plates)
    write_plates(sys.argv[1], plates)
    read_plates(sys.argv[1])

    raise RuntimeError
    ds = getLocations(sys.argv[1])

    ds["x"] = ds.R * np.cos(ds.phi)
    ds["y"] = ds.R * np.sin(ds.phi)
    print(ds)
