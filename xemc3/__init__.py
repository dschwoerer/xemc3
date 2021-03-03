"""
A python library for reading EMC3 simulation as xarray data.

Additionally also provides some basic routines for plotting and
analysing the data. The data is mostly in SI units, with the exception
of temperatures that are in eV.
"""

__all__ = ["load", "utils", "write"]

from . import load

# assert callable(load)
# assert callable(load.plates)


from . import write


# # should be deprecated
# load.read_plate = _load.read_plate
# load.read_plate = _load.read_plate
# load.read_mappings = _load.read_mappings
# load.read_mapped = _load.read_mapped
# load.write_mapped = _load.write_mapped


from .core.dataset import EMC3DatasetAccessor


# Without having the coordinates fixed, there is no point working on xr.da
# @xr.register_dataarray_accessor("emc3")
# class EMC3DataarrayAccessor:
#     def __init__(self, da):
#         self.data = da
#         self.metadata = da.attrs.get("metadata")  # None if just grid file
#         self.load = load
try:
    from importlib.metadata import (
        version as _version,
        PackageNotFoundError as _PackageNotFoundError,
    )
except ModuleNotFoundError:
    from importlib_metadata import (
        version as _version,
        PackageNotFoundError as _PackageNotFoundError,
    )
try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    from setuptools_scm import get_version as _get_version

    __version__ = _get_version(root="..", relative_to=__file__)
