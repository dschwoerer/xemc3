"""
A python library for reading EMC3 simulation as xarray data.

Additionally also provides some basic routines for plotting and
analysing the data. The data is mostly in SI units, with the exception
of temperatures that are in eV.
"""

__all__ = ["load", "utils", "write"]

from . import load, write
from .core.dataset import EMC3DatasetAccessor

# assert callable(load)
# assert callable(load.plates)


# # should be deprecated
# load.read_plate = _load.read_plate
# load.read_plate = _load.read_plate
# load.read_mappings = _load.read_mappings
# load.read_mapped = _load.read_mapped
# load.write_mapped = _load.write_mapped


try:
    from importlib.metadata import \
        PackageNotFoundError as _PackageNotFoundError
    from importlib.metadata import version as _version  # type: ignore
except ModuleNotFoundError:
    from importlib_metadata import \
        PackageNotFoundError as _PackageNotFoundError
    from importlib_metadata import version as _version  # type: ignore
try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    from setuptools_scm import get_version as _get_version  # type: ignore

    __version__ = _get_version(root="..", relative_to=__file__)
