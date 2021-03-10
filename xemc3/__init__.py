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


try:
    from importlib.metadata import (  # type: ignore
        version as _version,
        PackageNotFoundError as _PackageNotFoundError,
    )
except ModuleNotFoundError:
    from importlib_metadata import (  # type: ignore
        version as _version,
        PackageNotFoundError as _PackageNotFoundError,
    )
try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    from setuptools_scm import get_version as _get_version  # type: ignore

    __version__ = _get_version(root="..", relative_to=__file__)
