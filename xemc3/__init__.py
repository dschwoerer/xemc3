"""
A python library for reading EMC3 simulation as xarray data.

Additionally also provides some basic routines for plotting and
analysing the data. The data is mostly in SI units, with the exception
of temperatures that are in eV.
"""

__all__ = ["load", "write", "config"]

from . import load, write, config
from .core.dataset import EMC3DatasetAccessor


try:
    from importlib.metadata import (  # type: ignore
        PackageNotFoundError as _PackageNotFoundError,
        version as _version,
    )
except ModuleNotFoundError:
    from importlib_metadata import (  # type: ignore
        PackageNotFoundError as _PackageNotFoundError,
        version as _version,
    )
try:
    __version__ = _version(__name__)
except _PackageNotFoundError:
    from setuptools_scm import get_version as _get_version  # type: ignore

    __version__ = _get_version(root="..", relative_to=__file__)
