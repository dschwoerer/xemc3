"""Routines for writing EMC3 files."""

import sys as _sys
import types as _types

# Actual implementation is in load.py
from ..core.load import write_all_fortran as all
from ..core.load import write_locations as coordinates
from ..core.load import write_magnetic_field as magnetic_field
from ..core.load import write_mapped_nice as mapped
from ..core.load import write_plates_mag as plates_mag


class _all(_types.ModuleType):
    def __init__(self):
        super().__init__(__name__)
        self.__dict__.update(_sys.modules[__name__].__dict__)

    def __call__(self, ds, dir):
        return all(ds, dir)


_all.__call__.__doc__ = all.__doc__

_sys.modules[__name__] = _all()
