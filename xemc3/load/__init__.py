"""Routines for reading EMC3 files."""

__all__ = ["plates", "mappings", "all"]

import sys as _sys
import types as _types

# Actual implementation is in load.py
from ..core.load import (
    get_plates as plates,
    read_mappings as mappings,
    read_mapped as mapped_raw,
    read_mapped_nice as mapped,
    load_all as all,
)

# write_magnetic_field as magnetic_field,
# write_locations as coordinates,
# write_plates_mag as plates_mag,
# write_mapped_nice as mapped,
# write_all_fortran as _write_all,


class _all(_types.ModuleType):
    def __init__(self):
        super().__init__(__name__)
        self.__dict__.update(_sys.modules[__name__].__dict__)

    def __call__(self, path, ignore_missing=None):
        return all(path, ignore_missing)


_all.__call__.__doc__ = all.__doc__

_sys.modules[__name__] = _all()
