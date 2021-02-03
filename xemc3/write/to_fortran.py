"""Routines for writing EMC3 files."""

import sys as _sys
import types as _types

# Actual implementation is in load.py
from ..core.load import (
    write_magnetic_field as magnetic_field,
    write_locations as coordinates,
    write_plates_mag as plates_mag,
    write_mapped_nice as mapped,
    write_all_fortran as _write_all,
)


class all(_types.ModuleType):
    def __init__(self):
        super().__init__(__name__)
        self.__dict__.update(_sys.modules[__name__].__dict__)


all.__call__ = _write_all

_sys.modules[__name__] = all()
