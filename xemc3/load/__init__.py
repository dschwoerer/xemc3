"""Routines for reading EMC3 files."""

__all__ = ["plates", "mappings", "all", "mapped_raw", "file", "var", "plates_geom"]

import sys as _sys
import types as _types

# Actual implementation is in load.py
from ..core.load import (
    get_plates as plates,
    load_all as all,
    load_any as any,
    read_fort_file_pub as file,
    read_mapped as mapped_raw,
    read_plate_nice as plates_geom,
    read_var as var,
)


class _any(_types.ModuleType):
    def __init__(self):
        super().__init__(__name__)
        self.__dict__.update(_sys.modules[__name__].__dict__)

    def __call__(self, path, ignore_missing=None):
        return any(path, ignore_missing)


_any.__call__.__doc__ = any.__doc__

_sys.modules[__name__] = _any()
