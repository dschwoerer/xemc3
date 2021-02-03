#!/usr/bin/env python3

import xarray as xr
import xemc3
import sys
import os


for d in sys.argv[1:]:
    while d[-1] == "/":
        d = d[:-1]
    try:
        os.replace(d + ".nc", d + ".old.nc")
        old = xr.open_dataset(d + ".old.nc")
    except FileNotFoundError:
        old = None
    try:
        print(f"Loading {d} ...", end="")
        sys.stdout.flush()
        ds = xemc3.load(d)
        if old is not None:
            ds = xr.concat([old, ds], "time", "different")
        print(" writing ...", end="")
        sys.stdout.flush()
        ds.emc3.to_netcdf(d + ".nc")
    except:
        if old is not None:
            os.replace(d + ".old.nc", d + ".nc")
        raise
    if old is not None:
        os.remove(d + ".old.nc")
    print(" done")
