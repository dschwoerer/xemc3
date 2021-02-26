#!/usr/bin/env python3

import xarray as xr
import xemc3
import sys
import os


def main():
    parser = ArgumentParser(
        """
        Load the data from EMC3 simulations and store as netcdf file. The
        data is appended for each simulation to the netcdf file, which
        will be created if it does not yet exists.
        """
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="""Path of the directory to load. The netcdf file will be
        called dir.nc if the folder was called dir.""",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    args = parser.parse_args()

    for d in args.path:
        while d[-1] == "/":
            d = d[:-1]
        try:
            os.replace(d + ".nc", d + ".old.nc")
            old = xr.open_dataset(d + ".old.nc")
        except FileNotFoundError:
            old = None
        try:
            if not args.quiet:
                print(f"Loading {d} ...", end="")
                sys.stdout.flush()
            ds = xemc3.load(d)
            if old is not None:
                ds = xr.concat([old, ds], "time", "different")
            if not args.quiet:
                print(" writing ...", end="")
                sys.stdout.flush()
            ds.emc3.to_netcdf(d + ".nc")
        except:
            if old is not None:
                os.replace(d + ".old.nc", d + ".nc")
            raise
        if old is not None:
            os.remove(d + ".old.nc")
        if not args.quiet:
            print(" done")
