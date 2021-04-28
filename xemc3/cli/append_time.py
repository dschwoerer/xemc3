#!/usr/bin/env python3

from argparse import ArgumentParser
import xarray as xr
from ..core.load import load_all
from ..core import dataset
import sys
import os


def append_time(d: str, verbose: bool = False) -> None:
    while d[-1] == "/":
        d = d[:-1]
    try:
        os.replace(d + ".nc", d + ".old.nc")
        old = xr.open_dataset(d + ".old.nc")
    except FileNotFoundError:
        old = None
    try:
        if verbose:
            print(f"Loading {d} ...", end="")
            sys.stdout.flush()
        ds = load_all(d)
        if old is not None:
            ds = xr.concat([old, ds], "time", "different")
        if verbose:
            print(" writing ...", end="")
            sys.stdout.flush()
        ds.emc3.to_netcdf(d + ".nc")
    except Exception:
        if old is not None:
            os.replace(d + ".old.nc", d + ".nc")
        raise
    if old is not None:
        os.remove(d + ".old.nc")
    if verbose:
        print(" done")


def parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="""
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
    return parser


def main() -> None:
    args = parser().parse_args()

    for d in args.path:
        append_time(d, not args.quiet)


if __name__ == "__main__":
    main()
