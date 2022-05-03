#!/usr/bin/env python3
import os
import sys
from argparse import ArgumentParser

import xarray as xr

from ..core import dataset
from ..core.load import load_all
from ._common import commonparser, iter_dir


def append_time(read: str, name: str, verbose: bool = False) -> None:
    try:
        os.replace(name + ".nc", name + ".old.nc")
        old = xr.open_dataset(name + ".old.nc")
    except FileNotFoundError:
        old = None
    try:
        if verbose:
            print(f"Loading {read} ...", end="")
            sys.stdout.flush()
        ds = load_all(read)
        if old is not None:
            ds = xr.concat([old, ds], "time", "different")
        if verbose:
            print(" writing ...", end="")
            sys.stdout.flush()
        ds.emc3.to_netcdf(name + ".nc")
    except Exception:
        if old is not None:
            os.replace(name + ".old.nc", name + ".nc")
        raise
    if old is not None:
        os.remove(name + ".old.nc")
    if verbose:
        print(" done")


def parser() -> ArgumentParser:
    return commonparser(
        """
        Load the data from EMC3 simulations and store as netcdf file. The
        data is appended for each simulation to the netcdf file, which
        will be created if it does not yet exists.
        """
    )


def main() -> None:
    args = parser().parse_args()

    for d, n in iter_dir(args):
        append_time(d, n, not args.quiet)


if __name__ == "__main__":
    main()
