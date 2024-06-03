#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

from ..core import dataset
from ..core.load import load_all
from ._common import commonparser, iter_dir


def to_netcdf(read: str, name: str, quiet: bool, ignore_broken: bool) -> None:
    if not quiet:
        print(f"Loading {read} ...", end="")
        sys.stdout.flush()
    ds = load_all(read, ignore_broken=ignore_broken)
    if not quiet:
        print(" writing ...", end="")
        sys.stdout.flush()
    ds.emc3.to_netcdf(name + ".nc")
    if not quiet:
        print(" done")


def parser() -> ArgumentParser:
    cp = commonparser(
        "Load the data from EMC3 simulations and store as netcdf file. "
        "The data is written for each simulation to a netcdf file."
    )
    cp.add_argument(
        "--ignore-broken",
        action="store_true",
        help="Ignore if mapped files are incomplete",
    )
    return cp


def main() -> None:
    args = parser().parse_args()

    for d, n in iter_dir(args):
        to_netcdf(d, n, args.quiet, ignore_broken=args.ignore_broken)


if __name__ == "__main__":
    main()
