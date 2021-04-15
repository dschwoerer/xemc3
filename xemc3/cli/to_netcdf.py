#!/usr/bin/env python3

from argparse import ArgumentParser
from ..core.load import load_all
from ..core import dataset
import sys


def to_netcdf(d: str, quiet: bool = True) -> None:
    while d[-1] == "/":
        d = d[:-1]
    if not quiet:
        print(f"Loading {d} ...", end="")
        sys.stdout.flush()
    ds = load_all(d)
    if not quiet:
        print(" writing ...", end="")
        sys.stdout.flush()
    ds.emc3.to_netcdf(d + ".nc")
    if not quiet:
        print(" done")


def parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Load the data from EMC3 simulations and store as netcdf file. "
        "The data is written for each simulation to a netcdf file."
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="Path of the directory to load. The netcdf file will be called dir.nc if the folder was called dir.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    return parser


def main() -> None:
    args = parser().parse_args()

    for d in args.path:
        to_netcdf(d, args.quiet)


if __name__ == "__main__":
    main()
