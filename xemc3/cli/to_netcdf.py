#!/usr/bin/env python3

from argparse import ArgumentParser
import xemc3
import sys


def main():
    parser = ArgumentParser(
        "Load the data from EMC3 simulations and store as netcdf file. "
        "The data is written for each simulation to a netcdf file."
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="Path of the directory to load. The netcdf file will be called dir.nc if the folder was called dir.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    args = parser.parse_args()

    for d in args.path:
        while d[-1] == "/":
            d = d[:-1]
        if not args.quiet:
            print(f"Loading {d} ...", end="")
            sys.stdout.flush()
        ds = xemc3.load(d)
        if not args.quiet:
            print(" writing ...", end="")
            sys.stdout.flush()
        ds.emc3.to_netcdf(d + ".nc")
        if not args.quiet:
            print(" done")


if __name__ == "__main__":
    main()
