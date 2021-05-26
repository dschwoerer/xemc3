#!/usr/bin/env python3

from argparse import ArgumentParser
from ..core.load import load_all, archive
import xarray as xr
import sys
import os


def to_archive(d: str, quiet: bool, geom: bool, mapping: bool, delete: bool) -> None:
    while d[-1] == "/":
        d = d[:-1]
    fromds = False
    if not quiet:
        print(f"Loading {d} ...", end="")
        sys.stdout.flush()
    if os.path.isdir(d):
        ds = load_all(d)
    else:
        ds = xr.open_dataset(d)
        dorg = d
        d = ".".join(d.split(".")[:-1])
        fromds = True
    if not quiet:
        print(" writing ...", end="")
        sys.stdout.flush()
    archive(ds, d + ".arch.nc", geom, mapping)
    if delete and fromds:
        if not quiet:
            print(" deleting ...", end="")
            sys.stdout.flush()
        os.unlink(dorg)
    if not quiet:
        print(" done")


def parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Archive the data from EMC3 simulations as a netcdf file."
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="Path of the directory to load or a netcdf file. The netcdf file will be called dir.arc.nc if the folder was called dir.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    parser.add_argument(
        "-g", "--geometry", action="store_true", help="Also store geometry"
    )
    parser.add_argument(
        "-n",
        "--no-mapping",
        action="store_true",
        help="Do not include the mapping information.",
    )
    parser.add_argument(
        "-d",
        "--delete",
        action="store_true",
        help="Delete the uncompressed input file. Only used if the input was a netcdf file",
    )
    return parser


def main() -> None:
    args = parser().parse_args()

    for d in args.path:
        to_archive(
            d,
            args.quiet,
            geom=args.geometry,
            mapping=not args.no_mapping,
            delete=args.delete,
        )


if __name__ == "__main__":
    main()
