#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser

import xarray as xr

from ..core.load import archive, load_all
from ._common import commonparser, iter_dir

ext = "arch.nc"


def to_archive(
    read: str, name: str, quiet: bool, geom: bool, mapping: bool, delete: bool
) -> None:
    fromds = False
    if not quiet:
        print(f"Loading {read} ...", end="")
        sys.stdout.flush()
    if os.path.isdir(read):
        ds = load_all(read)
    else:
        ds = xr.open_dataset(read)
        fromds = True
    if not quiet:
        print(" writing ...", end="")
        sys.stdout.flush()
    archive(ds, f"{name}.{ext}", geom, mapping)
    if delete and fromds:
        if not quiet:
            print(" deleting ...", end="")
            sys.stdout.flush()
        os.unlink(read)
    if not quiet:
        print(" done")


def parser() -> ArgumentParser:
    parser = commonparser(
        "Archive the data from EMC3 simulations as a netcdf file.", ext=ext
    )
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
        help="Delete the uncompressed input file. "
        "Only used if the input was a netcdf file",
    )
    return parser


def main() -> None:
    args = parser().parse_args()

    for d, n in iter_dir(args):
        to_archive(
            d,
            n,
            args.quiet,
            geom=args.geometry,
            mapping=not args.no_mapping,
            delete=args.delete,
        )


if __name__ == "__main__":
    main()
