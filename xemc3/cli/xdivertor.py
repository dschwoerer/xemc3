#!/usr/bin/env python3

import os
import sys
from argparse import ArgumentParser

import xarray as xr

import xemc3

long_names = {
    "f_n": "Particle flux",
    "f_E": "Energy flux",
    "avg_n": "Averge density",
    "avg_Te": "Average electron temperature",
    "avg_Ti": "Average ion temperature",
}


def get_name(key):
    try:
        key = long_names[key]
    except KeyError:
        pass
    return key


def get_key(name):
    return {v: k for k, v in long_names.items()}[name]


def plot(cwd, args):
    while cwd[-1] == "/":
        cwd = cwd[:-1]
    if os.path.isdir(cwd):
        cwd += "/"
        plates = xemc3.load.plates(cwd)
    else:
        plates = xr.open_dataset(cwd)
    key = args.key
    if key not in plates:
        key = get_key(key)
    if key == "f_E":
        plates[key].data /= 1e6
        plates[key].attrs["units"] = "M" + plates[key].attrs["units"]

    if args.range == "":
        args.range = ":"
    vmin, vmax = [
        None if s in ["None", ""] else float(s) for s in args.range.split(":")
    ]

    if args.cutoff:
        cutoff = plates["tot_P"].sum() / 100
    else:
        cutoff = None

    segments = 5 if args.plotall else 1

    if args.title:
        title = args.title[0]
    else:
        title = None
    plt1 = plt2 = None
    if args.plotsym and args.plotlower:
        plt1 = cwd + "divertor.png"
        plt2 = cwd + "divertor_zoomed.png"

    plt = plates.emc3.plot_div(
        key,
        symmetry=args.plotsym,
        segments=segments,
        vmax=vmax,
        vmin=vmin,
        power_cutoff=cutoff,
        verbose=not args.quiet,
        only_lower=args.plotlower,
        title=title,
        phi_slices=args.phi_slices,
        path1=plt1,
        path2=plt2,
    )

    if args.gui:
        plt.show()


def parser() -> ArgumentParser:
    parser = ArgumentParser(description="Plot the heatflux on the divertor")
    parser.add_argument(
        "-c",
        "--cutoff",
        action="store_true",
        help="Show only tile with high target power flux",
    )
    parser.add_argument(
        "-s",
        "--plotsym",
        action="store_true",
        help="Plot assuming stellarator symmetry",
    )
    parser.add_argument(
        "-a", "--plotall", action="store_true", help="Plot all 5 segments"
    )
    parser.add_argument(
        "-l", "--plotlower", action="store_true", help="Plot only the lower half"
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    parser.add_argument("-g", "--gui", action="store_true", help="Start the gui")
    parser.add_argument("-r", "--range", default="", help="Set datarange")
    parser.add_argument("-t", "--title", nargs=1, help="Title for plot")
    parser.add_argument(
        "--phi_slices", action="store_true", help="Show where phi in deg is an integer"
    )

    parser.add_argument(
        "-k",
        "--key",
        default="Energy flux",
        type=get_name,
        # choices=[
        #     "Particle flux",
        #     "Energy flux",
        #     "Averge density",
        #     "Average electron temperature",
        #     "Average ion temperature",
        # ],
        help="Data to plot",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"xemc3 {xemc3.__version__}"
    )
    parser.add_argument("path", nargs=1, help="Path of data")
    return parser


def main():
    if sys.argv[0] == "mayavi2" or "-x" in sys.argv:
        print("Run with `-g` in python?")
        args = parser().parse_args(sys.argv[3:])
    else:
        args = parser().parse_args()

    for cwd in args.path:
        plot(cwd, args)


if __name__ == "__main__":
    main()
