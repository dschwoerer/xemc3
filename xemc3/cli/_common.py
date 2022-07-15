import os
from argparse import ArgumentParser

from .. import __version__


def commonparser(desc, ext="nc") -> ArgumentParser:
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "path",
        nargs="+",
        help=f"""Path of the directory to load. The netcdf file will be
        called dir.{ext} if the folder was called dir.""",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Be less verbose")
    parser.add_argument(
        "-o",
        "--name",
        help=f"Specify the name for the output file. Defaults to `dir.{ext}` "
        f"when not given. Otherwise `name.{ext}` is used.",
    )
    parser.add_argument(
        "-V", "--version", action="version", version=f"xemc3 {__version__}"
    )
    return parser


def iter_dir(args):
    if len(args.path) > 1 and args.name:
        raise RuntimeError(
            """Providing an explicit output name and more than one folder to be read is
currently not supported. Either use the default output name or read one file
after another. Further you can use the python interface that gives more
control."""
        )
    for d in args.path:
        while d[-1] == "/":
            d = d[:-1]
        if args.name:
            yield d, args.name
        else:
            if os.path.isdir(d):
                if d == ".":
                    yield d, os.getcwd()
                else:
                    yield d, d
            else:
                yield d, d.rsplit(".", 1)[0]
