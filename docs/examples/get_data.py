import os

import xarray as xr


def download_file():
    os.system(
        "git clone https://oauth2:glpat-gS1qiEX7Ncoys5xH3CfS@gitlab.mpcdf.mpg.de/dave/xemc3-data/ ../../example-data/ --depth 1"
    )


def load_example_data(get_path=False):
    # Check whether we have a local copy
    for path in "./", "../", "../../":
        for path in [path, path + "example-data/"]:
            try:
                fn = path + "emc3_example.nc"
                if get_path:
                    xr.open_dataset(fn)
                    return fn[:-3]
                return xr.open_dataset(fn)
            except FileNotFoundError:
                pass
            except ValueError:
                print("Skipping non-readable file", fn)

    # try file from AFS
    try:
        ds = xr.open_dataset(
            "/afs/ipp-garching.mpg.de/u/dave/public/xemc3/emc3_example.nc"
        )
        print("Accessing example file via AFS, which might be slow ...")
        return ds
    except:  # noqa: E722
        # Might fail for a number of reasons ...
        pass

    # Last resort, download and use that
    print("Trying to download an example file to ../../example-data/ ...")
    download_file()
    print("done")
    return xr.open_dataset("../../example-data/emc3_example.nc")
