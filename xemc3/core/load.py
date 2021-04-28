from .utils import to_interval, timeit, prod, rrange
import os
import xarray as xr
import numpy as np
import re
import typing

try:
    from numpy.typing import DTypeLike
except ImportError:
    # Workaround for python 3.6
    DTypeLike = typing.Type  # type: ignore


def _fromfile(
    f: typing.TextIO, *, count: int, dtype: DTypeLike, **kwargs
) -> np.ndarray:
    """
    Read count amount of dtype from the textio stream, i.e. a file
    opened for reading. It always reads full lines. If there is more
    data on the line then it should read, all data is returened. If
    less data is read, then a shorter array is returned. Thus the
    calling function needs to handle the case that more or less data
    is returned.

    Parameters
    ----------
    f : textio
        the opened file
    count : int
        amount that should be read
    dtype : dtype
        the datatype to expect
    kwargs : dict
        Additional arguments for np.fromstring

    Returns
    -------
    np.ndarray
        The read data.
    """
    ret = np.empty(count, dtype=dtype)  # type: np.ndarray
    pos = 0
    bad = re.compile(r"(\d)([+-]\d)")

    while True:
        line = f.readline()
        if f == "":
            break
        line = bad.sub(r"\1E\2", line)
        new = np.fromstring(line, dtype=dtype, **kwargs)
        ln = len(new)
        try:
            ret[pos : pos + ln] = new
        except Exception as e:
            print(e)
            if pos + ln >= count:
                ret = np.append(ret[:pos], new)
                print(f"Returning {pos + ln} rather then {count} elements")
            else:
                raise
        pos += ln
        if ln == 0 or count <= pos:
            break
    if count != pos:
        ret = ret[:pos]
        return ret
    return ret


def _block_write(f: typing.TextIO, d: np.ndarray, fmt: str, bs: int = 10) -> None:
    d = d.flatten()
    l = (len(d) // bs) * bs
    np.savetxt(f, d[:l].reshape(-1, bs), fmt=fmt)
    if l != len(d):
        np.savetxt(f, d[l:], fmt=fmt)


def write_locations(ds: xr.Dataset, fn: str) -> None:
    """
    Write spatial positions of grid points in EMC3 format.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with the EMC3 simulation data
    fn : str
        The filename to write to
    """
    rs = ds.emc3["R_corners"]
    phis = ds.emc3["phi_corners"]
    zs = ds.emc3["z_corners"]
    fmt = "%10.6f"

    def write(f, d):
        d = d.transpose()
        _block_write(f, d, fmt)

    with open(fn, "w") as f:
        # np.savetxt(f, t, "%d")
        f.write("  ".join([str(t) for t in rs.shape]) + "\n")
        for i, phi in enumerate(phis):
            np.savetxt(f, [phi * 180 / np.pi], fmt=fmt)
            write(f, rs.isel(phi=i).data * 100)
            write(f, zs.isel(phi=i).data * 100)


def read_magnetic_field(fn: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Read magnetic fieldstrength from grid

    Parameters
    ----------
    fn : str
        The filename to read from
    ds : xr.Dataset
        A xemc3 dataset required for the dimensions

    Returns
    -------
    xr.Dataset
        The magenetic field strength
    """
    if "R_bounds" in ds:
        shape = ds.R_bounds.shape
        assert len(shape) == 6
        shape = [i + 1 for i in shape[:3]]
        dims = ds.R_bounds.dims[:3]
    else:
        shape = ds._plasma_map.shape
        shape = [i + 1 for i in shape]
        dims = ds._plasma_map.dims
    nx, ny, nz = shape
    with open(fn) as f:
        raw = _fromfile(f, dtype=float, count=nx * ny * nz, sep=" ")
        _assert_eof(f, fn)
    raw = raw.reshape(shape[::-1])
    raw = np.swapaxes(raw, 0, 2)
    return to_interval(dims, raw)


def write_magnetic_field(path: str, ds: xr.Dataset) -> None:
    """
    Write the magnetic field to a file

    Parameters
    ----------
    path : str
        The filename to write to
    ds : xr.Dataset
        The xemc3 dataset with the magnetic field
    """
    bf = ds.emc3["bf_corners"].data
    bf = bf.transpose(2, 1, 0)
    with open(path, "w") as f:
        _block_write(f, bf, "%7.4f")


def read_locations(fn: str) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read spatial positions of grid points

    Parameters
    ----------
    fn : str
        The filename of the grid

    Returns
    -------
    np.array
        phi positions
    np.array
        radial positions
    np.array
        z position
    """
    with open(fn) as f:
        nx, ny, nz = [int(i) for i in f.readline().split()]
        phidata = np.empty(nz)
        rdata = np.empty((nx, ny, nz))
        zdata = np.empty((nx, ny, nz))

        def read(f, nx, ny):
            t = _fromfile(f, dtype=float, count=nx * ny, sep=" ")
            t = t.reshape(ny, nx)
            t = t.transpose()
            return t

        # Read and directly convert to SI units
        for i in range(nz):
            phidata[i] = float(f.readline()) * np.pi / 180.0
            rdata[:, :, i] = read(f, nx, ny) / 100
            zdata[:, :, i] = read(f, nx, ny) / 100

        _assert_eof(f, fn)

    return phidata, rdata, zdata


def read_plates_mag(fn: str, ds: xr.Dataset) -> xr.DataArray:
    """
    Read the magnetic plates

    Parameters
    ----------
    fn : str
        Location of the magnetic plates files
    ds : xr.Dataset
        xemc3 dataset

    Returns
    -------
    xr.DataArray
        The magnetic plates
    """
    shape = [x.shape[0] for x in [ds.r, ds.theta, ds.phi]]
    ret = np.zeros(shape, dtype=bool)
    with open(fn) as f:
        for line in f:
            lines = [int(x) for x in line.split()]
            zone, r, theta, num = lines[:4]
            assert zone == 0
            assert num % 2 == 0
            for t in range(num // 2):
                a, b = lines[4 + 2 * t : 6 + 2 * t]
                ret[r, theta, a : b + 1] = True
    return xr.DataArray(data=ret, dims=("r", "theta", "phi"))


def write_plates_mag(fn: str, ds: xr.Dataset) -> None:
    """
    Write the ``PLATES_MAG`` info to a file in the EMC3 format.

    Parameters
    ----------
    fn : str
        The file to write to
    ds : xr.Dataset or xr.DataArray
       either the PLATES_MAG dataarray or a dataset containing PLATES_MAG
    """
    if isinstance(ds, xr.Dataset):
        da = ds["PLATES_MAG"]
    else:
        da = ds
    with open(fn, "w") as f:
        # iterate over r:
        for r, slice2d in enumerate(da.data):
            if not slice2d.any():
                continue
            for theta, slice1d in enumerate(slice2d):
                if not slice1d.any():
                    continue
                if slice1d.all():
                    ab = [0, len(slice1d)]
                else:
                    last = False
                    ab = []
                    for i, v in enumerate(slice1d):
                        if v != last:
                            if v or i == 0:
                                ab.append(i)
                            else:
                                ab.append(i - 1)
                            last = v
                    if last:
                        ab.append(i)
                out = [0, r, theta, len(ab)] + ab
                outi = [str(i) for i in out]
                outs = "   %s   %3s   %3s" % tuple(out[:3])
                outs += " ".join(["    %2s" % i for i in outi[3:]])
                f.write(outs + "\n")
    return


def read_mappings(fn: str, dims: typing.Sequence[int]) -> xr.DataArray:
    """
    Read the mappings data

    Parameters
    ----------
    fn : str
        The path of the file to be read
    dims : tuple of int
        The shape of the domain

    Returns
    -------
    xr.DataArray
        The mapping information
    """
    with open(fn) as f:
        # dims =  [int(i) for i in f.readline().split()]
        dat = f.readline()
        infos = [int(i) for i in dat.split()]
        # print(infos, prod(dims))
        t = _fromfile(f, dtype=int, count=prod(dims), sep=" ")
        # fortran indexing
        t -= 1
        t = t.reshape(dims, order="F")
        _assert_eof(f, fn)
    # print(infos, prod(dims), np.max(t), np.min(t))
    da = xr.DataArray(dims=("r", "theta", "phi"), data=t)
    da.attrs = dict(numcells=infos[0], plasmacells=infos[1], other=infos[2])
    return da


def write_mappings(da: xr.DataArray, fn: str) -> None:
    """Write the mappings data to fortran"""
    with open(fn, "w") as f:
        infos = da.attrs["numcells"], da.attrs["plasmacells"], da.attrs["other"]
        f.write("%12d %11d %11d\n" % infos)
        _block_write(f, da.data.flatten(order="F") + 1, " %11d", 6)


def get_locations(path: str, ds: xr.Dataset = None) -> xr.Dataset:
    """
    Read locations from folder path and add to dataset (if given).

    Parameters
    ----------
    path : str
        folder from which to read
    ds : xr.Dataset (optional)
        The dataset to add to. If not given return a new dataset.

    Returns
    -------
    xr.Dataset
        A dataset with the coordinates set
    """
    if ds is None:
        ds = xr.Dataset()
    assert isinstance(ds, xr.Dataset)
    phi, r, z = read_locations(path + "/GRID_3D_DATA")
    ds = ds.assign_coords(
        {
            "R_bounds": to_interval(("r", "theta", "phi"), r),
            "z_bounds": to_interval(("r", "theta", "phi"), z),
            "phi_bounds": to_interval(("phi",), phi),
        }
    )
    ds.emc3.unit("R_bounds", "m")
    ds.emc3.unit("z_bounds", "m")
    assert isinstance(ds, xr.Dataset)
    return ds


def scrape(f: typing.TextIO, *, ignore=None, verbose=False) -> str:
    """read next data line from configuration file (skip lines with
    leading *)

    Parameters
    ----------
    f : textio
        fileobject (opened file) to read from
    ignore : (str or None)
        if any of the characters are given, interpret as comment and
        remove. In this case only return non-empty lines.

    Returns
    ------
    str
        the first non-ignored line
    """
    while True:
        s = f.readline()
        if s.startswith("*"):
            if verbose:
                if s.startswith("***"):
                    print(s[:-1])
            continue
        if ignore is not None:
            for i in ignore:
                if i in s:
                    s = s[: s.index(i)]
            if s.strip() == "":
                continue
        return s


def _assert_eof(f: typing.TextIO, fn: str) -> None:
    """Ensure the opened file is read to the end.

    Parameters
    ----------
    f : textio
        the opened file
    fn : str
        The name of the file, used for the error.

    Raises
    ------
    AssertionError
        if the file is not at the end.
    """
    test = _fromfile(f, dtype=float, count=2, sep=" ")
    assert len(test) == 0, f"Expected EOF, but found more data in {fn}"


def read_plate(filename: str) -> typing.Tuple[np.ndarray, ...]:
    """
    Read Target structures from a file that is in the Kisslinger
    format as used by EMC3. It returns the coordinates as plain array
    in the order or R, z, phi.

    Parameters
    ----------
    filename : str
        The location of the file to read

    Returns
    -------
    np.ndarray
        The major radius coordinates of the corners in meters.
    np.ndarray
        The z coordinate of the corners in meters
    np.ndarray
        the phi coordinate of the corners in radian
    """
    with open(filename) as f:
        # first line is a comment ...
        _ = next(f)
        setup = next(f).split()
        assert len(setup) == 5, f"Expected 5 values but got {setup}"
        for zero in setup[3:]:
            assert float(zero) == 0.0
        nx, ny, nz = [int(i) for i in setup[:3]]
        r = np.empty((nx, ny))
        z = np.empty((nx, ny))
        phi = np.empty(nx)
        for x in range(nx):
            phi[x] = float(next(f)) / 180 * np.pi
            for y in range(ny):
                s = next(f)
                try:
                    try:
                        r[x, y], z[x, y] = [float(i) / 100 for i in s.split()]
                    except ValueError:
                        s = s.split("!")[0]
                        r[x, y], z[x, y] = [float(i) / 100 for i in s.split()]
                except ValueError:
                    raise ValueError(f"Error with {s} in {filename}")
        _assert_eof(f, filename)
        return (r, z, phi)


plates_labels = ["f_n", "f_E", "avg_n", "avg_Te", "avg_Ti"]


def read_plates_raw(cwd: str, fn: str) -> typing.Sequence[xr.Dataset]:
    """
    Read the target mapped info to a list of xr.Datasets

    Parameters
    ----------
    cwd : str
        The folder in which to look for the files (with trailing slash)
    fn : str
        the relative name of the file to read

    Returns
    -------
    list of xr.Dataset
        The read data as a list of datasets
    """
    with open(cwd + fn) as f:
        s = scrape(f, ignore="!").split()
        _, num_plates = [int(i) for i in s]
        plates = []
        for plate in range(num_plates):
            s = scrape(f, ignore="!").split()
            assert len(s) == 2, f"{plate} : {s}"
            _, geom = s
            r, z, phi = read_plate(cwd + geom)
            nx, ny = r.shape
            nx -= 1
            ny -= 1
            s = scrape(f, ignore="!").split()
            total = np.array([float(i) for i in s])
            s = scrape(f, ignore="!").split()
            if len(s) == 3:
                items, yref, xref = [int(i) for i in s]
                nxr = nx * xref
                nyr = ny * yref
                mode = 2
            elif len(s) == 2:
                nyr, nxr = [int(i) for i in s]
                xref = nxr // nx
                yref = nyr // ny
                items = nx * ny
                mode = 1
            assert items == nx * ny
            if mode == 2:
                data = _fromfile(f, dtype=float, count=nx * ny * 12, sep=" ")

                corrs = [data[4 * items * i : 4 * items * (i + 1)] for i in range(3)]
                corrs[0] /= 100
                corrs[1] /= 100
                # phi is in rads ?!?
                # corrs[2] /= 180/np.pi
                corrs = [a.reshape(nx, ny, 4) for a in corrs]
            else:
                assert mode == 1
                data = _fromfile(
                    f, dtype=float, count=(nxr + 1) * (nyr + 1) * 3, sep=" "
                )

                coordinates = [
                    data[(nxr + 1) * (nyr + 1) * i : (nxr + 1) * (nyr + 1) * (i + 1)]
                    for i in range(3)
                ]
                coordinates[0] /= 100
                coordinates[1] /= 100
                coordinates = [a.reshape(nxr + 1, nyr + 1) for a in coordinates]

            data = _fromfile(f, dtype=float, count=nxr * nyr * 5, sep=" ")
            if mode == 2:
                data = data.reshape((5, nx, ny, xref, yref))
                data = data.transpose((0, 1, 3, 2, 4))
            data = data.reshape(5, nxr, nyr)
            if phi[0] > phi[1]:
                phi = phi[::-1]
                r = r[::-1, ...]
                z = z[::-1, ...]
                data = data[:, ::-1, ...]
                if mode == 2:
                    corrs = [a[::-1, ...] for a in corrs]
                else:
                    coordinates = [a[::-1, ...] for a in coordinates]

            if mode == 2:
                A = np.array(
                    [(1.0, a, b, a * b) for a, b in [(0, 0), (0, 1), (1, 1), (1, 0)]]
                ).T
                Ai = np.linalg.inv(A)
                a0, a1 = 0, 1
                b0, b1 = 0, 1
                A = np.array(
                    [
                        (1.0, a, b, a * b)
                        for a, b in [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
                    ]
                ).T
                for i, pos in enumerate(corrs):
                    # Transform into coefficient form
                    pos = pos @ Ai
                    newpos = np.empty((nxr, nyr, 4))
                    for ix, iy in rrange((xref, yref)):
                        # Get relative coordinates
                        a0, a1 = ix / xref, (ix + 1) / xref
                        b0, b1 = iy / yref, (iy + 1) / yref
                        A = np.array(
                            [
                                (1.0, b, a, a * b)
                                for a, b in [(a0, b0), (a0, b1), (a1, b0), (a1, b1)]
                            ]
                        ).T
                        newpos[ix::xref, iy::yref, :] = pos @ A

                    corrs[i] = newpos
                corrs = [
                    xr.DataArray(
                        data=a.reshape(nxr, nyr, 2, 2),
                        dims=("phi", "x", "delta_phi", "delta_x"),
                    )
                    for a in corrs
                ]
            else:
                assert mode == 1
                corrs = [to_interval(("phi", "x"), a) for a in coordinates]

            coords = {
                "R_bounds": corrs[0],
                "z_bounds": corrs[1],
                "phi_bounds": corrs[2],
            }
            ds = xr.Dataset(coords=coords)  # type: ignore
            ds.coords["R_bounds"].attrs["units"] = "m"
            ds.coords["z_bounds"].attrs["units"] = "m"
            ds.coords["phi_bounds"].attrs["units"] = "radian"
            long_names = {
                "f_n": "Particle flux",
                "f_E": "Energy flux",
                "avg_n": "Averge density",
                "avg_Te": "Average electron temperature",
                "avg_Ti": "Average ion temperature",
            }
            fac = {
                "f_n": 1,
                "f_E": 1e4,
                "avg_n": 1e6,
                "avg_Te": 1,
                "avg_Ti": 1,
            }
            units = {
                "f_n": None,
                "f_E": "W/m²",
                "avg_n": "m^-3",
                "avg_Te": "eV",
                "avg_Ti": "eV",
            }

            for i, l in enumerate(plates_labels):
                ds[l] = ("phi", "x"), data[i] * fac[l]
                if units[l] is not None:
                    ds[l].attrs["units"] = units[l]
                ds[l].attrs["long_name"] = long_names[l]
            for i, l in enumerate(["tot_n", "tot_P"]):
                ds[l] = total[i]
            plates.append(ds)

        # Make sure we have read everything
        _assert_eof(f, cwd + fn)
    return plates


def plates_raw_to_ds(plates: typing.Sequence[xr.Dataset]) -> xr.Dataset:
    """
    Convert a list of datasets to one dataset
    """
    dims = {d: 0 for d in plates[0].dims}
    for plate in plates:
        for k, v in plate.dims.items():
            if dims[k] < v:
                dims[k] = v
    ds = xr.Dataset()

    matching = {}
    for k, v in dims.items():
        matching[k] = True
        org_dims = []
        for plate in plates:
            org_dims.append(plate.dims[k])
            if plate.dims[k] != v:
                matching[k] = False
        if not matching[k]:
            assert isinstance(k, str)
            ds[k + "_dims"] = ("plate_ind", org_dims)

    dims["plate_ind"] = len(plates)

    def merge(var):
        shape = [dims["plate_ind"]] + [dims[d] for d in plates[0][var].dims]
        data = np.empty(shape)
        data[...] = np.nan
        for i, plate in enumerate(plates):
            tmp = plate[var]
            data[tuple([i] + [slice(None, i) for i in tmp.shape])] = tmp
        return ["plate_ind"] + list(plates[0][var].dims), data

    for coord in plates[0].coords.keys():
        ds = ds.assign_coords({coord: merge(coord)})

    for var in plates[0]:
        ds[var] = merge(var)
        ds[var].attrs = plates[0][var].attrs

    return ds


def load_plates(cwd: str) -> xr.Dataset:
    # Deprecate?
    if cwd[-1] != "/":
        cwd += "/"
    with timeit("\nReading raw: %f"):
        plates = read_plates_raw(cwd, "TARGET_PROFILES")
    with timeit("To xarray: %f"):
        return plates_raw_to_ds(plates)


def write_plates(cwd: str, plates: xr.Dataset) -> None:
    # Deprecate?
    with timeit("Writing ncs: %f"):
        # Note we really should compress to get rid of the NaN's we added
        plates.to_netcdf(
            f"{cwd}/TARGET_PROFILES.nc",
            encoding={
                i: {"zlib": True, "complevel": 1}
                for i in [i for i in plates] + [i for i in plates.coords]
            },
        )


def read_plates(cwd: str) -> xr.Dataset:
    # Deprecate?
    ds = xr.open_dataset(f"{cwd}/TARGET_PROFILES.nc")
    assert isinstance(ds, xr.Dataset)
    return ds


def get_plates(cwd: str, cache: bool = True) -> xr.Dataset:
    """
    Read the target fluxes from the EMC3_post procesing routine

    Parameters
    ----------
    cwd : str
        The directory in which the file is
    cache : bool (optional)
        Whether to check whether a netcdf file is present, and if not
        create one. This can result in faster reading the next time.

    Returns
    -------
    xr.Dataset
        The target profiles
    """
    if cache:
        try:
            if os.path.getmtime(cwd + "/TARGET_PROFILES.nc") > os.path.getmtime(
                cwd + "/TARGET_PROFILES"
            ):
                return read_plates(cwd)
        except OSError:
            pass
        data = load_plates(cwd)
        write_plates(cwd, data)
        return data
    return load_plates(cwd)


def ensure_mapping(
    dir: str,
    mapping: typing.Union[None, xr.Dataset, xr.DataArray] = None,
    need_mapping: bool = True,
    fn: typing.Union[None, str] = None,
) -> xr.Dataset:
    """
    Ensure that basic infos are present to read datafiles.

    Parameters
    ----------
    dir : str
        The folder in which to look for files
    mapping : None or xr.Dataset or xr.DataArray
        Either the required mapping, in which case nothing is done, or
        None, in which case the data is read.
    need_mapping : bool (optional)
        If true the mapping infomation has to be loaded, otherwise the
        shape of the data is sufficient. Defaults to True.
    fn : str (optional)
        The file to read, used for a potential error message

    Returns
    -------
    xr.Dataset
        A dataset with the required information

    Raises
    ------
    FileNotFoundError
        If the required info could not be read
    """
    if dir == "":
        dir = "."
    if mapping is None:
        try:
            mapping = get_locations(dir)
            if need_mapping:
                mapping = read_fort_file(mapping, f"{dir}/fort.70", **files["fort.70"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"""Reading {fn+ ' ' if fn is not None else ''} mapped requires mapping information, but the required
infomation in '{dir}' could not be found.  Ensure all files are present
in the folder or pass in a dataset that contains the mapping
informations."""
            )
    else:
        if isinstance(mapping, xr.DataArray):
            mapping = xr.Dataset(dict(_plasma_map=mapping))
    return mapping


def read_var(
    dir: str, var: str, ds: typing.Union[None, xr.Dataset, xr.DataArray] = None
) -> xr.Dataset:
    """
    Read the variable var from the simulation in directory dir.

    Parameters
    ----------
    dir: str
        The path of the simulation directory
    var: str
        The name of the variable to be read
    ds: xr.DataArray or xr.Dataset or None (optional)
        An xemc3 Dataset or an DataArray containing the mapping. If
        not given and a mapping is required to read the file, it is
        read from the directory.

    Returns
    -------
    xr.Dataset
        A dataset in wich at least ``var`` is set.
    """
    for fn, fd in files.items():
        if "vars" not in fd:
            continue
        for v in fd["vars"]:
            if v == var or (
                v.endswith("%d")
                and var.startswith(v[:-2])
                and var[len(v[:-2]) :].isdigit()
            ):
                ds = ensure_mapping(dir, ds, fd.get("type", "mapped") == "mapped")
                assert isinstance(ds, xr.Dataset)
                return read_fort_file(ds, f"{dir}/{fn}", **fd)
    raise ValueError(f"Don't know how to read {var}")


def read_mapped(
    fn: str,
    mapping: typing.Union[xr.Dataset, xr.DataArray],
    skip_first: int = 0,
    ignore_broken: bool = False,
    kinetic: bool = False,
    dtype: DTypeLike = float,
    squeeze: bool = True,
) -> typing.Sequence[xr.DataArray]:
    """
    Read a file with the emc3 mapping.

    Parameters
    ----------
    fn : str
        The full path of the file to read
    mapping : xr.DataArray or xr.Dataset
        The dataarray of the mesh mapping or a dataset containing the
        mapping
    skip_first : int (optional)
        Ignore the first n lines. Default 0
    ignore_broken: boolean (optional)
        if incomplete datasets at the end should be ignored. Default: False
    kinetic : boolen (optional)
        The file contains also data for cells that are only evolved by
        EIRENE, rather then EMC3. Default: False
    dtype : datatype (optional)
        The type of the data, e.g. float or int. Is passed to
        numpy. Default: float
    squeeze : bool
        If True return a DataArray if only one field is read.

    Returns
    -------
    xr.DataArray or list of xr.DataArray
        The data that has been read from the file. If squeeze is True
        and only one field is read only a sinlge DataArray is
        returend.
    """

    if isinstance(mapping, xr.Dataset):
        mapping = mapping["_plasma_map"]
    if kinetic:
        max = np.max(mapping.data) + 1
    else:
        max = mapping.attrs["plasmacells"]
    firsts = []
    with open(fn) as f:
        raws = []
        while True:
            if skip_first:
                first = ""
                for _ in range(skip_first):
                    first += f.readline()
                firsts.append(first)
            raw = _fromfile(f, dtype=dtype, count=max, sep=" ")
            if ignore_broken and len(raw) > max:
                print(f"Ignoring data: {raw[max:]}")
                raw = raw[:max]
            if len(raw) == max:
                raws.append(raw)
            elif len(raw) == 0:
                break
            else:
                if ignore_broken:
                    print(f"Ignoring {len(raw)} data points, {max} required")
                    break
                print(raw)
                raise RuntimeError(
                    f"Incomplete dataset found ({len(raw)} out of {max}) after reading {len(raws)} datasets of file {fn}"
                )

    def to_da(raw):
        out = np.ones(mapping.shape) * np.nan
        mapdat = mapping.data
        for ijk in rrange(mapping.shape):
            mapid = mapdat[ijk]
            if mapid < max:
                out[ijk] = raw[mapid]
        return xr.DataArray(data=out, dims=mapping.dims)

    das = [to_da(raw) for raw in raws]
    if skip_first:
        for first, da in zip(firsts, das):
            da.attrs["print_before"] = first
    if squeeze and len(das) == 1:
        das = das[0]
    return das


def write_mapped_nice(ds: xr.Dataset, dir: str, fn: str = None, **args) -> None:
    """
    Write a file for EMC3 using the mapped format.

    Parameters
    ----------
    ds : xr.Dataset
        A xemc3 dataset
    dir : str
        The directory to which to write the file
    fn : str or None (optional)
        In case of ``None`` all mapped files are written.
        In the case of a str only that file is written.
        Any missing data is ignored. Thus if a file is specified, but
        the data hasn't been loaded this is ignored.
    args : dict (optional)
        Can be used to overwrite options for writing. Defaults to the
        options used for that file.
    """
    meta: typing.Dict[str, typing.Any]
    if fn is None:
        for fn, meta in files.values():  # type: ignore
            assert isinstance(meta, dict)
            if meta["type"] == "mapped":
                write_mapped_nice(ds, dir, fn, **args)
    else:
        meta = files[fn]
        meta = meta.copy()
        meta.update(args)
        meta.pop("type", "ignore")
        ignore_missing = meta.pop("ignore_missing", True)
        try:
            vars = get_vars_for_file(ds, meta.pop("vars"))
        except KeyError:
            if not ignore_missing:
                raise
            return
        datas = [ds[k] for k, _ in vars]
        for i, ops in enumerate([m for _, m in vars]):
            if "scale" in ops:
                at = datas[i].attrs
                datas[i] = datas[i] / ops["scale"]
                datas[i].attrs = at
        assert (
            datas != []
        ), f"Requested to write file {dir}/{fn} but required data not found."
        write_mapped(datas, ds["_plasma_map"], f"{dir}/{fn}", **meta)


def get_vars_for_file(
    ds: xr.Dataset, fn: typing.Union[str, dict]
) -> typing.List[typing.Tuple[str, dict]]:
    """
    Check if the variables of fn are loaded.

    Parameters
    ----------
    ds : xr.Dataset
        An xemc3 dataset
    fn : str or dict
        A file name to check or the associated dictionary of mappings

    Returns
    -------
    list of  tuple of str and dict
        The variables that are from that file as well as a dict
        containing additional info about the variable.

    Raises
    ------
    KeyError
        If the file has not been loaded
    """
    vars: dict
    if isinstance(fn, dict):
        vars = fn
    else:
        vars = files[fn]["vars"].copy()
    keys: typing.List[typing.Tuple[str, dict]] = []
    for var in vars:
        # Equivalent writing code:
        # if "%" in keys[-1]:
        #     key = keys[-1]
        #     flexi = vars.pop(key)
        #     for i in range(len(vars), len(datas)):
        #         vars[key % i] = flexi
        if "%" in var:
            while True:
                i = len(keys)
                if var % i not in ds:
                    break
                keys.append((var % i, vars[var]))
        else:
            keys.append((var, vars[var]))
    if keys == []:
        raise KeyError(f"Didn't find any key for {fn}")
    for k, _ in keys:
        if k not in ds:
            raise KeyError(f"Key {k} not present in Dataset, only have {ds.keys()}")
    return keys


def write_mapped(
    datas,
    mapping,
    fn,
    skip_first=0,
    ignore_broken=False,
    kinetic=False,
    dtype=None,
    fmt=None,
):
    """
    Write a some files using the EMC3 mapping.

    Parameters
    ----------
    datas : list of xr.DataArray
        The data to be written
    mapping : xr.DataArray
        The info about the EMC3 data mapping
    fn : str
        The filename to write to
    skip_first : None or int
        If not None, it needs to be the number of lines that are in
        the `print_before` attribute.
    ignore_broken : any
        ignored.
    kinetic : boolean
        If true the data is defined also outside of the plasma region.
    dtype : any
        if not None, it needs to match the dtype of the data
    fmt : None or str
        The Format to be used for printing the data.
    """
    if kinetic:
        max = np.max(mapping.values) + 1
    else:
        max = np.max(mapping.attrs["plasmacells"])
    if not isinstance(datas, list):
        datas = [datas]
    # if dtype:
    #     if not dtype == datas[0].data.dtype:
    #         raise AssertionError(
    #             f"Expected dtype {dtype} but data has "
    #             f"actually {datas[0].data.dtype} for "
    #             f"file {fn}"
    #         )
    if skip_first is not None:
        for d in datas:
            if skip_first:
                assert d.attrs["print_before"] != ""
            else:
                if "print_before" in d.attrs:
                    assert d.attrs["print_before"] == ""
    if dtype is None:
        dtype = datas[0].values.dtype
    # assert all([d.data.dtype == dtype for d in datas])
    out = np.zeros((len(datas), max))
    mapdat = mapping.values
    assert mapdat.dtype == int
    for j, data in enumerate(datas):
        datdat = data.values
        count = np.zeros(max)
        for ijk in rrange(mapdat.shape):
            mapid = mapdat[ijk]
            if mapid < max:
                if not np.isnan(datdat[ijk]):  # and mapid:
                    out[j, mapid] += datdat[ijk]
                    count[mapid] += 1
        out[j, :] /= count
    with open(fn, "w") as f:
        for i, da in zip(out, datas):
            if "print_before" in da.attrs:
                f.write(da.attrs["print_before"])
            if fmt is None:
                tfmt = "%.4e" if dtype != int else "%d"
            else:
                tfmt = fmt
            _block_write(f, i, fmt=tfmt, bs=6)


files: typing.Dict[str, typing.Dict[str, typing.Any]] = {
    "fort.70": dict(type="mapping", vars={"_plasma_map": dict()}),
    "fort.31": dict(
        type="mapped",
        skip_first=1,
        kinetic=False,
        vars={
            "ne": dict(
                scale=1e6,
                attrs=dict(units="m$^{-3}$", long_name="Electron density"),
            ),
            "nZ%d": dict(
                scale=1e6,
                attrs=dict(units="m$^{-3}$"),
            ),
        },
    ),
    "fort.33": dict(
        type="mapped", vars={"M": dict(attrs=dict(long_name="Mach number"))}
    ),
    "fort.30": dict(
        type="mapped",
        vars={
            "Te": dict(attrs=dict(units="eV", long_name="Electron temperature")),
            "Ti": dict(attrs=dict(units="eV", long_name="Ion temperature")),
        },
    ),
    "CONNECTION_LENGTH": dict(
        type="mapped",
        vars={
            "Lc": dict(scale=1e-2, attrs=dict(units="m", long_name="Connection length"))
        },
    ),
    "DENSITY_A": dict(
        type="mapped",
        kinetic=True,
        vars=dict(
            nH=dict(
                scale=1e6,
                attrs=dict(units="m$^{-3}$", long_name="Atomic deuterium density"),
            )
        ),
    ),
    "DENSITY_M": dict(
        kinetic=True,
        vars=dict(
            nH2=dict(
                scale=1e6,
                attrs=dict(units="m$^{-3}$", long_name="D_2 density"),
            )
        ),
    ),
    "DENSITY_I": dict(
        kinetic=True,
        vars={
            "nH2+": dict(
                scale=1e6, attrs=dict(units="m$^{-3}$", long_name="D_2^+ density")
            )
        },
    ),
    "TEMPERATURE_A": dict(
        kinetic=True,
        vars={
            "TH": dict(attrs=dict(units="eV", long_name="Atomic hydrogen temperature"))
        },
    ),
    "TEMPERATURE_M": dict(
        kinetic=True,
        vars={
            "TH": dict(attrs=dict(units="eV", long_name="Atomic hydrogen temperature"))
        },
    ),
    "BFIELD_STRENGTH": dict(
        type="full",
        vars={
            "bf_bounds": dict(
                attrs=dict(units="T", long_name="Magnetic field strength")
            )
        },
    ),
    "PLATES_MAG": dict(
        type="plates_mag",
        vars={
            "PLATES_MAG": dict(
                attrs=dict(long_name="Cells that are within or behind plates")
            )
        },
    ),
    # Some files - but don't know what they are
    "TEMPERATURE_I": dict(
        type="mapped",
        kinetic=True,
        vars={"TEMPERATURE_I_%d": dict()},
    ),
    "DENSITY_E_A": dict(
        type="mapped",
        kinetic=True,
        vars={"DENSITY_E_A_%d": dict()},
    ),
    "DENSITY_E_I": dict(
        type="mapped",
        kinetic=True,
        vars={"DENSITY_E_I_%d": dict()},
    ),
    "DENSITY_E_M": dict(
        type="mapped",
        kinetic=True,
        vars={"DENSITY_E_M_%d": dict()},
    ),
    "fort.40": dict(
        type="mapped",
        vars={"fort.40_%d": dict()},
    ),
    "fort.42": dict(
        type="mapped",
        vars={"fort.42_%d": dict()},
    ),
    "fort.43": dict(
        type="mapped",
        vars={"fort.43_%d": dict()},
    ),
    "fort.46": dict(
        type="mapped",
        vars={"fort.46_%d": dict()},
    ),
    "fort.47": dict(
        type="mapped",
        vars={"fort.47_%d": dict()},
    ),
    "IMPURITY_IONIZATION_SOURCE": dict(
        type="mapped",
        vars={"IMPURITY_IONIZATION_SOURCE_%d": dict()},
    ),
    "IMPURITY_NEUTRAL": dict(
        type="mapped",
        vars={"IMPURITY_NEUTRAL_%d": dict()},
    ),
    "IMP_RADIATION": dict(
        type="mapped",
        vars={"IMP_RADIATION_%d": dict()},
    ),
    "FLUX_CONSERVATION": dict(
        type="mapped",
        vars={"FLUX_CONSERVATION_%d": dict()},
    ),
    "LG_CELL": dict(
        type="mapped",
        dtype=int,
        vars={"LG_CELL_%d": dict()},
    ),
}

if False:
    _files_bak = files.copy()
    for k in files:
        _files_bak[k] = files[k].copy()
        if isinstance(files[k], dict):
            for l in files[k]:
                try:
                    _files_bak[k][l] = files[k][l].copy()
                except:
                    try:
                        _files_bak[k][l] = files[k][l][:]
                    except:
                        pass
else:
    _files_bak = files


def read_fort_file_pub(
    fn: str, ds: typing.Union[None, xr.Dataset, xr.DataArray] = None, **opts
) -> xr.Dataset:
    """
    Read a EMC3 simulation file. The expected content is derived from
    the filename.

    Parameters
    ----------
    fn : str
        The location of the file to read
    ds : xr.DataArray or xr.Dataset or None
        The mapping or a dataset containing a mapping or None. If one
        is needed and none is provided, it is read from the folder.
    opts : dict
        Additional options depending on the type of file that is read.

    Returns
    -------
    xr.Dataset
        The read variable is set. If a Dataset was given, the newly
        read entries from the file are added. Otherwise a new Dataset
        is returned with the read variables added.
    """
    filename = fn.split("/")[-1]
    defaults = files[filename].copy()
    defaults.update(opts)
    type = defaults.get("type", "mapped")
    ds = ensure_mapping("/".join(fn.split("/")[:-1]), ds, type == "mapped")
    assert isinstance(ds, xr.Dataset)
    return read_fort_file(ds, fn, **defaults)


def read_fort_file(ds: xr.Dataset, fn: str, type: str = "mapped", **opts) -> xr.Dataset:
    """
    Read an EMC3 simulation file and add to a dataset.

    """
    assert files == _files_bak
    if type == "mapping":
        opts.pop("vars", False)
        ds["_plasma_map"] = read_mappings(fn, ds.R_bounds.data.shape[:3])
        assert opts == {}, "Unexpected arguments: " + ", ".join(
            [f"{k}={v}" for k, v in opts.items()]
        )
        return ds
    elif type == "mapped":
        vars = opts.pop("vars")
        datas = read_mapped(fn, ds["_plasma_map"], **opts, squeeze=False)
        opts = {}
    elif type == "full":
        vars = opts.pop("vars")
        datas = [read_magnetic_field(fn, ds)]
    elif type == "plates_mag":
        vars = opts.pop("vars")
        datas = [read_plates_mag(fn, ds)]
    else:
        raise RuntimeError(f"Unexpected type {type}")
    assert files == _files_bak
    assert opts == {}, "Unexpected arguments: " + ", ".join(
        [f"{k}={v}" for k, v in opts.items()]
    )
    vars = vars.copy()
    assert files == _files_bak
    assert opts == {}
    keys = [k for k in vars]
    if "%" in keys[-1]:
        key = keys[-1]
        flexi = vars.pop(key)
        for i in range(len(vars), len(datas)):
            vars[key % i] = flexi

    assert files == _files_bak
    assert len(vars) == len(
        datas
    ), f"in file {fn} we found {len(datas)} fields but only {len(vars)} vars are given!"
    assert files == _files_bak
    for (var, varopts), data in zip(vars.items(), datas):
        ds[var] = data
        varopts = varopts.copy()
        scale = varopts.pop("scale", 1)
        if scale != 1:
            ds[var].data *= scale
        attrs = varopts.pop("attrs", {})
        ds[var].attrs.update(attrs)
        assert varopts == {}
    assert files == _files_bak
    return ds


def load_all(path, ignore_missing=None):
    """
    Load all data from a path and return as dataset

    Parameters
    ----------
    path : str
         Directory that contains files to be read
    ignore_missing : None or bool
         True: ignore missing files.
         False: raise exceptions if a file is not found.
         None: use default option for that file.

    Returns
    -------
    xr.Dataset
        A dataset with all the simulation data. Unless
        ``ignore_missing=False`` has been set, some data might be
        missing because it was not found. However
        ``ignore_missing=False`` has the disadvantage that files added
        in later versions will cause errors if the files are not
        present.
    """
    ds = get_locations(path)
    for fn, opts in files.items():
        opts = opts.copy()
        try:
            ds = read_fort_file(ds, f"{path}/{fn}", **opts)
        except FileNotFoundError:
            if ignore_missing is None:
                if not opts.get("ignore_missing", True):
                    raise
            elif not ignore_missing:
                raise
    return ds


def write_fort_file(ds, dir, fn, type="mapped", **opts):
    if type == "mapping":
        write_mappings(ds["_plasma_map"], f"{dir}/{fn}")
    elif type == "mapped":
        write_mapped_nice(ds, dir, fn)
    elif type == "full":
        vars = opts.pop("vars")
        assert len(vars) == 1
        for var in vars:
            assert var == "bf_bounds"
            write_magnetic_field(f"{dir}/{fn}", ds)
    elif type == "plates_mag":
        vars = opts.pop("vars")
        write_plates_mag(f"{dir}/{fn}", ds)
    else:
        raise RuntimeError(f"Unexpected type {type}")


def write_all_fortran(ds, dir):
    """
    Write all files to directory dir

    Parameters
    ----------
    ds : xr.Dataset
        The xemc3 dataset
    dir : str
        The directory to write the files to
    """
    write_locations(ds, f"{dir}/GRID_3D_DATA")
    for fn, opts in files.items():
        try:

            get_vars_for_file(ds, fn)
        except KeyError:
            pass
        else:
            write_fort_file(ds, dir, fn, **opts)

    # if False:
    #     ds["phi"] = read_mapped(path + "/POTENTIAL", map, skip_first=0, kinetic=False)[
    #         0
    #     ]
    #     ds["phi"].attrs["units"] = "V"
    #     ds["phi"].attrs["long_name"] = "Potential"

    #     ds["current"] = read_mapped(
    #         path + "/CURRENT_V", map, skip_first=0, kinetic=False
    #     )[0]
    #     ds["current"].attrs["units"] = "A"
    #     ds["current"].attrs["long_name"] = "parallel current"

    #     ds["res"] = (
    #         read_mapped(path + "/RESISTIVITY", map, skip_first=0, kinetic=False)[0]
    #         * 1e-4
    #     )
    #     ds["res"].attrs["units"] = "$\Omega$ m$^2$"
    #     ds["res"].attrs["long_name"] = "$\int \sigma_{||}^{-1}dl$"

    # # SMalpha                        fort.46                        0 6.2415093237819604e+22 $S_{M,transp.}$ [kg m$^{-2}$ s$^{-2}$]
    #     ds["SMalpha"]=read_mapped(path + "/fort.46", map, skip_first=0, kinetic=False)[0]*6.2415093237819604e+22
    #     ds["SMalpha"].attrs["units"] = "$S_{M,transp.}$ [kg m$^{-2}$ s$^{-2}$]"
    #     ds["SMalpha"].attrs["long_name"] = "$S_{M,transp.}$ [kg m$^{-2}$ s$^{-2}$]"

    # # SM                             fort.47                        0 6.2415093237819604e+22 $S_{M,CX}$ [kg m$^{-2}$ s$^{-2}$]
    #     ds["SM"]=read_mapped(path + "/fort.47", map, skip_first=0, kinetic=False)[0]*6.2415093237819604e+22
    #     ds["SM"].attrs["units"] = "$S_{M,CX}$ [kg m$^{-2}$ s$^{-2}$]"
    #     ds["SM"].attrs["long_name"] = "$S_{M,CX}$ [kg m$^{-2}$ s$^{-2}$]"

    # # imprad                         IMP_RADIATION                  0 -1000000.0 $P_{rad,imp}$ [W m$^{-3}$]
    #     ds["imprad"]=read_mapped(path + "/IMP_RADIATION", map, skip_first=0, kinetic=False)[0]*-1000000.0
    #     ds["imprad"].attrs["units"] = "$P_{rad,imp}$ [W m$^{-3}$]"
    #     ds["imprad"].attrs["long_name"] = "$P_{rad,imp}$ [W m$^{-3}$]"

    # # Simp                           IMPURITY_IONIZATION_SOURCE     0 1.0 S$_Z$ []
    #     ds["Simp"]=read_mapped(path + "/IMPURITY_IONIZATION_SOURCE", map, skip_first=0, kinetic=False)[0]
    #     ds["Simp"].attrs["units"] = "S$_Z$ []"
    #     ds["Simp"].attrs["long_name"] = "S$_Z$ []"

    # # flux_conservation              FLUX_CONSERVATION              0 1.0 flux []
    #     ds["flux_conservation"]=read_mapped(path + "/FLUX_CONSERVATION", map, skip_first=0, kinetic=False)[0]
    #     ds["flux_conservation"].attrs["units"] = "flux []"
    #     ds["flux_conservation"].attrs["long_name"] = "flux []"

    # # R                              RECOMBINATION                  0 1.6021765699999998e-13 R [s$^{-1}$ m$^{-3}$]
    #     ds["R"]=read_mapped(path + "/RECOMBINATION", map, skip_first=0, kinetic=False)[0]*1.6021765699999998e-13
    #     ds["R"].attrs["units"] = "R [s$^{-1}$ m$^{-3}$]"
    #     ds["R"].attrs["long_name"] = "R [s$^{-1}$ m$^{-3}$]"

    return ds
