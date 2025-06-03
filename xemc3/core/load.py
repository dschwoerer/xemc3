import datetime
import os
import re
import typing
import uuid

import numpy as np
import xarray as xr

from .utils import from_interval, open, prod, rrange, timeit, to_interval, raise_issue
from .depo import read_depo_raw, write_depo_raw
from .config import files

try:
    from numba import jit  # type: ignore
except ImportError:

    def jit(x):
        return x


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
    data on the line then it should read, all data is returned. If
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
        if line == "":
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


def _block_write(
    f: typing.TextIO, d: np.ndarray, fmt: str, bs: int = 10, kinetic_fix: bool = False
) -> None:
    d = d.flatten()
    asblock = (len(d) // bs) * bs
    if kinetic_fix and fmt.startswith(" ") or fmt.endswith(" "):
        fmt1 = fmt * 6
        fmt2 = fmt * len(d[asblock:])
    else:
        fmt2 = fmt1 = fmt
    np.savetxt(f, d[:asblock].reshape(-1, bs), fmt=fmt1)
    if asblock != len(d):
        print(d[asblock:], fmt2)
        np.savetxt(f, d[asblock:].reshape(1, -1), fmt=fmt2)


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


def read_magnetic_field(fn: str, ds: xr.Dataset) -> xr.DataArray | xr.Dataset:
    """
    Read magnetic field strength from grid

    Parameters
    ----------
    fn : str
        The filename to read from
    ds : xr.Dataset
        A xemc3 dataset required for the dimensions

    Returns
    -------
    xr.Dataset
        The magnetic field strength
    """

    def readblock(f, shape):
        shape = [i + 1 for i in shape]
        nx, ny, nz = shape
        raw = _fromfile(f, dtype=float, count=nx * ny * nz, sep=" ")
        raw = raw.reshape(shape[::-1])
        raw = np.swapaxes(raw, 0, 2)
        return raw

    if "_r_dims" in ds:
        if "R_bounds" in ds:
            dims = ds.R_bounds.dims[:4]
        else:
            dims = ds._plasma_map.dims
        zone, *dims = dims
        shapes = np.array([ds[f"_{k}_dims"] for k in dims]).T

        with open(fn) as f:
            raws = [readblock(f, s) for s in shapes]
            _assert_eof(f, fn)
        dss = [to_interval(dims, raw) for raw in raws]
        return merge_blocks(dss, zone)
    else:
        if "R_bounds" in ds:
            shape = ds.R_bounds.shape
            assert (
                len(shape) == 6
            ), f"R_bounds have length {len(ds.R_bounds.shape)} with dims {ds.R_bounds.dims}={ds.R_bounds.shape} but expected 6"
            shape = shape[:3]
            dims = ds.R_bounds.dims[:3]
        else:
            shape = ds._plasma_map.shape
            dims = ds._plasma_map.dims

        with open(fn) as f:
            raw = readblock(f, shape)
            _assert_eof(f, fn)
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


def read_locations_raw(fn: str) -> typing.List[typing.Any]:
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
    blocks: typing.List[typing.Any] = [[] for _ in range(3)]
    with open(fn) as f:

        def read(f, nx, ny):
            t = _fromfile(f, dtype=float, count=nx * ny, sep=" ")
            t = t.reshape(ny, nx)
            t = t.transpose()
            return t

        while True:
            nxyz = _fromfile(f, dtype=int, count=3, sep=" ")
            if len(nxyz) == 0:
                break
            nx, ny, nz = nxyz
            phidata = np.empty(nz)
            rdata = np.empty((nx, ny, nz))
            zdata = np.empty((nx, ny, nz))

            # Read and directly convert to SI units
            for i in range(nz):
                phidata[i] = float(f.readline()) * np.pi / 180.0
                rdata[:, :, i] = read(f, nx, ny) / 100
                zdata[:, :, i] = read(f, nx, ny) / 100

            for i, d in enumerate((phidata, rdata, zdata)):
                blocks[i].append(d)

    return blocks


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
    haszone = "zone" in ds.dims
    dims: tuple[str, str, str, str] | tuple[str, str, str]
    if haszone:
        shape = [x.shape[0] for x in [ds.zone, ds.r, ds.theta, ds.phi]]
        dims = ("zone", "r", "theta", "phi")
    else:
        shape = [x.shape[0] for x in [ds.r, ds.theta, ds.phi]]
        dims = ("r", "theta", "phi")
    ret = np.zeros(shape, dtype=bool)
    with open(fn) as f:
        last = None
        for i, line in enumerate(f):
            lines = [int(x) for x in line.split()]
            if last:
                lines = last + lines
            zone, r, theta, num = lines[:4]
            if zone != 0 and not haszone:
                raise ValueError("Multiple zones not expected." + raise_issue)
            assert num % 2 == 0, f"Unexpected input in {fn}:{i} `line`" + raise_issue
            if num + 4 > len(lines):
                last = lines
                continue
            assert num + 4 == len(lines), (
                f"failed to parse line {i+1}{' (continued from previous incomplete line)' if last else ''}"
                f" from {fn}: {' '.join([str(x) for x in lines])}" + raise_issue
            )
            last = None
            for t in range(num // 2):
                a, b = lines[4 + 2 * t : 6 + 2 * t]
                ind: tuple[int, int, int, slice] | tuple[int, int, slice]
                ind = r, theta, slice(a, b + 1)
                if haszone:
                    ind = zone, *ind
                ret[ind] = True
    return xr.DataArray(data=ret, dims=dims)


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
                    ab = [0, len(slice1d) - 1]
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


def read_mappings(fn: str, dims: typing.Sequence[int]) -> xr.DataArray | xr.Dataset:
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
    haszones = hasattr(dims[0], "__len__")
    if haszones:
        toread = sum([prod(d) for d in zip(*dims)]).values
    else:
        toread = prod(dims)
    with open(fn) as f:
        dat = f.readline()
        infos = [int(i) for i in dat.split()]
        t = _fromfile(f, dtype=int, count=toread, sep=" ")
        # fortran indexing
        t -= 1
        if haszones:
            ts = []
            i = 0
            for d in zip(*dims):
                d = [int(x) for x in d]
                inext = i + prod(d)
                ti = t[i:inext]
                ti = ti.reshape(d, order="F")
                ts.append(ti)
                i = inext
        else:
            t = t.reshape(dims, order="F")
        _assert_eof(f, fn)
    if haszones:
        das = [xr.DataArray(dims=("r", "theta", "phi"), data=t) for t in ts]
        da = merge_blocks(das, "zone")
    else:
        da = xr.DataArray(dims=("r", "theta", "phi"), data=t)
    da.attrs = dict(numcells=infos[0], plasmacells=infos[1], other=infos[2])
    return da


def ensure_metadata(ds: xr.Dataset) -> xr.Dataset:
    if not ds:
        return ds
    meta = get_default_metadata()
    for k in meta:
        if k not in ds.attrs:
            ds.attrs.update(meta)
            return ds
    return ds


def add_metadata(ds: xr.Dataset) -> xr.Dataset:
    ds.attrs.update(get_default_metadata())
    return ds


def get_default_metadata() -> dict:
    # Delay import to avoid circular dependency
    from .. import __version__

    return dict(
        title="EMC3-EIRENE Simulation data",
        software_name="xemc3",
        software_version=__version__,
        date_created=datetime.datetime.utcnow().isoformat(),
        id=str(uuid.uuid1()),
        references="https://doi.org/10.5281/zenodo.5562265",
    )


def write_mappings(da: xr.DataArray, fn: str) -> None:
    """Write the mappings data to fortran"""
    with open(fn, "w") as f:
        at_keys = "numcells", "plasmacells", "other"
        for key in at_keys:
            assert key in da.attrs, (
                "Writing requires to be the following attributes to be present:"
                + ", ".join([f'"{k}"' for k in at_keys])
                + ". Please ensure that they are copied over from the read file."
            )
        infos = tuple([da.attrs[k] for k in at_keys])
        f.write("%12d %11d %11d\n" % infos)
        _block_write(f, da.data.flatten(order="F") + 1, " %11d", 6)


def read_raw(fn: str) -> xr.DataArray:
    with open(fn) as f:
        return xr.DataArray(f.read())


def write_raw(da: xr.DataArray, fn: str) -> None:
    with open(fn, "w") as f:
        f.write(str(da.data))


def read_locations(path: str, ds: typing.Optional[xr.Dataset] = None) -> xr.Dataset:
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
    phis, rs, zs = read_locations_raw(get_file_name(path, "geom"))
    dss = [
        xr.Dataset().assign_coords(
            {
                "R_bounds": to_interval(("r", "theta", "phi"), r),
                "z_bounds": to_interval(("r", "theta", "phi"), z),
                "phi_bounds": to_interval(("phi",), phi),
            }
        )
        for r, z, phi in zip(rs, zs, phis)
    ]
    if len(dss) > 1:
        ds_ = merge_blocks(dss, "zone")
    else:
        ds_ = dss[0]
    for v in ds_:
        ds[v] = ds_[v]
    ds = ds.assign_coords(coords=ds_.coords)

    for x in ds.coords:
        ds[x].attrs["xemc3_type"] = "geom"
    ds.emc3.unit("R_bounds", "m")
    ds.emc3.unit("z_bounds", "m")
    ds.emc3.unit("phi_bounds", "radian")
    assert isinstance(ds, xr.Dataset)
    return add_metadata(ds)


def scrape(f: typing.TextIO, *, ignore="!", verbose=False) -> str:
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
    assert len(test) == 0, f"Expected EOF, but found more data in {fn}" + raise_issue


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
        assert len(setup) == 5, f"Expected 5 values but got {setup}" + raise_issue
        for zero in setup[3:]:
            assert float(zero) == 0.0, (
                "A shifted divertor is currently not supported in xemc3." + raise_issue
            )
        nx, ny, _ = [int(i) for i in setup[:3]]
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


def write_plate(data: typing.Tuple[np.ndarray, ...], filename: str) -> None:
    shape = data[0].shape
    assert shape == data[1].shape
    assert shape[:1] == data[2].shape
    try:
        data[0].attrs  # type: ignore
    except AttributeError:
        pass
    else:
        data = [d.values for d in data]  # type: ignore
    data = [data[0] * 100, data[1] * 100, data[2] * 180 / np.pi]
    with open(filename, "w") as f:
        f.write("# Written by xemc3\n")
        f.write(
            f"           {shape[0]}           {shape[1]}           5  0.0000000E+00  0.0000000E+00\n"
        )
        for Rs, Zs, phi in zip(*data):
            f.write(f"  {phi}\n")
            for R, Z in zip(Rs, Zs):
                f.write(f"      {R}  {Z}\n")


def read_plate_nice(filename: typing.Union[str, typing.Sequence[str]]) -> xr.Dataset:
    """
    Read Target structures from a file that is in the Kisslinger
    format as used by EMC3.

    Parameters
    ----------
    filename : str or sequence of str
        The location of the file to read

    Returns
    -------
    xr.Dataset
        The coordinates
    """
    if isinstance(filename, str):
        return read_plate_ds(filename)
    dss = [read_plate_ds(fn) for fn in filename]
    ds = merge_blocks(dss)
    assert isinstance(ds, xr.Dataset)
    return ds


def read_add_sf_n0(filename: str) -> xr.Dataset:
    """
    Read the ADD_SF_N0 file and return the referenced geometry.

    Parameters
    ----------
    filename : str
        The path of the file

    Returns
    -------
    xr.Dataset
        The coordinates
    """
    if "/" in filename:
        dir = "/".join(filename.split("/")[:-1]) + "/"
    else:
        dir = ""
    files = []
    with open(filename) as f:
        num = int(scrape(f))
        for _ in range(num):
            line = scrape(f)
            lines = line.split()
            assert len(lines) == 3, (
                f"Unexpected content in {filename}:{line}." + raise_issue
            )
            if int(lines[0]) != 0:
                raise ValueError(
                    f"Only Kisslinger files are currently supported, not triangulated meshes - while reading {filename}:{line}"
                )
            # assert [int(x) for x in lines] == [
            #     0,
            #     -4,
            #     1,
            # ], f"Unexpected content in {filename}: {line}"
            files.append(dir + scrape(f).strip())
    return read_plate_nice(files)


plate_prefix = "plate_"


def read_plate_ds(filename: str) -> xr.Dataset:
    """
    Read Target structures from a file that is in the Kisslinger
    format as used by EMC3. It returns the coordinates as dataset
    containing R, z and phi.

    Parameters
    ----------
    filename : str
        The location of the file to read

    Returns
    -------
    xr.Dataset
        The coordinates
    """
    rzp = read_plate(filename)
    out = xr.Dataset()
    for name, dat in zip(("R", "z", "phi"), rzp):
        if len(dat.shape) == 2:
            dims = ["phi", "x"]
        else:
            assert len(dat.shape) == 1, (
                f"Unexpected number of dimensions {dat.shape}." + raise_issue
            )
            dims = ["phi"]
        out = out.assign_coords(
            {f"{plate_prefix}{name}": ([f"{plate_prefix}{d}_plus1" for d in dims], dat)}
        )
    for var, attrs in files["ADD_SF_N0"]["vars"].items():
        out[var].attrs.update(attrs)

    return out


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
        s = scrape(f).split()
        _, num_plates = [int(i) for i in s]
        plates = []
        for plate in range(num_plates):
            s = scrape(f).split()
            assert len(s) == 2, (
                f"Unexpected string `{s}` while reading {plate}." + raise_issue
            )
            _, geom = s
            if not geom.startswith("/"):
                geom = cwd + geom
            r, z, phi = read_plate(geom)
            nx, ny = r.shape
            nx -= 1
            ny -= 1
            s = scrape(f).split()
            total = np.array([float(i) for i in s])
            s = scrape(f).split()
            if len(s) == 3:
                items, yref, xref = [int(i) for i in s]
                nxr = nx * xref
                nyr = ny * yref
                mode = 2
            else:
                assert len(s) == 2, (
                    f"Unexpected string `{s}` while reading {cwd + fn}" + raise_issue
                )
                nyr, nxr = [int(i) for i in s]
                xref = nxr // nx
                yref = nyr // ny
                items = nx * ny
                mode = 1

            assert items == nx * ny
            corrs: typing.List[np.ndarray[tuple[int, ...], typing.Any]]
            coordinates: typing.List[np.ndarray[tuple[int, ...], typing.Any]]
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
                dims = [plate_prefix + x for x in ("phi", "x")]
                dims += ["delta_" + x for x in dims]
                corrs_da = [
                    xr.DataArray(
                        data=a.reshape(nxr, nyr, 2, 2),
                        dims=dims,
                    )
                    for a in corrs
                ]
            else:
                assert mode == 1
                corrs_da = [
                    to_interval((plate_prefix + "phi", plate_prefix + "x"), a)
                    for a in coordinates
                ]

            coords = {
                plate_prefix + "R_bounds": corrs_da[0],
                plate_prefix + "z_bounds": corrs_da[1],
                plate_prefix + "phi_bounds": corrs_da[2],
            }
            ds = xr.Dataset(coords=coords)  # type: ignore
            ds.coords[plate_prefix + "R_bounds"].attrs["units"] = "m"
            ds.coords[plate_prefix + "z_bounds"].attrs["units"] = "m"
            ds.coords[plate_prefix + "phi_bounds"].attrs["units"] = "radian"

            vars = files[get_file_name(None, "target_flux")]["vars"].copy()
            for i, (l, meta) in enumerate(vars.items()):
                ds[l] = (plate_prefix + "phi", plate_prefix + "x"), data[i] * meta.get(
                    "scale", 1
                )
                for k in meta:
                    ds[l].attrs[k] = meta[k]
            for i, l in enumerate(["tot_n", "tot_P"]):
                ds[l] = total[i]
            plates.append(ds)

        # Make sure we have read everything
        _assert_eof(f, cwd + fn)
    return plates


def merge_blocks(
    dss: typing.Sequence[xr.Dataset | xr.DataArray], axes=plate_prefix + "ind"
) -> xr.Dataset | xr.DataArray:
    """
    Convert a list of datasets to one dataset

    Parameters
    ----------
    dss : sequence of xr.Datasets
        The sequence of datasets that should be merged. Unlike
        xr.combine_nested the input's do not need to have the same
        shape, as the data is nan-padded.

    axis : str
        The name of the new axis

    Returns
    -------
    xr.Dataset
        The merged dataset with the new axes
    """
    dims = {d: 0 for d in dss[0].sizes}
    for plate in dss:
        for k, v in plate.sizes.items():
            if dims[k] < v:
                dims[k] = v
    ds = xr.Dataset()

    matching = {}
    for k, v in dims.items():
        matching[k] = True
        org_dims = []
        for plate in dss:
            org_dims.append(plate.sizes[k])
            if plate.sizes[k] != v:
                matching[k] = False
        assert isinstance(k, str)
        ds[f"_{k}_dims"] = (axes, org_dims)

    dims[axes] = len(dss)

    def merge(var):
        shape = [dims[axes]] + [dims[d] for d in dss[0][var].dims]
        data = np.empty(shape)
        data[...] = np.nan
        for i, plate in enumerate(dss):
            tmp = plate[var]
            data[tuple([i] + [slice(None, i) for i in tmp.shape])] = tmp
        return (axes, *dss[0][var].dims), data

    if isinstance(dss[0], xr.DataArray):
        shape = [dims[axes]] + [dims[d] for d in dss[0].dims]
        data = np.empty(shape, dtype=dss[0].dtype)
        data[...] = -1 if data.dtype == int else np.nan
        for i, plate in enumerate(dss):
            tmp = plate
            data[tuple([i] + [slice(None, i) for i in tmp.shape])] = tmp
        coords = {c: merge(c) for c in dss[0].coords.keys()}
        return xr.DataArray(data=data, coords=coords, dims=(axes, *dss[0].dims))

    for coord in dss[0].coords.keys():
        ds = ds.assign_coords({coord: merge(coord)})

    for var in dss[0]:
        ds[var] = merge(var)
        ds[var].attrs = dss[0][var].attrs

    return ds


def load_plates(dir: str, fn: typing.Optional[str] = None) -> xr.Dataset:
    """
    Read the target heatflux mapping from EMC3 Postprocessing routine.

    Parameters
    ----------
    dir : str
        The location of the directory in which the files are to be read

    fn : str
        The name of the deposition file

    Returns
    -------
    xr.Dataset
        The read data
    """
    if fn is None:
        fn = get_file_name(None, "target_flux")
    if dir[-1] != "/":
        dir += "/"

    plates = read_plates_raw(dir, fn)
    mplates = merge_blocks(plates)
    assert isinstance(mplates, xr.Dataset)
    return mplates


def write_plates(dir: str, plates: xr.Dataset) -> None:
    # Deprecate?
    with timeit("Writing ncs: %f"):
        # Note we really should compress to get rid of the NaN's we added
        plates.to_netcdf(
            f"{dir}/TARGET_PROFILES.nc",
            encoding={
                i: {"zlib": True, "complevel": 1}
                for i in [i for i in plates] + [i for i in plates.coords]
            },
        )


def read_plates(dir: str) -> xr.Dataset:
    # Deprecate?
    ds = xr.open_dataset(f"{dir}/TARGET_PROFILES.nc")
    assert isinstance(ds, xr.Dataset)
    return ds


def _dir_of(fn: str) -> str:
    if os.path.isdir(fn):
        return fn
    if "/" in fn:
        return fn.rsplit("/", 1)[0]
    return "."


def get_plates(dir: str, cache: bool = True) -> xr.Dataset:
    """
    Read the target fluxes from the EMC3_post processing routine

    Parameters
    ----------
    dir : str
        The directory in which the file is
    cache : bool (optional)
        Whether to check whether a netcdf file is present, and if not
        create one. This can result in faster reading the next time.

    Returns
    -------
    xr.Dataset
        The target profiles
    """
    dir = _dir_of(dir)

    if cache:
        try:
            if os.path.getmtime(dir + "/TARGET_PROFILES.nc") > os.path.getmtime(
                get_file_name(dir, "target_flux")
            ):
                return read_plates(dir)
        except OSError:
            pass
        data = load_plates(dir)
        write_plates(dir, data)
        return data
    return load_plates(dir)


def ensure_mapping(
    dir: str,
    mapping: typing.Union[None, xr.Dataset, xr.DataArray] = None,
    need_mapping: bool = True,
    fn: typing.Union[None, str] = None,
) -> xr.Dataset:
    """
    Ensure that basic info's are present to read datafiles.

    Parameters
    ----------
    dir : str
        The folder in which to look for files
    mapping : None or xr.Dataset or xr.DataArray
        Either the required mapping, in which case nothing is done, or
        None, in which case the data is read.
    need_mapping : bool (optional)
        If true the mapping information has to be loaded, otherwise the
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
    error = f"""Reading {fn+ ' ' if fn is not None else ''}mapped requires mapping information, but the required
information in '{dir}' could not be found.  Ensure all files are present
in the folder or pass in a dataset that contains the mapping
information. Failed to open `%s`."""
    if dir == "":
        dir = "."
    if mapping is None:
        try:
            mapping = read_locations(dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(33, error % e.filename)
    if isinstance(mapping, xr.DataArray):
        if mapping.name == "_plasma_map":
            return xr.Dataset(dict(_plasma_map=mapping))
    assert isinstance(mapping, xr.Dataset)
    if need_mapping:
        try:
            mapping = read_fort_file(mapping, f"{dir}/fort.70", **files["fort.70"])
        except FileNotFoundError as e:
            raise FileNotFoundError(33, error % e.filename)
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
        A dataset in which at least ``var`` is set. If there are other
        variables in the read file, it is also added. Finally, mapping
        and other variables that are required to read are also added.
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
                ds = ensure_mapping(
                    dir, ds, fd.get("type", "mapped") == "mapped", fn=fn
                )
                assert isinstance(ds, xr.Dataset)
                return read_fort_file(ds, f"{dir}/{fn}", **fd)
    raise ValueError(f"Don't know how to read {var}")


def read_mapped(
    fn: str,
    mapping: typing.Union[xr.Dataset, xr.DataArray],
    skip_first: int = 0,
    ignore_broken: bool = False,
    kinetic: bool = False,
    unmapped: bool = False,
    dtype: DTypeLike = float,
    squeeze: bool = True,
) -> typing.Sequence[xr.DataArray]:
    """
    Read a file with the EMC3 mapping. Note that this function does
    not add meta-data or does normalisation of the data, thus
    `xemc3.load.file` is generally preferred.

    Parameters
    ----------
    fn : str
        The full path of the file to read
    mapping : xr.DataArray or xr.Dataset
        The dataarray of the mesh mapping or a dataset containing the
        mapping
    skip_first : int (optional)
        Ignore the first n lines. Default 0
    ignore_broken: bool (optional)
        if incomplete datasets at the end should be ignored. Default: False
    kinetic : bool (optional)
        The file contains also data for cells that are only evolved by
        EIRENE, rather then EMC3. Default: False
    unmapped : bool (optional)
        The file contains unmapped data, i.e. on value for each cell.
        Default: False
    dtype : datatype (optional)
        The type of the data, e.g. float or int. Is passed to
        numpy. Default: float
    squeeze : bool
        If True return a DataArray if only one field is read.

    Returns
    -------
    xr.DataArray or list of xr.DataArray
        The data that has been read from the file. If squeeze is True
        and only one field is read only a single DataArray is
        returned.

    """
    if isinstance(mapping, xr.Dataset):
        mapping = ensure_mapping(_dir_of(fn), mapping, fn=fn)
        mapping = mapping["_plasma_map"]
    if kinetic:
        assert unmapped is False
        max = np.max(mapping.data) + 1
    elif unmapped:
        max = mapping.attrs["numcells"]
    else:
        max = mapping.attrs["plasmacells"]
    firsts = []
    with open(fn) as f:
        raws = []
        while True:
            if skip_first:
                first = ""
                if isinstance(skip_first, int):
                    sf = skip_first
                else:
                    sf = skip_first[min(len(raws), len(skip_first) - 1)]
                for _ in range(sf):
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

    def to_da_mapped(raw):
        out = np.ones(mapping.shape) * np.nan
        mapdat = mapping.data
        for ijk in rrange(mapping.shape):
            mapid = mapdat[ijk]
            if mapid < max:
                out[ijk] = raw[mapid]
        return xr.DataArray(data=out, dims=mapping.dims)

    def to_da_unmapped(raw):
        return xr.DataArray(
            data=raw.reshape(mapping.shape[::-1]).transpose(2, 1, 0), dims=mapping.dims
        )

    to_da = to_da_unmapped if unmapped else to_da_mapped
    das = [to_da(raw) for raw in raws]
    if skip_first:
        for first, da in zip(firsts, das):
            da.attrs["print_before"] = first
    if squeeze and len(das) == 1:
        das = das[0]
    return das


def write_mapped_nice(
    ds: xr.Dataset, dir: str, fn: typing.Optional[str] = None, **args
) -> None:
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
                datas[i].attrs["scaled_by"] = ops["scale"]
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
    list of tuple of str and dict
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


@jit
def to_mapped_core_3d(
    datdat: np.ndarray, mapdat: np.ndarray, out: np.ndarray, count: np.ndarray, max: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    if len(datdat.shape) == 3:
        for i in range(mapdat.shape[0]):
            for j in range(mapdat.shape[1]):
                for k in range(mapdat.shape[2]):
                    mapid = mapdat[i, j, k]
                    if mapid < max:
                        cdat = datdat[(..., i, j, k)]
                        if not (np.isnan((cdat))):
                            out[..., mapid] += cdat
                            count[mapid] += 1
    else:
        for i in range(mapdat.shape[0]):
            for j in range(mapdat.shape[1]):
                for k in range(mapdat.shape[2]):
                    mapid = mapdat[i, j, k]
                    if mapid < max:
                        cdat = datdat[(..., i, j, k)]
                        if not (np.isnan((cdat))):
                            out[..., mapid] += cdat
                            count[mapid] += 1
    return out, count


@jit
def to_mapped_core_4d(
    datdat: np.ndarray, mapdat: np.ndarray, out: np.ndarray, count: np.ndarray, max: int
) -> typing.Tuple[np.ndarray, np.ndarray]:
    if len(datdat.shape) == 4:
        for i in range(mapdat.shape[0]):
            for j in range(mapdat.shape[1]):
                for k in range(mapdat.shape[2]):
                    for l in range(mapdat.shape[3]):
                        mapid = mapdat[i, j, k, l]
                        if mapid < max:
                            cdat = datdat[(..., i, j, k, l)]
                            if not (np.isnan((cdat))):
                                out[..., mapid] += cdat
                                count[mapid] += 1
    else:
        for i in range(mapdat.shape[0]):
            for j in range(mapdat.shape[1]):
                for k in range(mapdat.shape[2]):
                    for l in range(mapdat.shape[3]):
                        mapid = mapdat[i, j, k, l]
                        if mapid < max:
                            cdat = datdat[(..., i, j, k, l)]
                            if not (np.isnan((cdat))):
                                out[..., mapid] += cdat
                                count[mapid] += 1
    return out, count


def to_mapped(
    data: xr.DataArray,
    mapping: xr.DataArray,
    kinetic: bool = False,
    dtype: typing.Union[DTypeLike, None] = None,
) -> np.ndarray:
    if kinetic:
        max = np.max(mapping.values) + 1
    else:
        max = np.max(mapping.attrs["plasmacells"])
    if dtype is None:
        dtype = data.values.dtype

    out = np.zeros((*data.shape[:-3], max), dtype=dtype)
    mapdat = mapping.values
    assert mapdat.dtype == int
    datdat = data.values
    if out.dtype != datdat.dtype:
        if out.dtype == np.int64 and datdat.dtype == np.float64:
            datdat = (datdat + 0.5).astype(dtype)
    count = np.zeros(max, dtype=int)
    args = datdat, mapdat, out, count
    for arg in args:
        assert isinstance(
            arg, np.ndarray
        ), f"Expected to write np.ndarray, but got {type(arg)}."
    to_mapped_core = to_mapped_core_4d if "zone" in mapping.dims else to_mapped_core_3d
    out, count = to_mapped_core(*args, max)
    if out.dtype in [np.dtype(x) for x in [int, np.int32, np.int64]]:
        out //= count
    else:
        out /= count
    assert isinstance(out, np.ndarray)
    return out


def write_mapped(
    datas,
    mapping,
    fn,
    skip_first=0,
    ignore_broken=False,
    kinetic=False,
    unmapped=False,
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
    kinetic : bool
        If true the data is defined also outside of the plasma region.
    unmapped : bool
        If true the data is not mapped
    dtype : any
        if not None, it needs to match the dtype of the data
    fmt : None or str
        The Format to be used for printing the data.
    """
    if skip_first is not None:
        for d in datas:
            if skip_first:
                assert "print_before" in d.attrs, (
                    'The "print_before" attribute is needed for writing. '
                    "Please ensure it is preserved or copied from the read data."
                )
                assert d.attrs["print_before"] != ""
            else:
                if "print_before" in d.attrs:
                    assert d.attrs["print_before"] == ""
    if not isinstance(datas, (list, tuple)):
        datas = [datas]
    if unmapped:
        assert kinetic is False
        out = [np.ravel(x, order="F") for x in datas]
    else:
        out = [to_mapped(x, mapping, kinetic, dtype) for x in datas]
    with open(fn, "w") as f:
        for i, da in zip(out, datas):
            if "print_before" in da.attrs:
                f.write(da.attrs["print_before"])
            if fmt is None:
                tfmt = "%.4e" if dtype != int else "%d"
                if kinetic:
                    assert dtype != int
                    tfmt = " %11.4E"
            else:
                tfmt = fmt
            _block_write(f, i, fmt=tfmt, bs=6, kinetic_fix=kinetic)


def read_info_file(
    fn: str,
    vars: dict,
    index: str = "iteration",
    length: int = 1000,
) -> typing.List[xr.DataArray]:
    block = len(vars)
    assert block > 0
    ret: typing.List[np.ndarray] = []
    with open(fn) as f:
        while True:
            dat = _fromfile(f, dtype=float, count=block, sep=" ")
            if len(dat) == 0:
                dat = np.array(ret)[-length:]
                if len(dat) < length:
                    padded = np.empty((length, block))
                    padded[: -len(dat)] = np.nan
                    padded[-len(dat) :] = dat
                    dat = padded
                coords: typing.Mapping[typing.Hashable, typing.Any] = {
                    index: xr.DataArray(np.arange(-length + 1, 1), dims=index)
                }
                return [xr.DataArray(d, dims=index, coords=coords) for d in dat.T]
            if not len(dat) == block:
                raise RuntimeError(
                    f"Error while reading `{fn}` - expected {block} items, but got {len(dat)}"
                )
            ret.append(dat)


def write_info_file(fn: str, ds: xr.Dataset) -> None:
    info = files[fn.split("/")[-1]]
    fmt = info["fmt"]
    dats: typing.List[np.ndarray] = []
    for v, i in info["vars"].items():
        dats.append(ds[v].data)
        if "scale" in i:
            # Make copy, so we don't change underlying data
            dats[-1] = dats[-1] / i["scale"]
    dat = np.array(dats)
    fmtc = fmt.count("%")
    assert fmtc == len(
        dat
    ), f"Found {fmtc} format specifiers but data has {len(dat)} values. Format is {fmt}."
    # dat.shape == x, 1000
    valid_entries = np.sum(np.isfinite(dat), axis=1)
    assert len(valid_entries) == fmtc
    with open(fn, "w") as f:
        for d in dat.T[-np.max(valid_entries) :]:
            f.write(fmt % tuple(d) + "\n")


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
    if type == "info":
        if not isinstance(ds, xr.Dataset):
            ds = xr.Dataset()
        return read_fort_file(ds, fn, **defaults)
    if type in ["target_flux", "mapping"]:
        ds = ds or xr.Dataset()
    else:
        ds = ensure_mapping("/".join(fn.split("/")[:-1]), ds, type == "mapped", fn=fn)
    assert isinstance(ds, xr.Dataset)
    return read_fort_file(ds, fn, **defaults)


def read_fort_file(ds: xr.Dataset, fn: str, type: str = "mapped", **opts) -> xr.Dataset:
    """
    Read an EMC3 simulation file and add to a dataset.

    """
    datas = None
    vars = opts.pop("vars")
    _ = opts.pop("fmt", None)
    if type == "mapping":
        dims: typing.Any = ("r", "theta", "phi")
        if "_r_dims" in ds:
            dims = tuple([ds[f"_{k}_dims"] for k in dims])
        else:
            dims = tuple([len(ds[k]) for k in dims])
        ds["_plasma_map"] = read_mappings(fn, dims)
    elif type == "mapped":
        # Ensure file is present before we try to read mapping
        # This is because missing mapping is handled differently.
        with open(fn):
            pass
        ds = ensure_mapping(_dir_of(fn), ds, fn=fn)
        datas = read_mapped(fn, ds["_plasma_map"], **opts, squeeze=False)
        opts = {}
    elif type == "full":
        tmp = read_magnetic_field(fn, ds)
        assert isinstance(tmp, xr.DataArray)
        datas = [tmp]
    elif type == "plates_mag":
        datas = [read_plates_mag(fn, ds)]
    elif type == "geom":
        ds_ = read_locations(_dir_of(fn))
        ds = ds.assign_coords(ds_.coords)
    elif type == "info":
        opts.pop("ignore_broken", None)
        if "iteration" in ds.dims and "length" not in opts:
            opts["length"] = len(ds["iteration"])
        datas = read_info_file(fn, vars, **opts)
        opts = {}
    elif type == "surfaces":
        ds_ = read_add_sf_n0(fn)
        ds = ds.assign_coords(ds_.coords)
        for k in ds_:
            ds[k] = ds_[k]
        datas = None
    elif type == "target_flux":
        ds_ = get_plates(fn, False)
        ds = ds.assign_coords(ds_.coords)
        for k in ds_:
            ds[k] = ds_[k]
        datas = None
    elif type == "raw":
        datas = [read_raw(fn)]
    elif type == "depo":
        datas = read_depo_raw(ds, fn)
    else:
        raise RuntimeError(f"Unexpected type {type}")
    opts.pop("ignore_broken", None)
    assert opts == {}, "Unexpected arguments: " + ", ".join(
        [f"{k}={v}" for k, v in opts.items()]
    )
    if datas is None:
        return ensure_metadata(ds)
    vars = vars.copy()
    assert opts == {}
    keys = [k for k in vars]
    if "%" in keys[-1]:
        key = keys[-1]
        flexi = vars.pop(key)
        for i in range(len(vars), len(datas)):
            vars[key % i] = flexi
    assert len(vars) == len(
        datas
    ), f"in file {fn} we found {len(datas)} fields but only {len(vars)} vars are given!"
    for (var, varopts), data in zip(vars.items(), datas):
        ds[var] = data
        varopts = varopts.copy()
        scale = varopts.pop("scale", 1)
        if scale != 1:
            ds[var].data *= scale

        attrs = varopts.pop("attrs", {})
        attrs["xemc3_type"] = type
        for k in "long_name", "units", "notes", "description":
            if k in varopts:
                attrs[k] = varopts.pop(k)
        k = "parallel_flux"
        attrs[k] = varopts.pop(k, 0)

        ds[var].attrs.update(attrs)
        assert (
            varopts == {}
        ), f"variable {var} has options {varopts} but didn't expect anything"
    return ensure_metadata(ds)


def guess_type(ds: xr.Dataset, key: typing.Hashable) -> str:
    data = ds[key]
    assert isinstance(key, str)
    try:
        ret = data.attrs["xemc3_type"]
        assert isinstance(ret, str)
        return ret
    except KeyError:
        pass
    if key == "_plasma_map":
        return "mapping"
    nameparts = key.split("_")
    if len(nameparts) == 2:
        if nameparts[0] in ["R", "z", "phi"] and nameparts[1] in [
            "bounds",
            "corners",
        ]:
            return "geom"
    fulldims = len(
        {"R", "phi", "z", "delta_R", "delta_z", "delta_phi"}.intersection(ds[key].dims)
    )
    if fulldims == 6:
        return "full"
    phidims = len({"phi", "delta_phi"}.intersection(ds[key].dims))
    if phidims == fulldims == 2:
        return "geom"
    if data.dtype == bool:
        return "plates_mag"
    if data.dtype == int:
        return "mapping"
    return "mapped"


def guess_kinetic(ds: xr.Dataset, key: typing.Hashable) -> bool:
    pm = ds._plasma_map
    assert ds[key].dtype in [float, np.float32, np.float16]
    return bool(np.any(np.isfinite(ds[key].data[(..., pm == pm.attrs["plasmacells"])])))


def archive(ds: xr.Dataset, fn: str, geom: bool = False, mapping: bool = True) -> None:
    arch = xr.Dataset()
    arch.attrs = ds.attrs
    for k in list(ds.coords) + list(ds):
        type = guess_type(ds, k)
        if type == "mapped":
            kinetic = guess_kinetic(ds, k)
            arch[k] = (
                *ds[k].dims[:-3],
                "kinetic_map" if kinetic else "plasma_map",
            ), to_mapped(ds[k], ds._plasma_map, guess_kinetic(ds, k))
        elif type in ("geom", "full"):
            if not geom:
                continue
            arch[k] = (
                tuple(
                    [
                        f"{x}_plus1"
                        for x in ds[k].dims
                        if x[:6] != "delta_"  # type: ignore
                    ]
                ),
                from_interval(ds[k]),
            )
        elif type == "plates_mag":
            if not geom:
                continue
            arch[k] = ds[k]
        elif type == "mapping":
            if not mapping:
                continue
            arch[k] = ds[k]
        else:
            arch[k] = ds[k]
        arch[k].attrs = ds[k].attrs
    arch.to_netcdf(
        fn,
        encoding={
            i: {"zlib": True, "complevel": 9} for i in list(arch) + list(arch.coords)
        },
    )
    print(f"done with {fn}")


def load_all(
    path: str,
    ignore_missing: typing.Optional[bool] = None,
    ignore_broken: typing.Optional[bool] = None,
) -> xr.Dataset:
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
    ignore_broken : None or bool
         True: ignore if files are incomplete.
         False: raise exceptions if a file is incomplete.
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
    ds = read_locations(path)
    for fn, opts in files.items():
        opts = opts.copy()
        try:
            ds = read_fort_file(ds, f"{path}/{fn}", ignore_broken=ignore_broken, **opts)
        except FileNotFoundError as e:
            if e.args[0] == 33:
                raise
            if ignore_missing is None:
                if not opts.get("ignore_missing", True):
                    raise
            elif not ignore_missing:
                raise
    return ds


def load_any(path: str, *args, **kwargs) -> xr.Dataset:
    """
    Read a file or directory. For possible kwargs see the respective
    functions that are called:

    For a directory xemc3.load.all is called.

    xemc3.load.plates for the "TARGET_PROFILES" file.

    xemc3.load.file otherwise.
    """
    if os.path.isdir(path):
        return load_all(path, *args, **kwargs)
    return read_fort_file_pub(path, *args, **kwargs)


def write_fort_file(ds, dir, fn, type="mapped", **opts):
    if type == "mapping":
        assert list(opts["vars"].keys()) == ["_plasma_map"]
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
        assert list(vars.keys()) == ["PLATES_MAG"]
        write_plates_mag(f"{dir}/{fn}", ds)
    elif type == "info":
        write_info_file(f"{dir}/{fn}", ds)
    elif type == "geom":
        write_locations(ds, f"{dir}/{fn}")
    elif type == "raw":
        vars = opts.pop("vars")
        assert len(vars) == 1
        write_raw(ds[list(vars.keys())[0]], f"{dir}/{fn}")
    elif type == "depo":
        vars = get_vars_for_file(ds, opts.pop("vars"))
        datas = [ds[k] for k, _ in vars]
        write_depo_raw(datas, f"{dir}/{fn}")
    elif type in ["surfaces", "target_flux"]:
        print(f"writing {fn} is not yet implemented. (PRs welcome)")
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
    write_locations(ds, get_file_name(dir, "geom"))
    for fn, opts in files.items():
        try:
            get_vars_for_file(ds, fn)
        except KeyError:
            pass
        else:
            write_fort_file(ds, dir, fn, **opts)


def get_file_name(dir: typing.Optional[str], type: str) -> str:
    found: typing.List[str] = []
    for file, data in files.items():
        if data.get("type", "mapped") == type:
            found.append(file)
    assert len(found) == 1
    if dir:
        return f"{dir}/{found[0]}"
    return found[0]
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
