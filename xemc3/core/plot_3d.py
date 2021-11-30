import itertools
import sys

import mayavi.mlab as mlab  # type: ignore
import numpy as np
import xarray as xr
from tvtk.api import tvtk  # type: ignore

from .utils import rrange


def _pprint(*a, **k):
    print(*a, **k, end="")
    sys.stdout.flush()


class divertor:
    def __init__(
        self,
        ds,
        index,
        *,
        symmetry=False,
        segments=1,
        power_cutoff=None,
        verbose=False,
        vmax=None,
        vmin=None,
        only_lower=False,
        title=None,
        phi_slices=False,
        path1=None,
        path2=None,
    ):
        """
        Plot the first wall and the mapped plasma parameters.

        Parameters
        ----------
        ds : xr.Dataset
            The emc3 dataset of the mapped parameters
        index : str
            The index to plot
        symmetry : boolean
            Plot as well a copyied version of the data assuming
            stellerator symmetry.
        segments : int
            Plot several copies of the the data, spread around
            symmetrically.
        power_cutoff : float or None
            if not None, only plot target structures on which the
            total power flux on the structure is larger then the
            cutoff.
        verbose : boolean
            print informations during plotting
        vmax : None or float
            if not None set the upper limit of the colorbar
        vmin : None or float
            if not None set the lower limit of the colorbar
        only_lower : boolean
            only plot structures that are below :math:`z=0`
        title : str
            set a title for the plot
        phi_slices : boolean
            plot some additional debugging slices
        path1 : None or str
            Take a screenshot and save at location of str if not None
        path2 : None or str
            like path1 but screenshot is taken from different position
        """
        if vmax is None:
            vmax = np.nanmax(ds[index])
        if vmin is None:
            vmin = np.nanmin(ds[index])
        fig = mlab.figure(size=(1920, 1080))

        if verbose:
            _pprint("\nplotting")
        for plate in ds.emc3.iter_plates(symmetry=symmetry, segments=segments):
            if power_cutoff and power_cutoff > plate.tot_P:
                continue

            z = plate.emc3["z_corners"].data
            if only_lower:
                doit = False
                data = plate.emc3[index].data.copy()
                for i, j in rrange(data.shape):
                    if (z[i : i + 2, j : j + 2] > 0).all():
                        data[i, j] = np.nan
                    else:
                        doit = True
                if not doit:
                    continue
            else:
                data = plate.emc3[index].data
            r = plate.emc3["R_corners"].data
            phi = plate.emc3["phi_corners"].data
            if phi.shape != r.shape:
                phi = phi.reshape((z.shape[0], 1)) * np.ones((1, z.shape[1]))
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            grid = [x, y, z]
            grid_centered = []
            ij = [(i % 2, i // 2) for i in range(4)]
            for s in grid:
                new = np.empty([(i - 1) * 2 for i in x.shape])
                slz = (slice(None, -1), slice(1, None))
                slz2 = (slice(None, None, 2), slice(1, None, 2))
                for i, j in ij:
                    new[slz2[i], slz2[j]] = s[slz[i], slz[j]]
                grid_centered.append(new)
            x, y, z = grid_centered
            datanew = np.empty_like(x)
            for i, j in ij:
                datanew[slz2[i], slz2[j]] = data
            msh = mlab.mesh(
                x, y, z, scalars=datanew, vmin=vmin, vmax=vmax, opacity=0.99
            )
            msh.module_manager.scalar_lut_manager.lut.nan_color = 0, 0, 0, 0
            if verbose:
                _pprint(".")
        if phi_slices:
            for phi in np.arange(np.pi * -0.2, np.pi * 0.2, np.pi / 180):
                R = np.array([[3.0, 6], [3, 6]])
                z = np.array([[1, 1], [-1, -1]])
                x = R * np.cos(phi)
                y = R * np.sin(phi)
                d = np.ones((2, 2))
                msh = mlab.mesh(x, y, z, scalars=d, opacity=0.99)
        if verbose:
            print(" done")
        if title:
            mlab.title(title, size=0.1, height=0.98)
        label = xr.plot.utils.label_from_attrs(ds[index])
        mlab.colorbar(title=label)
        # print("debug:",mlab.gcf() == fig)
        if path1:
            array = np.array
            mlab.view(
                -50,
                78.87131880429496,
                3.8,
                np.array([4.6, 0.17730785, -1.02087631]),
            )
            mlab.savefig(path1)  # , figure=self.fig)
            # print(mlab.screenshot())
        if path2:
            mlab.view(
                -28.62775839705135,
                64.3530947622391,
                2.6,
                array([4.28868816, -0.44711861, -1.2959868]),
            )
            # mlab.draw(fig)
            mlab.savefig(path2, figure=fig)  # , figure=self.fig)

        self.fig = fig
        # self.scene = scene
        # return self

    def savefig(self, path):
        return self

    def show(self):
        mlab.show()
        return self


class volume:
    def __init__(self, ds, updownsym=False, periodicity=1):
        print("__init__ started")
        print(updownsym, periodicity)
        self.ds = ds
        self.sym = updownsym
        self.period = periodicity
        vol_dims = [i[0] for i in [ds.r.shape, ds.theta.shape, ds.phi.shape]]
        # vol_dims[2] *= periodicity
        # if updownsym:
        #    vol_dims[2] *= 2
        print(vol_dims)
        self.dims = [i + 1 for i in vol_dims]
        self.grids = []
        mz = None
        z = ds.emc3["z_corners"].data.ravel(order="F")
        for i in range(self.period):
            ds["x_bounds"] = ds["R_bounds"] * np.cos(
                ds["phi_bounds"] + (2 * np.pi * i / self.period)
            )
            ds["y_bounds"] = ds["R_bounds"] * np.sin(
                ds["phi_bounds"] + (2 * np.pi * i / self.period)
            )
            x = ds.emc3["x_corners"].data.ravel(order="F")
            y = ds.emc3["y_corners"].data.ravel(order="F")
            self.grids.append(tvtk.StructuredGrid(dimensions=self.dims))
            self.grids[-1].points = np.ascontiguousarray(np.vstack([x, y, z]).T)
            if self.sym:
                if mz is None:
                    mz = -1 * z
                ds["x_bounds"] = ds["R_bounds"] * np.cos(
                    -ds["phi_bounds"] + (2 * np.pi * i / self.period)
                )
                ds["y_bounds"] = ds["R_bounds"] * np.sin(
                    -ds["phi_bounds"] + (2 * np.pi * i / self.period)
                )
                x = ds.emc3["x_corners"].data.ravel(order="F")
                y = ds.emc3["y_corners"].data.ravel(order="F")
                self.grids.append(tvtk.StructuredGrid(dimensions=self.dims))
                self.grids[-1].points = np.ascontiguousarray(np.vstack([x, y, mz]).T)

        # phi = ds["phi_bounds"]
        # phidims = phi.dims
        # phi = phi.data
        # R = ds["R_bounds"]
        # Rdims = R.dims
        # R = R.data
        # z = ds["z_bounds"].data
        # if self.sym:
        #     phi = np.append(-phi[::-1, ::-1], phi, axis=phidims.index("phi"))
        #     Rslc = [slice(None)] * 6
        #     Rslc[Rdims.index("phi")] = slice(None, None, -1)
        #     Rslc[Rdims.index("delta_phi")] = slice(None, None, -1)
        #     R = np.append(R[Rslc], R, axis=Rdims.index("phi"))
        #     z = np.append(-z[Rslc], z, axis=Rdims.index("phi"))
        # if self.period:
        #     phi0 = phi
        #     R0 = R
        #     z0 = z
        #     for i in range(self.period - 1):
        #         phi = np.append(
        #             phi,
        #             phi0 + (2 * np.pi) * (i + 1) / self.period,
        #             axis=phidims.index("phi"),
        #         )
        #         R = np.append(R, R0, axis=Rdims.index("phi"))
        #         z = np.append(z, z0, axis=Rdims.index("phi"))
        # dplot = xr.Dataset(
        #     coords=dict(
        #         R_bounds=(Rdims, R),
        #         z_bounds=(Rdims, z),
        #         phi_bounds=(phidims, phi),
        #     )
        # )
        print("__init__ finished")

    def plot(self, key, **kwargs):
        """
        Plot a some quantities in 3D

        Parameters
        ----------
        key : str
            The key to plot
        """
        print("plot started")
        # cell_data doesn't quite work ...
        # self.grid.cell_data.scalars = np.ascontiguousarray(self.ds[key].data).ravel(
        #    order="F"
        # )
        # self.grid.cell_data.scalars.name = "scalars"
        pnt_data = np.zeros(self.dims)
        fac = np.ones(self.dims)
        # rawdims = self.ds[key].dims
        rawdata = self.ds[key].data
        # if self.sym:
        #    rawdata = np.append(rawdata, rawdata, axis=rawdims.index("phi"))
        # rawdata0 = rawdata
        # for i in range(self.period - 1):
        #    rawdata = np.append(rawdata, rawdata0, axis=rawdims.index("phi"))
        for ijk in itertools.product(*[[slice(1, None), slice(None, -1)]] * 3):
            pnt_data[ijk] += rawdata
            fac[ijk] += 1
        pnt_data /= fac
        default = dict(opacity=0.9)
        default.update(kwargs)
        ml, mu = np.nanmin(pnt_data), np.nanmax(pnt_data)
        for i in range(self.period * (2 if self.sym else 2)):
            self.grids[i].point_data.scalars = pnt_data.ravel(order="F")
            self.grids[i].point_data.scalars.name = "scalars"
            d = mlab.pipeline.add_dataset(self.grids[i])
            # gx = mlab.pipeline.grid_plane(d)
            # gy = mlab.pipeline.grid_plane(d)
            # gy.grid_plane.axis = "y"
            # gz = mlab.pipeline.grid_plane(d)
            # gz.grid_plane.axis = "z"
            iso = mlab.pipeline.iso_surface(d, **default)
            # ml, mu = self.ds[key].min(), self.ds[key].max()
            iso.contour.maximum_contour = (mu - ml) / 2 + ml
        print("plot finished")
        return self

    def show(self):
        mlab.show()
        return self
