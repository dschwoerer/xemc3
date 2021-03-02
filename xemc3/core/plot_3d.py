import numpy as np
import mayavi.mlab as mlab
from .utils import rrange
import sys
import xarray as xr
from tvtk.api import tvtk


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
    def __init__(self, ds):
        print("__init__ started")
        self.ds = ds
        vol_dims = [ds.r.shape, ds.theta.shape, ds.phi.shape]
        print(vol_dims)
        self.dims = [i[0] + 1 for i in vol_dims]
        sgrid = tvtk.StructuredGrid(dimensions=self.dims)
        ds["x_bounds"] = ds["R_bounds"] * np.cos(ds["phi_bounds"])
        ds["y_bounds"] = ds["R_bounds"] * np.sin(ds["phi_bounds"])
        x = ds.emc3["x_corners"].data.ravel(order="F")
        y = ds.emc3["y_corners"].data.ravel(order="F")
        z = ds.emc3["z_corners"].data.ravel(order="F")
        sgrid.points = np.ascontiguousarray(np.vstack([x, y, z]).T)
        self.grid = sgrid
        print("__init__ finished")

    def plot(self, key):
        """
        Plot a some quantities in 3D

        Parameters
        ----------
        key : str
            The key to plot
        """
        print("plot started")
        self.grid.point_data.scalars = np.ascontiguousarray(self.ds[key].data).ravel(
            order="F"
        )
        self.grid.point_data.scalars.name = "scalars"
        print("plot finished")
        return self

    def show(self):
        mlab.show()
        return self
