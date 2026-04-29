import numpy as np
import eudist  # type: ignore

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:
    tqdm = None  # type: ignore


def _rz_to_ab(rz, grid, ij):
    _, nz, _ = grid.shape
    nz -= 1
    i, j = ij // nz, ij % nz
    ABCD = grid[i : i + 2, j : j + 2]
    A = ABCD[0, 0]
    a = ABCD[0, 1] - A
    b = ABCD[1, 0] - A
    c = ABCD[1, 1] - A - a - b
    rz0 = rz - A

    def fun(albe):
        al, be = albe
        return rz0 - a * al - b * be - c * al * be

    def J(albe):
        al, be = albe
        return np.array([-a - c * be, -b - c * al])

    tol = 1e-13
    albe = np.ones(2) / 2
    while True:
        albe = albe - np.linalg.inv(J(albe).T) @ fun(albe)
        res = np.sum(fun(albe) ** 2)
        if res < tol:
            return albe


def _ab_to_rz(ab, grid, ij):
    _, nz, _ = grid.shape
    nz -= 1
    i, j = ij // nz, ij % nz
    A = grid[i, j]
    a = grid[i, j + 1] - A
    b = grid[i + 1, j] - A
    c = grid[i + 1, j + 1] - A - a - b
    al, be = ab
    return A + al * a + be * b + al * be * c


class OutOfDomainError(ValueError):
    pass


class FixPointConvergenceError(ValueError):
    pass


def rz_to_ab(rz, mesh, plot=False):
    ij = mesh.find_cell(rz)
    if ij < 0:
        print(ij)
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(mesh.r, mesh.z)
            plt.plot(*rz, "xr")
            plt.figure()
            rz1 = np.array([mesh.r, mesh.z])
            print(rz1.shape)
            pr, pz = np.unravel_index(
                np.argmin(np.sqrt(np.sum((rz1 - rz[:, None, None]) ** 2, axis=0))),
                rz1.shape[1:],
            )
            i = 0
            print(pr, pz)
            for nx in range(max(pr - 1, 0), min(pr + 2, rz1.shape[1] - 1)):
                nx = np.array([nx, nx + 2])
                r1 = mesh.r[nx]
                z1 = mesh.z[nx]
                for ny in range(pz - 1, pz + 2):
                    ny = [ny, ny + 1]
                    ny = [x % (rz1.shape[2] - 1) for x in ny]
                    plt.pcolormesh(
                        r1[:, ny], z1[:, ny], np.array([[i]]), vmin=0, vmax=9
                    )
                    i += 1

            plt.plot(*rz, "xr")
            plt.show()
        raise OutOfDomainError()
    return _rz_to_ab(rz, mesh.grid, ij), ij


def ab_to_rz(ab, ij, mesh):
    return _ab_to_rz(ab, mesh.grid, ij)


def trace(rz, meshes, n=100):
    pnts = np.empty((n, 2))
    pnts[0] = rz
    fac = np.array([1, -1])
    for i in range(1, n):
        try:
            abij = rz_to_ab(rz, meshes[0])
            rz = ab_to_rz(*abij, meshes[1])
            rz *= fac
            abij = rz_to_ab(rz, meshes[1])
            rz = ab_to_rz(*abij, meshes[0])
            rz *= fac
            pnts[i] = rz
        except OutOfDomainError:
            return pnts[:i]
    return pnts


def trace4(rz, meshes, n=100, ood=True):
    """
    Trace a point for a given number of iterations through the mesh
    """
    fac = np.array([1, -1])
    for i in range(n):
        abij = rz_to_ab(rz, meshes[0], plot=ood)
        rz = ab_to_rz(*abij, meshes[1])
        rz *= fac
        abij = rz_to_ab(rz, meshes[1], plot=ood)
        rz = ab_to_rz(*abij, meshes[0])
        rz *= fac
    return rz


def trace2(rz, meshes):
    """
    Trace up to a given point
    """
    fac = np.array([1, -1])
    if len(meshes) == 2:
        abij = rz_to_ab(rz, meshes[0])
        rz = ab_to_rz(*abij, meshes[1])
        return rz
    else:
        abij = rz_to_ab(rz, meshes[0])
        rz = ab_to_rz(*abij, meshes[1])
        rz *= fac
        abij = rz_to_ab(rz, meshes[1])
        rz = ab_to_rz(*abij, meshes[2])
        # rz *= fac
        return rz


class mymesh(eudist.PolyMesh):
    def __init__(self, x, y):
        super().__init__()
        self.r = x
        self.z = y
        self.grid = np.array([x, y]).transpose(1, 2, 0)

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(self.r, self.z)
        plt.show()


def getMeshes(ds):
    R = ds.emc3["R_corners"]
    Z = ds.emc3["z_corners"]

    return [mymesh(R.isel(phi=k).data, Z.isel(phi=k).data) for k in [0, -1]]


class Tracer:
    def __init__(self, ds):
        """
        Create a Tracer object

        Requires an xemc3 dataset as argument.
        """
        self.ds = ds
        self.R = ds.emc3["R_corners"]
        self.Z = ds.emc3["z_corners"]
        self.meshes = [
            mymesh(self.R.isel(phi=k).data, self.Z.isel(phi=k).data) for k in [0, -1]
        ]

    def poincare_phi0(self, pnts, n):
        """
        Always at phi=0
        """
        pnts = np.atleast_2d(pnts)
        return np.array([trace(rz, self.meshes, n=n) for rz in pnts])

    def trace_to_phi_index(self, pnts, phis, phi0=0, progress=False):
        """
        Trace points to phi_index.


        pnts: array, shape (n, 2), double

        phis: array, shape (m,), index

        phi0: integer
              index where pnts are

        returns:
            array, shape (m, n, 2)
        """

        def getmesh(self, phi):
            return mymesh(self.R.isel(phi=phi).data, self.Z.isel(phi=phi).data)

        pnts = np.array(pnts)
        assert (len(pnts.shape) == 2) and (pnts.shape[1] == 2), (
            f"Expected shape (n, 2) but got {pnts.shape}"
        )
        mesh0 = getmesh(self, phi0)
        meshes = [getmesh(self, phi) for phi in phis]
        mytrace2 = trace2
        if progress and tqdm:
            myprog = tqdm(total=len(meshes) * len(pnts))

            def mytrace2(*args):
                tmp = trace2(*args)
                myprog.update()
                return tmp

        return [[mytrace2(pnt, [mesh0, mesh1]) for pnt in pnts] for mesh1 in meshes]

    def fix_point(
        self,
        xy,
        periodicity=5,
        dx=1e-5,
        eps=1e-20,
        lim=1e-1,
        maxiter=100,
        plotOnError=False,
    ):
        """
        Try to converge to a fix point for xy.

        After periodicity iteration the point should be on the same
        position. For the axis periodicity would be 1 and for a 5/5 island
        chain, that would be 5.

        lim : float
            The allowed maximum step size. Larger steps abort the iteration.
        """
        ress = []
        while True:
            start = [xy, xy.copy(), xy.copy()]
            start[1][0] += dx
            start[2][1] += dx

            out = self.trace4(start, periodicity)

            fun = out[0] - xy

            J = (np.array([out[1] - start[1], out[2] - start[2]]) - fun) / dx

            update = np.linalg.inv(J.T) @ fun

            if np.sum(update**2) > lim:
                raise FixPointConvergenceError(
                    f"Failed at {xy} with proposed step {update}."
                )
            xy -= update
            res = np.sum(fun**2)
            ress.append(res)
            if len(ress) > maxiter:
                if plotOnError:
                    import matplotlib.pyplot as plt

                    plt.plot(ress)
                    plt.show()
                raise FixPointConvergenceError(
                    f"Aborting after {maxiter} iterations at {xy}."
                )
            dx = np.sqrt(res)
            if res < eps:
                return xy

    def trace4(self, pnts, num):
        """
        Trace one or more point for a given number of iterations through the mesh

        pnts: array, shape (n, 2) or (2,)

        num: integer
        """
        if isinstance(pnts, (list, tuple)):
            pnts = np.array(pnts)
        if len(pnts.shape) == 1:
            return trace4(pnts, self.meshes, num)
        return np.array([trace4(p, self.meshes, num) for p in pnts])
