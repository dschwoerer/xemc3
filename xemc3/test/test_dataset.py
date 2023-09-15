import warnings
from timeit import timeit

import numpy as np
import pytest
import xarray as xr

from ..core import dataset
from . import gen_ds

try:
    import matplotlib  # type: ignore
except ImportError:
    matplotlib = None  # type: ignore


def test_sel():
    ds = xr.Dataset()

    ds.emc3["r_corners"] = "r", np.linspace(0, 4, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3, 4)

    for i in range(5):
        dss = ds.emc3.sel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == min(i, 3)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.sel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == int(i)


def test_isel():
    ds = xr.Dataset()
    ds.emc3["r_corners"] = "r", np.linspace(0, 4, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3, 4)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == min(i, 3)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i
        assert dss["x"] == int(i)


def test_isel2():
    ds = xr.Dataset()
    rf = 2
    xf = 3
    ds.emc3["r_corners"] = "r", np.linspace(0, 4 * rf, 5)
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i * rf
        assert dss["x"] == min(i, 3) * xf
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert dss["r_bounds"] == i * rf
        assert dss["x"] == int(i) * xf


def test_isel_multi():
    ds = xr.Dataset()
    rf = 2
    zf = 5
    xf = 3
    rr = np.linspace(0, 4 * rf, 5)
    zr = np.linspace(0, 7 * zf, 8)
    ds.emc3["r_corners"] = ("r", "z"), (rr[:, None] + zr[None, :])
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)
    ds.emc3["y"] = "z", np.linspace(0, 6 * xf, 7)
    # ds.emc3["k"] = ("r","z"), np.linspace(0, 6 * xf, 7)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert all(dss.emc3["r_corners"] == i * rf + zr)
        assert dss["x"] == min(i, 3) * xf
        assert all(dss["y"] == ds["y"])
    for j in range(8):
        dss = ds.emc3.isel(z=j)
        assert all(dss.emc3["r_corners"] == j * zf + rr)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert np.allclose(dss.emc3["r_corners"], i * rf + zr)
        assert dss["x"] == int(i) * xf


def test_isel_multi2():
    ds = xr.Dataset()
    rf = 2
    zf = 5
    xf = 3
    rr = np.linspace(0, 4 * rf, 5)
    zr = np.linspace(0, 7 * zf, 8)
    ds_ = ds.copy()
    ds_.emc3["r_corners"] = ("r", "z"), (rr[:, None] + zr[None, :])
    ds = ds.assign_coords({"r_bounds": ds_["r_bounds"]})
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)
    ds.emc3["y"] = "z", np.linspace(0, 6 * xf, 7)
    # ds.emc3["k"] = ("r","z"), np.linspace(0, 6 * xf, 7)

    for i in range(5):
        dss = ds.emc3.isel(r=i)
        assert all(dss.emc3["r_corners"] == i * rf + zr)
        assert dss["x"] == min(i, 3) * xf
        assert all(dss["y"] == ds["y"])
    for j in range(8):
        dss = ds.emc3.isel(z=j)
        assert all(dss.emc3["r_corners"] == j * zf + rr)
    for i in np.random.random(10) * 4:
        dss = ds.emc3.isel(r=i)
        assert np.allclose(dss.emc3["r_corners"], i * rf + zr)
        assert dss["x"] == int(i) * xf


def test_sel_multi2():
    ds = xr.Dataset()
    rf = 2
    zf = 5
    xf = 3
    rr = np.linspace(0, 4 * rf, 5)
    zr = np.linspace(0, 7 * zf, 8)
    ds_ = ds.copy()
    ds_.emc3["r_corners"] = ("r", "z"), (rr[:, None] + zr[None, :])
    ds = ds.assign_coords({"r_bounds": ds_["r_bounds"]})
    ds.emc3["x"] = "r", np.linspace(0, 3 * xf, 4)
    ds.emc3["z_corners"] = "z", zr
    # ds.emc3["k"] = ("r","z"), np.linspace(0, 6 * xf, 7)

    for j in range(8):
        dss = ds.emc3.sel(z=j * zf)
        assert all(dss.emc3["r_corners"] == j * zf + rr)
    for i in np.random.random(10) * 7:
        dss = ds.emc3.sel(z=i * zf)
        assert np.allclose(dss.emc3["r_corners"], i * zf + rr)


def test_mean_dtype():
    ds = xr.Dataset()
    ds["pos"] = [1, 2, 3]
    ds["data"] = ("pos", "time"), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ds["var"] = "pos", [2, 3, 4]
    ds2 = ds.emc3.mean_time()
    assert all(ds2["var"] == ds["var"])
    assert ds2["var"].dtype == ds["var"].dtype


class Test_eval_at_rpz(object):
    def setup(self, shape=None, **kwargs):
        if not isinstance(shape, tuple):
            shape = None
        self.shape = shape or (2, 20, 4)
        self.geom = gen_ds.rotating_circle(5, **kwargs)
        self.ds = self.geom.gends(self.shape)
        self.dims = self.ds["_plasma_map"].dims
        self.dphi = 2 * np.pi / self.shape[2] / 5

    def rand_rpt(self, i):
        slc = [slice(None)]
        if isinstance(i, np.ndarray):
            slc += [None] * len(i.shape)
        else:
            slc += [None]
        tmp = (
            np.random.random((3, i))
            * np.array([self.geom.r * 0.9, np.pi * 2, np.pi * 2])[tuple(slc)]
        )
        r, p, t = tmp
        return r, p, t

    def rand(self, i):
        return self.geom.rpt_to_rpz(*self.rand_rpt(i))

    def phi_test_value(self, a, b):
        dphi = self.dphi
        phi = np.linspace(dphi / 2, 2 * np.pi / 5 - dphi / 2, self.shape[2])
        self.ds["var"] = self.dims, np.zeros(self.shape) + phi
        for r, p, z in [self.rand(a) for _ in range(b)]:
            exp = (np.round((p - dphi / 2) / dphi) % self.shape[2]) * dphi + dphi / 2
            got = self.ds.emc3.evaluate_at_rpz(r, p, z, "var", updownsym=self.geom.sym)[
                "var"
            ]
            assert np.allclose(
                exp,
                got,
            )

    def phi_test_index(self, a, b):
        dphi = self.dphi
        for r, p, z in [self.rand(a) for _ in range(b)]:
            assert np.allclose(
                np.round((p - dphi / 2) / dphi) % self.shape[2],
                self.ds.emc3.evaluate_at_rpz(r, p, z, updownsym=self.geom.sym)["phi"],
            )

    def test_phi_single1(self):
        self.setup()
        self.phi_test_value(1, 20)

    def test_phi_single2(self):
        self.setup()
        self.phi_test_index(1, 20)

    def test_phi_some1(self):
        self.setup()
        self.phi_test_value(2, 5)

    def test_phi_some2(self):
        self.setup()
        self.phi_test_value(5, 1)

    def test_phi_some3(self):
        self.setup()
        self.phi_test_index(4, 1)

    def r_test_value(self, a, b):
        dr = self.geom.r / self.shape[0]
        r = np.linspace(dr / 2, self.geom.r - dr / 2, self.shape[0])
        self.ds["var"] = self.dims, np.zeros(self.shape) + r[:, None, None]
        for r, p, t in [self.rand_rpt(a) for _ in range(b)]:
            expl = (np.round((r - dr * 0.51) / dr)) * dr + dr * 0.49
            exp = (
                np.round((r / np.cos(np.pi / self.shape[1]) - dr * 0.48) / dr)
            ) * dr + dr * 0.52
            R, p, z = self.geom.rpt_to_rpz(r, p, t)
            got = self.ds.emc3.evaluate_at_rpz(R, p, z, "var", updownsym=self.geom.sym)[
                "var"
            ].data
            if a == 1:
                isgood = expl < got < exp
            else:
                isgood = all([a < b < c for a, b, c in zip(expl, got, exp)])
            if not isgood:
                for i, t in enumerate([a < b < c for a, b, c in zip(expl, got, exp)]):
                    if not t:
                        import matplotlib.pyplot as plt  # type: ignore

                        self.ds.emc3.plot_rz(key=None, phi=p[i] % (np.pi * 2 / 5))
                        plt.plot(R[i], z[i], "rx")
                        plt.show()
            assert isgood, f"""{expl} <  {got} < {exp}
{R}, {p}, {z}
{r}, {p}, {t})"""

    def theta_test_value(self, a, b):
        dt = 2 * np.pi / self.shape[1]
        t = np.linspace(0, 2 * np.pi - dt, self.shape[1])
        self.ds["var"] = self.dims, np.zeros(self.shape) + t[None, :, None]
        for r, p, t in [self.rand_rpt(a) for _ in range(b)]:
            # No test within phi, as then we need to calculate where we end up, which is non-trivial
            p = np.zeros_like(p)
            exp = (np.round(((t)) / dt) % self.shape[1]) * dt
            R, p, z = self.geom.rpt_to_rpz(r, p, t)
            got = self.ds.emc3.evaluate_at_rpz(R, p, z, "var", updownsym=self.geom.sym)[
                "var"
            ].data
            isgood = np.allclose(got, exp)
            if not isgood:
                print(got / dt, exp / dt)
                print(R, p, z)
                print(r, p, t / dt)
                assert False

    def test_r_single(self):
        self.setup()
        self.r_test_value(1, 20)

    def test_r_some(self):
        self.setup()
        self.r_test_value(2, 3)
        self.r_test_value(3, 4)

    def test_theta_single(self):
        self.setup()
        self.theta_test_value(1, 20)

    def test_trace_line(self):
        self.setup(shape=(5, 30, 10))
        var = np.zeros(self.shape)
        for i in range(5):
            var[2, i * self.shape[1] // 5, :] = 1

        p = np.linspace(0, 2 * np.pi, 100)
        t = np.zeros_like(p)
        r = self.geom.r * 0.5 + np.zeros_like(p)
        R, p, z = self.geom.rpt_to_rpz(r, p, t)

        self.ds["var"] = self.dims, var

        # plt = self.ds.emc3.plot(
        #     "var", updownsym=self.geom.sym, periodicity=1  # self.geom.period
        # )
        # import mayavi.mlab as mlab  # type: ignore

        # for i in range(5):
        #     mlab.plot3d(
        #         *self.geom.rpz_to_xyz(
        #             *self.geom.rpt_to_rpz(r, p, t + i * np.pi * 2 / 5)
        #         ),
        #         opacity=0.9,
        #     )
        # plt.show()
        got = self.ds.emc3.evaluate_at_rpz(R, p, z, "var", updownsym=self.geom.sym)[
            "var"
        ].data
        assert np.allclose(got, 1), f"got {got} but expected 1"

    def test_trace_line_updown(self):
        self.setup(shape=(5, 60, 10), sym=True)
        var = np.zeros(self.shape)
        for i in range(5):
            var[2, i * self.shape[1] // 5, :] = 1

        p = np.linspace(0, 2 * np.pi, 100)
        t = np.zeros_like(p)  # + np.pi / self.shape[1]
        r = self.geom.r * 0.5 + np.zeros_like(p)
        R, p, z = self.geom.rpt_to_rpz(r, p, t)
        self.ds["var"] = self.dims, var

        # plt = self.ds.emc3.plot(
        #     "var", updownsym=self.geom.sym, periodicity=1  # self.geom.period
        # )
        # import mayavi.mlab as mlab  # type: ignore

        # for i in range(5):
        #     mlab.plot3d(
        #         *self.geom.rpz_to_xyz(
        #             *self.geom.rpt_to_rpz(r, p, t + i * np.pi * 2 / 5)
        #         ),
        #         opacity=0.9,
        #     )
        # plt.show()
        got = self.ds.emc3.evaluate_at_rpz(R, p, z, "var", updownsym=self.geom.sym)[
            "var"
        ].data
        assert np.allclose(got, 1), f"got {got} but expected 1"

    def test_cached_eval(self):
        a = 200
        b = 1
        self.setup()  # (10, 20, 30))
        dphi = self.dphi
        for r, p, z in [self.rand(a) for _ in range(b)]:
            exp = np.round((p - dphi / 2) / dphi) % self.shape[2]
            got = self.ds.emc3.evaluate_at_rpz(
                r, p, z, updownsym=self.geom.sym, delta_phi=dphi
            )["phi"]
            assert np.allclose(
                exp, got
            ), f"Expected \n{exp} but got \n{got.data} \n{p/dphi} % {self.shape[2]}"
            assert got.dims == ("dim_0",)

    def test_cached_eval_perf(self):
        self.setup((2, 20, 30))
        dphi = self.dphi
        ds = self.ds
        r, p, z = self.rand(40)
        cached = timeit(
            """ds.emc3.evaluate_at_rpz(
                r, p, z, updownsym=self.geom.sym, delta_phi=dphi
        )""",
            number=1,
            globals=locals(),
        )

        slow = timeit(
            """ds.emc3.evaluate_at_rpz(
                r, p, z, updownsym=self.geom.sym
        )""",
            number=1,
            globals=locals(),
        )
        print(slow, cached, slow / cached)
        assert (
            slow > cached
        ), f"Expected the cached version to be faster then the non-cached {slow} vs {cached}. Note that this might sometimes fail. Increase the number of samples to avoid that."

    def test_nan_value(self):
        self.setup()
        dt = 2 * np.pi / self.shape[1]
        t = np.linspace(0, 2 * np.pi - dt, self.shape[1])
        self.ds["var"] = self.dims, np.zeros(self.shape) + t[None, :, None]
        r, p, t = self.rand_rpt(6)
        # No test within phi, as then we need to calculate where we end up, which is non-trivial
        p = np.zeros_like(p)
        exp = (np.round(((t)) / dt) % self.shape[1]) * dt
        dat = np.array([r, p, t, exp])
        dat[:, [0, 2, 5]] = np.nan
        r, p, t, exp = dat
        R, p, z = self.geom.rpt_to_rpz(r, p, t)
        got = self.ds.emc3.evaluate_at_rpz(R, p, z, "var", updownsym=self.geom.sym)[
            "var"
        ].data
        isgood = all(np.isnan(exp) == np.isnan(got))
        assert isgood, f"""
        {exp / dt} {got / dt}
        {R} {p} {z}
        {r} {p} {t / dt}
"""

    def test_at_xyz(self):
        self.setup()
        dt = 2 * np.pi / self.shape[1]
        t = np.linspace(0, 2 * np.pi - dt, self.shape[1])
        self.ds["var"] = self.dims, np.zeros(self.shape) + t[None, :, None]
        r, p, t = self.rand_rpt(6)
        # No test within phi, as then we need to calculate where we end up, which is non-trivial
        p = np.zeros_like(p)
        exp = (np.round(((t)) / dt) % self.shape[1]) * dt
        Rpz = self.geom.rpt_to_rpz(r, p, t)
        xyz = self.geom.rpz_to_xyz(*Rpz)
        got = self.ds.emc3.evaluate_at_xyz(*xyz, "var", updownsym=self.geom.sym)[
            "var"
        ].data
        assert all(np.isclose(got, exp))

    def test_at_dataarrays(self):
        self.setup()
        t = np.linspace(0, 2 * np.pi, self.shape[2])
        self.ds["var"] = self.dims, np.zeros(self.shape) + t
        ds = xr.Dataset(coords=dict(x=np.linspace(0, 3, 5), y=np.linspace(0, 2, 6)))
        x = ds.x
        y = ds.y
        for z in 0, x * y:
            result = self.ds.emc3.evaluate_at_xyz(
                x, y, z, "var", updownsym=self.geom.sym
            )
            assert "x" in result.dims
            assert "y" in result.dims
            assert all(result.coords["x"] == x)
            assert all(result.coords["y"] == y)
            assert result["var"].dims == ("x", "y")
            assert result["var"].shape == (5, 6)

    def test_plot_rz(self):
        """
        Ensure no error is raised.
        """
        if matplotlib is None:
            pytest.skip("matplotlib missing")
        self.setup()
        ret = self.ds.emc3.plot_rz(key=None, phi=0.3)
        assert isinstance(ret, matplotlib.collections.QuadMesh)

    def test_plot_Rz(self):
        """
        Ensure no error is raised.
        """
        if matplotlib is None:
            pytest.skip("matplotlib missing")
        self.setup()
        with warnings.catch_warnings(record=True) as w:
            ret = self.ds.emc3.plot_Rz(key=None, phi=0.3)
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "plot_rz" in str(w[-1].message)
        assert isinstance(ret, matplotlib.collections.QuadMesh)


if __name__ == "__main__":
    Test_eval_at_rpz().test_nan_value()
