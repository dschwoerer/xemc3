import xemc3
from xemc3.cli.append_time import append_time as append
import gen_ds as g
import tempfile
import xarray as xr
from hypothesis import given, settings, assume, strategies as st
from test_write_load import assert_ds_are_equal


@settings(deadline=None)
@given(
    g.hypo_shape(100),
    g.hypo_vars12(),
    st.integers(min_value=1, max_value=10),
)
def disabled_test_average(shape, v12, rep):
    v1, v2 = v12
    assume(len(v2))
    org = g.gen_rand(shape, v1)
    orgs = [org]
    read = []
    with tempfile.TemporaryDirectory() as dir:
        xemc3.write.fortran(org, dir)
        append(dir)
        for i in range(rep):
            d2 = g.gen_updated(org, v2)
            orgs.append(d2)
            xemc3.write.fortran(d2, dir)
            append(dir)
        read = xr.open_dataset(f"{dir}.nc")
        for i, o in enumerate(orgs):
            assert_ds_are_equal(o, read.isel(time=i), True, 1e-2, 1e-2)

        with tempfile.TemporaryDirectory() as dir2:
            xemc3.write.fortran(read.emc3.average_time(), dir2)
            dn = xemc3.load.all(dir2)
        # print(xemc3.core.load.files)
        assert_ds_are_equal(read.emc3.average_time(), dn, True, 1e-4)
