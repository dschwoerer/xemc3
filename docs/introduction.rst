Getting started
===============

Installing
----------
To get the full install including plotting run:

.. code-block:: bash

  pip install --user 'xemc3[full]'

If you want a minimal install without plotting capabilities you can run:

.. code-block:: bash

  pip install --user xemc3

Note that without matplotlib plotting will not be available.


Basic tasks
-----------

Checking convergence
~~~~~~~~~~~~~~~~~~~~

If you have run a simulations, typically the first thing you want to do, is
check the ``*_INFO`` files to check whether the simulation is converged.

This can be done by reading in the info files, and plotting the traces, to
have a look at the variation:


.. code-block:: python

  import xemc3
  import matplotlib.pyplot as plt
  ds = xemc3.load.file("path_to_simu/ENERGY_INFO")
  for key in ds:
      ds[key].plot()
  plt.show()

See `an example notebook <examples/info.ipynb>`_ for more info on reading these files.



Reading the output
~~~~~~~~~~~~~~~~~~

``xemc3`` assumes that all files are called the way that also EMC3 is
expecting them to be named. If you use different names, ensure that
the files are linked to the EMC3 names, before loading the files with
``xemc3``.

Once the simulation is sufficiently converged, further analysis can start.
For this ``xemc3`` allows to read in all the files, and store the data as a
netcdf file. This has the advantage that successive reads are very fast, and
is especially convenient if the data is stored in the netcdf file after
running the simulation. There is a command line version, that can be
conveniently called from shell, `xemc3-to-netcdf
<cli.html#xemc3-to-netcdf---cli-interface>`_ that is roughly equivalent to the
following python snippet:

.. code-block:: python

  import xemc3
  ds = xemc3.load.all(path)
  ds.to_netcdf(path + ".nc")

Besides faster loads, the netcdf also makes it easier to share the
data for analysis, as all data is stored in a single file. This also
allows to unlink the EMC3 names, or share the data with users that
have a different naming convention for their files.
The netcdf files can then be read again via

.. code-block:: python

  import xarray as xr
  ds = xr.open_dataset(path + ".nc")



Post processing
~~~~~~~~~~~~~~~

`xarray <https://pypi.org/project/xarray/>`_ provides a wide variety of
functionality for post processing.  Good `documentation
<https://xarray.pydata.org/en/stable/index.html>`_ and `tutorials
<https://xarray-contrib.github.io/xarray-tutorial/index.html>`_ are available.


Plotting
~~~~~~~~

xarray handles plotting already, but xemc3 extends this with some more
specific routines, for example to plot an :math:`R\times z` slice.  The
functionally is documented `here <xemc3.html>`_ and can be accessed via the
``emc3`` accessor of a dataset, for example the
`xemc3.EMC3DatasetAccessor.plot_rz
<xemc3.html#xemc3.EMC3DatasetAccessor.plot_rz>`_ can be used by calling
``ds.emc3.plot_rz(...)`` with ``ds`` an `xr.Dataset
<https://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_.
Plotting in simulation coordinates can be done using `xr.DataArray.plot
<https://xarray.pydata.org/en/stable/generated/xarray.Dataset.html>`_ as e.g.
``ds["Te"].isel(r=-3).plot()`` to plot the third outermost slice of the
electron temperature.


Exercises
---------

To get an overview what is possible with xemc3, you can try the exercises.

You can find them in the ``docs/exercises/`` folder or try them `online
<https://mybinder.org/v2/gh/dschwoerer/xemc3/next?filepath=docs%2Fexercises>`_.
