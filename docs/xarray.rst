Using xarray
============

Printing the dataset
--------------------

xarray by default only prints `12 rows by default
<https://docs.xarray.dev/en/stable/generated/xarray.set_options.html>`_.

This can be changed using ``xarray.set_options``, for example using a context
manager to preserve the default outside of the block:

.. code-block:: python

  with xr.set_options(display_max_rows=40):
      print(ds)

An alternative way to get all the variables, is to convert the dataset to a
list before printing. This however only prints the keys, no additional data:

.. code-block:: python

  print(list(ds))


Merging datasets
----------------

If you run a parameter study, it is convienient to have all runs in a single
dataset.

.. code-block:: python

  dss = []
  for n in ns:
      dss.append([])
      for d in Ds:
          dir = name(n, d)
          dss[-1].append(load(dir))
  
  ds = xr.combine_nested(dss, ["N", "D"])
  ds["N"] = [x * 1e19 for x in ns]
  ds.N.attrs = dict(long_name="Separatrix density", units="m^{-3}")
  ds["D"] = [x * 0.1 for x in Ds]
  ds.D.attrs = dict(long_name="Diffusion coefficient", units="m^2/s")
  

This allows to easily share the simulation results. Note that variables that
are the same for all runs, for example the grid data, will be automatically
deduplicated.
