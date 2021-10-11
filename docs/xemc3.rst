API reference
=============


xemc3 Dataset
-----------------

All of the functions that are part of the EMC3DatasetAccessor can be
accessed via the ``emc3`` dataset accessor. For a given dataset ``ds``
they can be reached as ``ds.emc3.*``, e.g. ``ds.emc3.plot_rz(...)``.

.. automodule:: xemc3.EMC3DatasetAccessor


xemc3.load module
-----------------

.. note::
  Please ensure that files are linked to their EMC3 names, if
  you use alternative file names.

.. automodule:: xemc3.load


xemc3.write module
------------------

.. automodule:: xemc3.write.fortran
.. automodule:: xemc3.write.nc
