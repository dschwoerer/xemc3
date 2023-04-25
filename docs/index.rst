.. xemc3 documentation master file, created by
   sphinx-quickstart on Mon Dec 10 14:17:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   2021: Copied from xBOUT

Welcome to xemc3's documentation!
=================================

:py:mod:`xemc3` provides an interface for collecting the output data
from a ``EMC3`` simulation into an xarray_ dataset, as well as accessor
methods for common EMC3 analysis and plotting tasks.

Currently only in alpha (until 1.0 released) so please report any
bugs, and feel free to raise issues asking questions or making
suggestions.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   examples
   xemc3
   xarray
   config
   filenames
   cli
   citing

Installation
------------

With ``pip``:

.. code-block:: bash

  pip install --user xemc3

You can run the tests by running ``pytest --pyargs xemc3``.

xemc3 will install the required python packages when you run one of
the above install commands if they are not already installed on your
system.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _xarray: https://xarray.pydata.org/en/stable/index.html
