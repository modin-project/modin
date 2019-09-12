Pandas on Ray
=============

This section describes usage related documents for the Pandas on Ray component of Modin.

.. raw:: html

   <img src="https://img.shields.io/badge/pandas%20api%20coverage-71.77%25-orange.svg">

Currently, Modin support ~71% of the pandas API. The exact methods we have implemented
are listed in the respective subsections:

* DataFrame_
* Series_
* utilities_
* `I/O`_

We have taken a community-driven approach to implementing new methods. We did a `study
on pandas usage`_ to learn what the most-used APIs are. Modin currently supports **93%**
of the pandas API based on our study of pandas usage, and we are actively expanding the
API.

Modin uses Pandas on Ray by default, but if you wanted to be explicit, you could set the
following environment variables:

.. code-block:: bash

   export MODIN_ENGINE=ray
   export MODIN_BACKEND=pandas

.. toctree::
   :maxdepth: 1

   dataframe_supported
   series_supported
   utilities_supported
   io_supported

.. _DataFrame: dataframe_supported.html
.. _Series: series_supported.html
.. _utilities: utilities_supported.html
.. _I/O: io_supported.html
.. _study on pandas usage: https://github.com/modin-project/study_kaggle_usage
