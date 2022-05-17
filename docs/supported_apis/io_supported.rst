``pd.read_<file>`` and I/O APIs
=================================

The I/O supported APIs table lists both implemented and not implemented methods.
If you have need of an operation that is listed as not implemented, feel free to open an
issue on the `GitHub repository`_, or give a thumbs up to already created issues. Contributions
are also welcome!

Supported APIs table is structured as follows: The first column contains the method name,
the second column - the parameter name of this method, and other columns contain
different flags describing particular properties of method parameters for a concrete
execution.

The flags stand for the following:

.. table::
   :widths: 1, 5

   +-------------+-----------------------------------------------------------------------------------------------+
   | Flag        | Meaning                                                                                       |
   +=============+===============================================================================================+
   | Supported   | Parameter is supported, it's usage brings performance improvement                             |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Harmful     | Usage of this parameter can be harmful for performance of your application. Usually this      |
   |             | happens when parameter (full range of values and all types) is not supported and default      |
   |             | pandas implementation is used                                                                 |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Non-lazy    | Usage of this parameter can trigger non-lazy execution (actual for OmniSci execution only)    |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Partial     | Parameter can be partly unsupported, it's usage can be harmful for performance of your        |
   |             | appcication. This can happen if some parameter values or types are not supported (for example |
   |             | boolean values are suported while integer are not) and default pandas implementation is used  |
   +-------------+-----------------------------------------------------------------------------------------------+
   | pure pandas | Usage of this parameter, triggers usage of original pandas function as is, no performance     |
   |             | degradation/improvement should be observed                                                    |
   +-------------+-----------------------------------------------------------------------------------------------+

Parameters Notes
----------------

.. csv-table::
   :file: io_supported.csv
   :header-rows: 1

.. _`GitHub repository`: https://github.com/modin-project/modin/issues
