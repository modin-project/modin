``pd.Series`` supported APIs
============================

The ``pd.Series`` supported APIs table lists both implemented and not implemented methods.
If you have need of an operation that is listed as not implemented, feel free to open an
issue on the `GitHub repository`_, or give a thumbs up to already created issues. Contributions
are also welcome!

Supported APIs table is structured as follows: The first column contains the method name,
the second, third and fourth columns contain different flags which describes overall status for
whole method for concrete execution, and the last column contains method supported APIs
notes. In order to check method parameters supported status please follow method link. Please note,
that the tables only list unsupported/partially supported parameters. If a parameter is supported,
it won't be present or marked somehow in the table.

The flags stand for the following:

.. table::
   :widths: 1, 5

   +-------------+-----------------------------------------------------------------------------------------------+
   | Flag        | Meaning                                                                                       |
   +=============+===============================================================================================+
   | Harmful     | Usage of this parameter can be harmful for performance of your application. This usually      |
   |             | happens when parameter (full range of values and all types) is not supported and Modin        |
   |             | is defaulting to pandas (see more on defaulting to pandas mechanism on                        |
   |             | :doc:`defaulting to pandas page </supported_apis/defaulting_to_pandas>`)                      |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Non-lazy    | Usage of this parameter can trigger non-lazy execution (actual for OmniSci execution only)    |
   +-------------+-----------------------------------------------------------------------------------------------+
   | Partial     | Parameter can be partly unsupported, it's usage can be harmful for performance of your        |
   |             | appcication. This can happen if some parameter values or types are not supported (for example |
   |             | boolean values are suported while integer are not) and default pandas implementation is used  |
   |             | (see more on defaulting to pandas mechanism on                                                |
   |             | :doc:`defaulting to pandas page </supported_apis/defaulting_to_pandas>`)                      |
   +-------------+-----------------------------------------------------------------------------------------------+
   | pure pandas | Usage of this parameter, triggers usage of original pandas function as is, no performance     |
   |             | degradation/improvement should be observed                                                    |
   +-------------+-----------------------------------------------------------------------------------------------+

Supported APIs table
--------------------

.. csv-table::
   :file: series_supported.csv
   :header-rows: 1

.. _`GitHub repository`: https://github.com/modin-project/modin/issues
