``pd.read_<file>`` and I/O APIs
=================================

A number of IO methods default to pandas. We have parallelized ``read_csv``,
``read_parquet`` and some more (see table), though many of the remaining methods
can be relatively easily parallelized. Some of the operations default to the
pandas implementation, meaning it will read in serially as a single, non-distributed
DataFrame and distribute it. Performance will be affected by this.

The following table is structured as follows: The first column contains the method name.
The second column is a flag for whether or not there is an implementation in Modin for
the method in the left column. ``Y`` stands for yes, ``N`` stands for no, ``P`` stands
for partial (meaning some parameters may not be supported yet), and ``D`` stands for
default to pandas.

.. note::
    Currently, the second column reflects implementation status for ``Ray`` and ``Dask`` engines. By default, support for a method
    in the ``Hdk`` engine could be treated as ``D`` unless ``Notes`` column contains additional information.

.. note::
    Support for fully asynchronous reading has been added for the following functions:
    ``read_csv``, ``read_fwf``, ``read_table``, ``read_custom_text``.
    This mode is disabled by default, one can enable it using ``MODIN_ASYNC_READ_MODE=True``
    environment variable. Some parameter combinations are not supported and the function
    will be executed in synchronous mode.

+-------------------+---------------------------------+--------------------------------------------------------+
| IO method         | Modin Implementation? (Y/N/P/D) | Notes for Current implementation                       |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_csv`_       | Y                               | **Hdk**: ``P``, only basic cases and parameters        |
|                   |                                 | supported: ``filepath_or_buffer`` can be local file    |
|                   |                                 | only, ``sep``, ``delimiter``,  ``header`` (partly)     |
|                   |                                 | ``names``, ``usecols``, ``dtype``,                     |
|                   |                                 | ``true/false_values``, ``skiprows`` (partly)           |
|                   |                                 | ``skip_blank_lines`` (partly), ``parse_dates``         |
|                   |                                 | (partly), ``compression`` (inferred automatically,     |
|                   |                                 | should not be specified), ``quotechar``,               |
|                   |                                 | ``escapechar``, ``doublequote``,                       |
|                   |                                 | ``delim_whitespace``                                   |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_fwf`_       | Y                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_table`_     | Y                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_parquet`_   | P                               | Parameters besides ``filters`` and ``storage_options`` |
|                   |                                 | passed via ``**kwargs`` are not supported.             |
|                   |                                 | ``use_nullable_dtypes`` == True is not supported.      |
|                   |                                 |                                                        |
|                   |                                 | Experimental implementation: read_parquet_glob         |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_json`_      | P                               | Implemented for ``lines=True``                         |
|                   |                                 | Experimental implementation: read_json_glob            |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_xml`        | D                               | Experimental implementation: read_xml_glob             |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_html`_      | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_clipboard`_ | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_excel`_     | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_hdf`_       | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_feather`_   | Y                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_stata`_     | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_sas`_       | D                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_pickle`_    | D                               | Experimental implementation:                           |
|                   |                                 | read_pickle_distributed                                |
+-------------------+---------------------------------+--------------------------------------------------------+
| `read_sql`_       | Y                               |                                                        |
+-------------------+---------------------------------+--------------------------------------------------------+

.. _`read_csv`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html#pandas.read_csv
.. _`read_fwf`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_fwf.html#pandas.read_fwf
.. _`read_table`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_table.html#pandas.read_table
.. _`read_parquet`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_parquet.html#pandas.read_parquet
.. _`read_json`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html#pandas.read_json
.. _`read_html`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_html.html#pandas.read_html
.. _`read_clipboard`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_clipboard.html#pandas.read_clipboard
.. _`read_excel`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html#pandas.read_excel
.. _`read_hdf`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_hdf.html#pandas.read_hdf
.. _`read_feather`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_feather.html#pandas.read_feather
.. _`read_stata`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_stata.html#pandas.read_stata
.. _`read_sas`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sas.html#pandas.read_sas
.. _`read_pickle`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_pickle.html#pandas.read_pickle
.. _`read_sql`: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html#pandas.read_sql
