Input/Output
~~~~~~~~~~~~

.. currentmodule:: modin.pandas

Modin's I/O functions API is backed by a distributed object(s) providing an identical
API to pandas. After the user calls some I/O function, this call is internally
rewritten into a representation that can be processed in parallel by the partitions.
Once I/O function call is finished, each partition will contain chunk of data, and then
these partitions can be processed in parallel using Modin API.

.. autosummary::
    :toctree: api/

    json_normalize
    read_clipboard
    read_csv
    read_excel
    read_feather
    read_fwf
    read_gbq
    read_hdf
    read_html
    read_json
    read_orc
    read_parquet
    read_pickle
    read_sas
    read_spss
    read_sql
    read_sql_query
    read_sql_table
    read_stata
    read_table
    read_xml
    to_pickle
