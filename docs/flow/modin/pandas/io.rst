Input/Output
~~~~~~~~~~~~

.. currentmodule:: modin.pandas

Modin's I/O functions API is backed by a distributed object(s) providing an identical
API to pandas. After the user calls some IO function, this call is internally
rewritten into a representation that can be processed in parallel by the partitions. These
results can be e.g., reduced to single output, identical to the single threaded
pandas method output.

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
