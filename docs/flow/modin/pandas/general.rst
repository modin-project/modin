General Functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: modin.pandas

Modin's general functions API is backed by a distributed object(s) providing an identical
API to pandas. After the user calls some general function, this call is internally
rewritten into a representation that can be processed in parallel by the partitions. These
results can be e.g., reduced to single output, identical to the single threaded
pandas method output.

.. autosummary::
    :toctree: api/

    concat
    crosstab
    get_dummies
    isna
    isnull
    lreshape
    melt
    merge
    merge_asof
    merge_ordered
    notna
    notnull
    pivot
    pivot_table
    to_datetime
    to_numeric
    unique
    value_counts
    wide_to_long
