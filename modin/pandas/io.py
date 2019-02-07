from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import pandas
import re

from .dataframe import DataFrame
from modin.data_management.factories import BaseFactory

PQ_INDEX_REGEX = re.compile("__index_level_\d+__")  # noqa W605


# Parquet
def read_parquet(path, engine="auto", columns=None, **kwargs):
    """Load a parquet object from the file path, returning a DataFrame.

    Args:
        path: The filepath of the parquet file.
              We only support local files for now.
        engine: This argument doesn't do anything for now.
        kwargs: Pass into parquet's read_pandas function.
    """
    return DataFrame(
        query_compiler=BaseFactory.read_parquet(
            path=path, columns=columns, engine=engine, **kwargs
        )
    )


# CSV and table
def _make_parser_func(sep):
    """Creates a parser function from the given sep.

    Args:
        sep: The separator default to use for the parser.

    Returns:
        A function object.
    """

    def parser_func(
        filepath_or_buffer,
        sep=sep,
        delimiter=None,
        header="infer",
        names=None,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=None,
        mangle_dupe_cols=True,
        dtype=None,
        engine=None,
        converters=None,
        true_values=None,
        false_values=None,
        skipinitialspace=False,
        skiprows=None,
        nrows=None,
        na_values=None,
        keep_default_na=True,
        na_filter=True,
        verbose=False,
        skip_blank_lines=True,
        parse_dates=False,
        infer_datetime_format=False,
        keep_date_col=False,
        date_parser=None,
        dayfirst=False,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal=b".",
        lineterminator=None,
        quotechar='"',
        quoting=0,
        escapechar=None,
        comment=None,
        encoding=None,
        dialect=None,
        tupleize_cols=None,
        error_bad_lines=True,
        warn_bad_lines=True,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
    ):
        _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
        return _read(**kwargs)

    return parser_func


def _read(**kwargs):
    """Read csv file from local disk.
    Args:
        filepath_or_buffer:
              The filepath of the csv file.
              We only support local files for now.
        kwargs: Keyword arguments in pandas.read_csv
    """
    pd_obj = BaseFactory.read_csv(**kwargs)
    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    return DataFrame(query_compiler=pd_obj)


read_table = _make_parser_func(sep="\t")
read_csv = _make_parser_func(sep=",")


def read_json(
    path_or_buf=None,
    orient=None,
    typ="frame",
    dtype=True,
    convert_axes=True,
    convert_dates=True,
    keep_default_dates=True,
    numpy=False,
    precise_float=False,
    date_unit=None,
    encoding=None,
    lines=False,
    chunksize=None,
    compression="infer",
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_json(**kwargs))


def read_gbq(
    query,
    project_id=None,
    index_col=None,
    col_order=None,
    reauth=False,
    verbose=None,
    private_key=None,
    dialect="legacy",
    **kwargs
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))
    return DataFrame(query_compiler=BaseFactory.read_gbq(**kwargs))


def read_html(
    io,
    match=".+",
    flavor=None,
    header=None,
    index_col=None,
    skiprows=None,
    attrs=None,
    parse_dates=False,
    tupleize_cols=None,
    thousands=",",
    encoding=None,
    decimal=".",
    converters=None,
    na_values=None,
    keep_default_na=True,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_html(**kwargs))


def read_clipboard(sep=r"\s+"):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_clipboard(**kwargs))


def read_excel(
    io,
    sheet_name=0,
    header=0,
    names=None,
    index_col=None,
    usecols=None,
    squeeze=False,
    dtype=None,
    engine=None,
    converters=None,
    true_values=None,
    false_values=None,
    skiprows=None,
    nrows=None,
    na_values=None,
    parse_dates=False,
    date_parser=None,
    thousands=None,
    comment=None,
    skipfooter=0,
    convert_float=True,
    **kwds
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.pop("kwds")
    kwargs.update(kwds)
    return DataFrame(query_compiler=BaseFactory.read_excel(**kwargs))


def read_hdf(path_or_buf, key=None, mode="r", **kwargs):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))
    return DataFrame(query_compiler=BaseFactory.read_hdf(**kwargs))


def read_feather(path, nthreads=1):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    kwargs.update(kwargs.pop("kwargs", {}))
    return DataFrame(query_compiler=BaseFactory.read_feather(**kwargs))


def read_msgpack(path_or_buf, encoding="utf-8", iterator=False):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_msgpack(**kwargs))


def read_stata(
    filepath_or_buffer,
    convert_dates=True,
    convert_categoricals=True,
    encoding=None,
    index_col=None,
    convert_missing=False,
    preserve_dtypes=True,
    columns=None,
    order_categoricals=True,
    chunksize=None,
    iterator=False,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_stata(**kwargs))


def read_sas(
    filepath_or_buffer,
    format=None,
    index=None,
    encoding=None,
    chunksize=None,
    iterator=False,
):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_sas(**kwargs))


def read_pickle(path, compression="infer"):
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_pickle(**kwargs))


def read_sql(
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
):
    """ Read SQL query or database table into a DataFrame.

    Args:
        sql: string or SQLAlchemy Selectable (select or text object) SQL query to be executed or a table name.
        con: SQLAlchemy connectable (engine/connection) or database string URI or DBAPI2 connection (fallback mode)
        index_col: Column(s) to set as index(MultiIndex).
        coerce_float: Attempts to convert values of non-string, non-numeric objects (like decimal.Decimal) to
                      floating point, useful for SQL result sets.
        params: List of parameters to pass to execute method. The syntax used
                to pass parameters is database driver dependent. Check your
                database driver documentation for which of the five syntax styles,
                described in PEP 249's paramstyle, is supported.
        parse_dates:
                     - List of column names to parse as dates.
                     - Dict of ``{column_name: format string}`` where format string is
                       strftime compatible in case of parsing string times, or is one of
                       (D, s, ns, ms, us) in case of parsing integer timestamps.
                     - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
                       to the keyword arguments of :func:`pandas.to_datetime`
                       Especially useful with databases without native Datetime support,
                       such as SQLite.
        columns: List of column names to select from SQL table (only used when reading a table).
        chunksize: If specified, return an iterator where `chunksize` is the number of rows to include in each chunk.

    Returns:
        Modin Dataframe
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=BaseFactory.read_sql(**kwargs))
