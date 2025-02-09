# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""
Module houses `BaseIO` class.

`BaseIO` is base class for IO classes, that stores IO functions.
"""

from typing import Any

import pandas
from pandas._libs.lib import no_default
from pandas.util._decorators import doc

from modin.core.storage_formats import BaseQueryCompiler
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.pandas.io import ExcelFile
from modin.utils import _inherit_docstrings

_doc_default_io_method = """
{summary} using pandas.
For parameters description please refer to pandas API.

Returns
-------
{returns}
"""

_doc_returns_qc = """BaseQueryCompiler
    QueryCompiler with read data."""

_doc_returns_qc_or_parser = """BaseQueryCompiler or TextParser
    QueryCompiler or TextParser with read data."""


class BaseIO:
    """Class for basic utils and default implementation of IO functions."""

    query_compiler_cls: BaseQueryCompiler = None
    frame_cls = None
    _should_warn_on_default_to_pandas: bool = True

    @classmethod
    def _maybe_warn_on_default(cls, *, message: str = "", reason: str = "") -> None:
        """
        If this class is configured to warn on default to pandas, warn.

        Parameters
        ----------
        message : str, default: ""
            Method that is causing a default to pandas.
        reason : str, default: ""
            Reason for default.
        """
        if cls._should_warn_on_default_to_pandas:
            ErrorMessage.default_to_pandas(message=message, reason=reason)

    @classmethod
    def from_non_pandas(cls, *args, **kwargs):
        """
        Create a Modin `query_compiler` from a non-pandas `object`.

        Parameters
        ----------
        *args : iterable
            Positional arguments to be passed into `func`.
        **kwargs : dict
            Keyword arguments to be passed into `func`.
        """
        return None

    @classmethod
    def from_pandas(cls, df):
        """
        Create a Modin `query_compiler` from a `pandas.DataFrame`.

        Parameters
        ----------
        df : pandas.DataFrame
            The pandas DataFrame to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the `pandas.DataFrame`.
        """
        return cls.query_compiler_cls.from_pandas(df, cls.frame_cls)

    @classmethod
    def from_arrow(cls, at):
        """
        Create a Modin `query_compiler` from a `pyarrow.Table`.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Arrow Table.
        """
        return cls.query_compiler_cls.from_arrow(at, cls.frame_cls)

    @classmethod
    def from_interchange_dataframe(cls, df):
        """
        Create a Modin QueryCompiler from a DataFrame supporting the DataFrame exchange protocol `__dataframe__()`.

        Parameters
        ----------
        df : DataFrame
            The DataFrame object supporting the DataFrame exchange protocol.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the DataFrame.
        """
        return cls.query_compiler_cls.from_interchange_dataframe(df, cls.frame_cls)

    @classmethod
    def from_ray(cls, ray_obj):
        """
        Create a Modin `query_compiler` from a Ray Dataset.

        Parameters
        ----------
        ray_obj : ray.data.Dataset
            The Ray Dataset to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Ray Dataset.

        Notes
        -----
        Ray Dataset can only be converted to a Modin Dataframe if Modin uses a Ray engine.
        If another engine is used, the runtime exception will be raised.
        """
        raise RuntimeError(
            "Modin Dataframe can only be converted to a Ray Dataset if Modin uses a Ray engine."
        )

    @classmethod
    def from_dask(cls, dask_obj):
        """
        Create a Modin `query_compiler` from a Dask DataFrame.

        Parameters
        ----------
        dask_obj : dask.dataframe.DataFrame
            The Dask DataFrame to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Dask DataFrame.

        Notes
        -----
        Dask DataFrame can only be converted to a Modin DataFrame if Modin uses a Dask engine.
        If another engine is used, the runtime exception will be raised.
        """
        raise RuntimeError(
            "Modin DataFrame can only be converted to a Dask DataFrame if Modin uses a Dask engine."
        )

    @classmethod
    def from_map(cls, func, iterable, *args, **kwargs):
        """
        Create a Modin `query_compiler` from a map function.

        This method will construct a Modin `query_compiler` split by row partitions.
        The number of row partitions matches the number of elements in the iterable object.

        Parameters
        ----------
        func : callable
            Function to map across the iterable object.
        iterable : Iterable
            An iterable object.
        *args : tuple
            Positional arguments to pass in `func`.
        **kwargs : dict
            Keyword arguments to pass in `func`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data returned by map function.
        """
        raise RuntimeError(
            "Modin DataFrame can only be created if Modin uses Ray, Dask or MPI engine."
        )

    @classmethod
    @_inherit_docstrings(pandas.read_parquet, apilink="pandas.read_parquet")
    @doc(
        _doc_default_io_method,
        summary="Load a parquet object from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def read_parquet(cls, **kwargs):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_parquet`")
        return cls.from_pandas(pandas.read_parquet(**kwargs))

    @classmethod
    @_inherit_docstrings(pandas.read_csv, apilink="pandas.read_csv")
    @doc(
        _doc_default_io_method,
        summary="Read a comma-separated values (CSV) file into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def read_csv(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_csv`")
        pd_obj = pandas.read_csv(filepath_or_buffer, **kwargs)
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a Modin DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kw: cls.from_pandas(pd_read(*args, **kw))
        return pd_obj

    @classmethod
    @_inherit_docstrings(pandas.read_json, apilink="pandas.read_json")
    @doc(
        _doc_default_io_method,
        summary="Convert a JSON string to query compiler",
        returns=_doc_returns_qc,
    )
    def read_json(
        cls,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_json`")
        return cls.from_pandas(pandas.read_json(**kwargs))

    @classmethod
    @_inherit_docstrings(pandas.read_gbq, apilink="pandas.read_gbq")
    @doc(
        _doc_default_io_method,
        summary="Load data from Google BigQuery into query compiler",
        returns=_doc_returns_qc,
    )
    def read_gbq(
        cls,
        query: str,
        project_id=None,
        index_col=None,
        col_order=None,
        reauth=False,
        auth_local_webserver=False,
        dialect=None,
        location=None,
        configuration=None,
        credentials=None,
        use_bqstorage_api=None,
        private_key=None,
        verbose=None,
        progress_bar_type=None,
        max_results=None,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_gbq`")
        return cls.from_pandas(
            pandas.read_gbq(
                query,
                project_id=project_id,
                index_col=index_col,
                col_order=col_order,
                reauth=reauth,
                auth_local_webserver=auth_local_webserver,
                dialect=dialect,
                location=location,
                configuration=configuration,
                credentials=credentials,
                use_bqstorage_api=use_bqstorage_api,
                progress_bar_type=progress_bar_type,
                max_results=max_results,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_html, apilink="pandas.read_html")
    @doc(
        _doc_default_io_method,
        summary="Read HTML tables into query compiler",
        returns=_doc_returns_qc,
    )
    def read_html(
        cls,
        io,
        *,
        match=".+",
        flavor=None,
        header=None,
        index_col=None,
        skiprows=None,
        attrs=None,
        parse_dates=False,
        thousands=",",
        encoding=None,
        decimal=".",
        converters=None,
        na_values=None,
        keep_default_na=True,
        displayed_only=True,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_html`")
        result = pandas.read_html(
            io=io,
            match=match,
            flavor=flavor,
            header=header,
            index_col=index_col,
            skiprows=skiprows,
            attrs=attrs,
            parse_dates=parse_dates,
            thousands=thousands,
            encoding=encoding,
            decimal=decimal,
            converters=converters,
            na_values=na_values,
            keep_default_na=keep_default_na,
            displayed_only=displayed_only,
            **kwargs,
        )
        return (cls.from_pandas(df) for df in result)

    @classmethod
    @_inherit_docstrings(pandas.read_clipboard, apilink="pandas.read_clipboard")
    @doc(
        _doc_default_io_method,
        summary="Read text from clipboard into query compiler",
        returns=_doc_returns_qc,
    )
    def read_clipboard(cls, sep=r"\s+", **kwargs):  # pragma: no cover # noqa: PR01
        cls._maybe_warn_on_default(message="`read_clipboard`")
        return cls.from_pandas(pandas.read_clipboard(sep=sep, **kwargs))

    @classmethod
    @_inherit_docstrings(pandas.read_excel, apilink="pandas.read_excel")
    @doc(
        _doc_default_io_method,
        summary="Read an Excel file into query compiler",
        returns="""BaseQueryCompiler or dict :
    QueryCompiler or dict with read data.""",
    )
    def read_excel(cls, **kwargs):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_excel`")
        if isinstance(kwargs["io"], ExcelFile):
            # otherwise, Modin objects may be passed to the pandas context, resulting
            # in undefined behavior
            # for example in the case: pd.read_excel(pd.ExcelFile), since reading from
            # pd.ExcelFile in `read_excel` isn't supported
            kwargs["io"]._set_pandas_mode()
        intermediate = pandas.read_excel(**kwargs)
        if isinstance(intermediate, dict):
            parsed = type(intermediate)()
            for key in intermediate.keys():
                parsed[key] = cls.from_pandas(intermediate.get(key))
            return parsed
        else:
            return cls.from_pandas(intermediate)

    @classmethod
    @_inherit_docstrings(pandas.read_hdf, apilink="pandas.read_hdf")
    @doc(
        _doc_default_io_method,
        summary="Read data from hdf store into query compiler",
        returns=_doc_returns_qc,
    )
    def read_hdf(
        cls,
        path_or_buf,
        key=None,
        mode: str = "r",
        errors: str = "strict",
        where=None,
        start=None,
        stop=None,
        columns=None,
        iterator=False,
        chunksize=None,
        **kwargs,
    ):  # noqa: PR01
        from modin.pandas.io import HDFStore

        cls._maybe_warn_on_default(message="`read_hdf`")
        modin_store = isinstance(path_or_buf, HDFStore)
        if modin_store:
            path_or_buf._return_modin_dataframe = False
        df = pandas.read_hdf(
            path_or_buf,
            key=key,
            mode=mode,
            columns=columns,
            errors=errors,
            where=where,
            start=start,
            stop=stop,
            iterator=iterator,
            chunksize=chunksize,
            **kwargs,
        )
        if modin_store:
            path_or_buf._return_modin_dataframe = True

        return cls.from_pandas(df)

    @classmethod
    @_inherit_docstrings(pandas.read_feather, apilink="pandas.read_feather")
    @doc(
        _doc_default_io_method,
        summary="Load a feather-format object from the file path into query compiler",
        returns=_doc_returns_qc,
    )
    def read_feather(
        cls,
        path,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_feather`")
        return cls.from_pandas(
            pandas.read_feather(
                path,
                **kwargs,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_stata, apilink="pandas.read_stata")
    @doc(
        _doc_default_io_method,
        summary="Read Stata file into query compiler",
        returns=_doc_returns_qc,
    )
    def read_stata(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_stata`")
        return cls.from_pandas(pandas.read_stata(filepath_or_buffer, **kwargs))

    @classmethod
    @_inherit_docstrings(pandas.read_sas, apilink="pandas.read_sas")
    @doc(
        _doc_default_io_method,
        summary="Read SAS files stored as either XPORT or SAS7BDAT format files\ninto query compiler",
        returns=_doc_returns_qc,
    )
    def read_sas(
        cls,
        filepath_or_buffer,
        *,
        format=None,
        index=None,
        encoding=None,
        chunksize=None,
        iterator=False,
        **kwargs,
    ):  # pragma: no cover # noqa: PR01
        cls._maybe_warn_on_default(message="`read_sas`")
        return cls.from_pandas(
            pandas.read_sas(
                filepath_or_buffer,
                format=format,
                index=index,
                encoding=encoding,
                chunksize=chunksize,
                iterator=iterator,
                **kwargs,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_pickle, apilink="pandas.read_pickle")
    @doc(
        _doc_default_io_method,
        summary="Load pickled pandas object (or any object) from file into query compiler",
        returns=_doc_returns_qc,
    )
    def read_pickle(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_pickle`")
        return cls.from_pandas(
            pandas.read_pickle(
                filepath_or_buffer,
                **kwargs,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_sql, apilink="pandas.read_sql")
    @doc(
        _doc_default_io_method,
        summary="Read SQL query or database table into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        columns=None,
        chunksize=None,
        dtype_backend=no_default,
        dtype=None,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_sql`")
        if isinstance(con, ModinDatabaseConnection):
            con = con.get_connection()
        result = pandas.read_sql(
            sql,
            con,
            index_col=index_col,
            coerce_float=coerce_float,
            params=params,
            parse_dates=parse_dates,
            columns=columns,
            chunksize=chunksize,
            dtype_backend=dtype_backend,
            dtype=dtype,
        )

        if isinstance(result, (pandas.DataFrame, pandas.Series)):
            return cls.from_pandas(result)
        return (cls.from_pandas(df) for df in result)

    @classmethod
    @_inherit_docstrings(pandas.read_fwf, apilink="pandas.read_fwf")
    @doc(
        _doc_default_io_method,
        summary="Read a table of fixed-width formatted lines into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def read_fwf(
        cls,
        filepath_or_buffer,
        *,
        colspecs="infer",
        widths=None,
        infer_nrows=100,
        dtype_backend=no_default,
        iterator=False,
        chunksize=None,
        **kwds,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_fwf`")
        pd_obj = pandas.read_fwf(
            filepath_or_buffer,
            colspecs=colspecs,
            widths=widths,
            infer_nrows=infer_nrows,
            dtype_backend=dtype_backend,
            iterator=iterator,
            chunksize=chunksize,
            **kwds,
        )
        if isinstance(pd_obj, pandas.DataFrame):
            return cls.from_pandas(pd_obj)
        if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
            # Overwriting the read method should return a Modin DataFrame for calls
            # to __next__ and get_chunk
            pd_read = pd_obj.read
            pd_obj.read = lambda *args, **kwargs: cls.from_pandas(
                pd_read(*args, **kwargs)
            )
        return pd_obj

    @classmethod
    @_inherit_docstrings(pandas.read_sql_table, apilink="pandas.read_sql_table")
    @doc(
        _doc_default_io_method,
        summary="Read SQL database table into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql_table(
        cls,
        table_name,
        con,
        schema=None,
        index_col=None,
        coerce_float=True,
        parse_dates=None,
        columns=None,
        chunksize=None,
        dtype_backend=no_default,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_sql_table`")
        return cls.from_pandas(
            pandas.read_sql_table(
                table_name,
                con,
                schema=schema,
                index_col=index_col,
                coerce_float=coerce_float,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
                dtype_backend=dtype_backend,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_sql_query, apilink="pandas.read_sql_query")
    @doc(
        _doc_default_io_method,
        summary="Read SQL query into query compiler",
        returns=_doc_returns_qc,
    )
    def read_sql_query(
        cls,
        sql,
        con,
        **kwargs,
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_sql_query`")
        return cls.from_pandas(
            pandas.read_sql_query(
                sql,
                con,
                **kwargs,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_spss, apilink="pandas.read_spss")
    @doc(
        _doc_default_io_method,
        summary="Load an SPSS file from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def read_spss(
        cls, path, usecols, convert_categoricals, dtype_backend
    ):  # noqa: PR01
        cls._maybe_warn_on_default(message="`read_spss`")
        return cls.from_pandas(
            pandas.read_spss(
                path,
                usecols=usecols,
                convert_categoricals=convert_categoricals,
                dtype_backend=dtype_backend,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.DataFrame.to_sql, apilink="pandas.DataFrame.to_sql")
    def to_sql(
        cls,
        qc,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):  # noqa: PR01
        """
        Write records stored in a DataFrame to a SQL database using pandas.

        For parameters description please refer to pandas API.
        """
        cls._maybe_warn_on_default(message="`to_sql`")
        df = qc.to_pandas()
        df.to_sql(
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    @classmethod
    @_inherit_docstrings(
        pandas.DataFrame.to_pickle, apilink="pandas.DataFrame.to_pickle"
    )
    def to_pickle(
        cls,
        obj: Any,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01, D200
        """
        Pickle (serialize) object to file.
        """
        cls._maybe_warn_on_default(message="`to_pickle`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return pandas.to_pickle(
            obj,
            filepath_or_buffer=filepath_or_buffer,
            **kwargs,
        )

    @classmethod
    @_inherit_docstrings(pandas.DataFrame.to_csv, apilink="pandas.DataFrame.to_csv")
    def to_csv(cls, obj, **kwargs):  # noqa: PR01
        """
        Write object to a comma-separated values (CSV) file using pandas.

        For parameters description please refer to pandas API.
        """
        cls._maybe_warn_on_default(message="`to_csv`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_csv(**kwargs)

    @classmethod
    @_inherit_docstrings(pandas.DataFrame.to_json, apilink="pandas.DataFrame.to_json")
    def to_json(cls, obj, path, **kwargs):  # noqa: PR01
        """
        Convert the object to a JSON string.

        For parameters description please refer to pandas API.
        """
        cls._maybe_warn_on_default(message="`to_json`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_json(path, **kwargs)

    @classmethod
    @_inherit_docstrings(pandas.DataFrame.to_xml, apilink="pandas.DataFrame.to_xml")
    def to_xml(cls, obj, path_or_buffer, **kwargs):  # noqa: PR01
        """
        Convert the object to a XML string.

        For parameters description please refer to pandas API.
        """
        cls._maybe_warn_on_default(message="`to_xml`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_xml(path_or_buffer, **kwargs)

    @classmethod
    @_inherit_docstrings(
        pandas.DataFrame.to_parquet, apilink="pandas.DataFrame.to_parquet"
    )
    def to_parquet(cls, obj, path, **kwargs):  # noqa: PR01
        """
        Write object to the binary parquet format using pandas.

        For parameters description please refer to pandas API.
        """
        cls._maybe_warn_on_default(message="`to_parquet`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_parquet(path, **kwargs)

    @classmethod
    def to_ray(cls, modin_obj):
        """
        Convert a Modin DataFrame/Series to a Ray Dataset.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        ray.data.Dataset
            Converted object with type depending on input.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Ray Dataset if Modin uses a Ray engine.
        If another engine is used, the runtime exception will be raised.
        """
        raise RuntimeError(
            "Modin Dataframe can only be converted to a Ray Dataset if Modin uses a Ray engine."
        )

    @classmethod
    def to_dask(cls, modin_obj):
        """
        Convert a Modin DataFrame to a Dask DataFrame.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            Converted object with type depending on input.

        Notes
        -----
        Modin DataFrame/Series can only be converted to a Dask DataFrame/Series if Modin uses a Dask engine.
        If another engine is used, the runtime exception will be raised.
        """
        raise RuntimeError(
            "Modin DataFrame can only be converted to a Dask DataFrame if Modin uses a Dask engine."
        )
