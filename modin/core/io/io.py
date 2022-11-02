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

from collections import OrderedDict
from typing import Any

import pandas
from pandas.util._decorators import doc

from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import _inherit_docstrings

from modin._compat.core.base_io import (
    BaseIOCompat,
    _doc_default_io_method,
    _doc_returns_qc,
    _doc_returns_qc_or_parser,
)
from modin._compat.core.pandas_common import pandas_to_csv


class BaseIO(BaseIOCompat):
    """Class for basic utils and default implementation of IO functions."""

    query_compiler_cls: BaseQueryCompiler = None
    frame_cls = None

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
    def from_dataframe(cls, df):
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
        return cls.query_compiler_cls.from_dataframe(df, cls.frame_cls)

    @classmethod
    @_inherit_docstrings(pandas.read_parquet, apilink="pandas.read_parquet")
    @doc(
        _doc_default_io_method,
        summary="Load a parquet object from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def _read_parquet(cls, **kwargs):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_parquet`")
        return cls.from_pandas(
            pandas.read_parquet(
                **kwargs,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_csv, apilink="pandas.read_csv")
    @doc(
        _doc_default_io_method,
        summary="Read a comma-separated values (CSV) file into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def _read_csv(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_csv`")
        return cls._read(filepath_or_buffer=filepath_or_buffer, **kwargs)

    @classmethod
    def _read(cls, **kwargs):
        """
        Read csv file into query compiler.

        Parameters
        ----------
        **kwargs : dict
            `read_csv` function kwargs including `filepath_or_buffer` parameter.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler with read data.
        """
        pd_obj = pandas.read_csv(**kwargs)
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
    @_inherit_docstrings(pandas.read_json, apilink="pandas.read_json")
    @doc(
        _doc_default_io_method,
        summary="Convert a JSON string to query compiler",
        returns=_doc_returns_qc,
    )
    def _read_json(
        cls,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_json`")
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
        ErrorMessage.default_to_pandas("`read_gbq`")
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
        ErrorMessage.default_to_pandas("`read_html`")
        return cls.from_pandas(
            pandas.read_html(
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
            )[0]
        )

    @classmethod
    @_inherit_docstrings(pandas.read_clipboard, apilink="pandas.read_clipboard")
    @doc(
        _doc_default_io_method,
        summary="Read text from clipboard into query compiler",
        returns=_doc_returns_qc,
    )
    def read_clipboard(cls, sep=r"\s+", **kwargs):  # pragma: no cover # noqa: PR01
        ErrorMessage.default_to_pandas("`read_clipboard`")
        return cls.from_pandas(pandas.read_clipboard(sep=sep, **kwargs))

    @classmethod
    @_inherit_docstrings(pandas.read_excel, apilink="pandas.read_excel")
    @doc(
        _doc_default_io_method,
        summary="Read an Excel file into query compiler",
        returns="""BaseQueryCompiler or dict/OrderedDict :
    QueryCompiler or OrderedDict/dict with read data.""",
    )
    def read_excel(
        cls,
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
        keep_default_na=True,
        verbose=False,
        parse_dates=False,
        date_parser=None,
        thousands=None,
        comment=None,
        skip_footer=0,
        skipfooter=0,
        convert_float=True,
        mangle_dupe_cols=True,
        na_filter=True,
        **kwds,
    ):  # noqa: PR01
        if skip_footer != 0:
            skipfooter = skip_footer
        ErrorMessage.default_to_pandas("`read_excel`")
        intermediate = pandas.read_excel(
            io,
            sheet_name=sheet_name,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            squeeze=squeeze,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            verbose=verbose,
            parse_dates=parse_dates,
            date_parser=date_parser,
            thousands=thousands,
            comment=comment,
            skipfooter=skipfooter,
            convert_float=convert_float,
            mangle_dupe_cols=mangle_dupe_cols,
            na_filter=na_filter,
            **kwds,
        )
        if isinstance(intermediate, (OrderedDict, dict)):
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

        ErrorMessage.default_to_pandas("`read_hdf`")
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
    def _read_feather(
        cls,
        path,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_feather`")
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
    def _read_stata(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_stata`")
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
        format=None,
        index=None,
        encoding=None,
        chunksize=None,
        iterator=False,
        **kwargs,
    ):  # pragma: no cover # noqa: PR01
        ErrorMessage.default_to_pandas("`read_sas`")
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
    def _read_pickle(
        cls,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_pickle`")
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
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_sql`")
        if isinstance(con, ModinDatabaseConnection):
            con = con.get_connection()
        return cls.from_pandas(
            pandas.read_sql(
                sql,
                con,
                index_col=index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_fwf, apilink="pandas.read_fwf")
    @doc(
        _doc_default_io_method,
        summary="Read a table of fixed-width formatted lines into query compiler",
        returns=_doc_returns_qc_or_parser,
    )
    def read_fwf(
        cls, filepath_or_buffer, colspecs="infer", widths=None, infer_nrows=100, **kwds
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_fwf`")
        pd_obj = pandas.read_fwf(
            filepath_or_buffer,
            colspecs=colspecs,
            widths=widths,
            infer_nrows=infer_nrows,
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
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_sql_table`")
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
            )
        )

    @classmethod
    @_inherit_docstrings(pandas.read_sql_query, apilink="pandas.read_sql_query")
    @doc(
        _doc_default_io_method,
        summary="Read SQL query into query compiler",
        returns=_doc_returns_qc,
    )
    def _read_sql_query(
        cls,
        sql,
        con,
        **kwargs,
    ):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_sql_query`")
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
    def read_spss(cls, path, usecols, convert_categoricals):  # noqa: PR01
        ErrorMessage.default_to_pandas("`read_spss`")
        return cls.from_pandas(pandas.read_spss(path, usecols, convert_categoricals))

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
        ErrorMessage.default_to_pandas("`to_sql`")
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
    def _to_pickle(
        cls,
        obj: Any,
        filepath_or_buffer,
        **kwargs,
    ):  # noqa: PR01, D200
        """
        Pickle (serialize) object to file.
        """
        ErrorMessage.default_to_pandas("`to_pickle`")
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
        ErrorMessage.default_to_pandas("`to_csv`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return pandas_to_csv(obj, **kwargs)

    @classmethod
    @_inherit_docstrings(
        pandas.DataFrame.to_parquet, apilink="pandas.DataFrame.to_parquet"
    )
    def to_parquet(cls, obj, **kwargs):  # noqa: PR01
        """
        Write object to the binary parquet format using pandas.

        For parameters description please refer to pandas API.
        """
        ErrorMessage.default_to_pandas("`to_parquet`")
        if isinstance(obj, BaseQueryCompiler):
            obj = obj.to_pandas()

        return obj.to_parquet(**kwargs)
