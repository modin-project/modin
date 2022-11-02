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
Module houses ``LatestBaseIOCompat`` class.

``LatestBaseIOCompat`` is base mixin for IO classes relying on modern pandas.
"""

from typing import Optional, Any
import pickle

import pandas
import pandas._libs.lib as lib
from pandas.util._decorators import doc

from modin.utils import _inherit_docstrings
from ..doc_common import (
    _doc_default_io_method,
    _doc_returns_qc,
    _doc_returns_qc_or_parser,
)


class LatestBaseIOCompat:
    """Class for basic utils and default implementation of IO functions for older pandas."""

    @classmethod
    @_inherit_docstrings(pandas.read_parquet, apilink="pandas.read_parquet")
    @doc(
        _doc_default_io_method,
        summary="Load a parquet object from the file path, returning a query compiler",
        returns=_doc_returns_qc,
    )
    def read_parquet(
        cls, path, engine, columns, storage_options, use_nullable_dtypes, **kwargs
    ):  # noqa: PR01
        return cls._read_parquet(
            path=path,
            engine=engine,
            columns=columns,
            storage_options=storage_options,
            use_nullable_dtypes=use_nullable_dtypes,
            **kwargs
        )

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
        sep=lib.no_default,
        delimiter=None,
        header="infer",
        names=lib.no_default,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=lib.no_default,
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
        cache_dates=True,
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
        encoding_errors="strict",
        dialect=None,
        error_bad_lines=None,
        warn_bad_lines=None,
        on_bad_lines=None,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
        storage_options=None,
    ):  # noqa: PR01
        return cls._read_csv(
            filepath_or_buffer=filepath_or_buffer,
            sep=sep,
            delimiter=delimiter,
            header=header,
            names=names,
            index_col=index_col,
            usecols=usecols,
            squeeze=squeeze,
            prefix=prefix,
            dtype=dtype,
            engine=engine,
            converters=converters,
            true_values=true_values,
            false_values=false_values,
            skipinitialspace=skipinitialspace,
            skiprows=skiprows,
            nrows=nrows,
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            skip_blank_lines=skip_blank_lines,
            parse_dates=parse_dates,
            infer_datetime_format=infer_datetime_format,
            keep_date_col=keep_date_col,
            date_parser=date_parser,
            dayfirst=dayfirst,
            cache_dates=cache_dates,
            iterator=iterator,
            chunksize=chunksize,
            compression=compression,
            thousands=thousands,
            decimal=decimal,
            lineterminator=lineterminator,
            quotechar=quotechar,
            quoting=quoting,
            escapechar=escapechar,
            comment=comment,
            encoding=encoding,
            encoding_errors=encoding_errors,
            dialect=dialect,
            error_bad_lines=error_bad_lines,
            warn_bad_lines=warn_bad_lines,
            on_bad_lines=on_bad_lines,
            skipfooter=skipfooter,
            doublequote=doublequote,
            delim_whitespace=delim_whitespace,
            low_memory=low_memory,
            memory_map=memory_map,
            float_precision=float_precision,
            storage_options=storage_options,
        )

    @classmethod
    @_inherit_docstrings(pandas.read_json, apilink="pandas.read_json")
    @doc(
        _doc_default_io_method,
        summary="Convert a JSON string to query compiler",
        returns=_doc_returns_qc,
    )
    def read_json(
        cls,
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
        encoding_errors="strict",
        lines=False,
        chunksize=None,
        compression="infer",
        nrows: Optional[int] = None,
        storage_options=None,
    ):  # noqa: PR01
        return cls._read_json(
            path_or_buf=path_or_buf,
            orient=orient,
            typ=typ,
            dtype=dtype,
            convert_axes=convert_axes,
            convert_dates=convert_dates,
            keep_default_dates=keep_default_dates,
            numpy=numpy,
            precise_float=precise_float,
            date_unit=date_unit,
            encoding=encoding,
            encoding_errors=encoding_errors,
            lines=lines,
            chunksize=chunksize,
            compression=compression,
            nrows=nrows,
            storage_options=storage_options,
        )

    @classmethod
    @_inherit_docstrings(pandas.read_feather, apilink="pandas.read_feather")
    @doc(
        _doc_default_io_method,
        summary="Load a feather-format object from the file path into query compiler",
        returns=_doc_returns_qc,
    )
    def read_feather(
        cls, path, columns=None, use_threads=True, storage_options=None
    ):  # noqa: PR01
        return cls._read_feather(
            path=path,
            columns=columns,
            use_threads=use_threads,
            storage_options=storage_options,
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
        convert_dates=True,
        convert_categoricals=True,
        index_col=None,
        convert_missing=False,
        preserve_dtypes=True,
        columns=None,
        order_categoricals=True,
        chunksize=None,
        iterator=False,
        compression="infer",
        storage_options=None,
    ):  # noqa: PR01
        return cls._read_stata(
            filepath_or_buffer=filepath_or_buffer,
            convert_dates=convert_dates,
            convert_categoricals=convert_categoricals,
            index_col=index_col,
            convert_missing=convert_missing,
            preserve_dtypes=preserve_dtypes,
            columns=columns,
            order_categoricals=order_categoricals,
            chunksize=chunksize,
            iterator=iterator,
            compression=compression,
            storage_options=storage_options,
        )

    @classmethod
    @_inherit_docstrings(pandas.read_pickle, apilink="pandas.read_pickle")
    @doc(
        _doc_default_io_method,
        summary="Load pickled pandas object (or any object) from file into query compiler",
        returns=_doc_returns_qc,
    )
    def read_pickle(
        cls, filepath_or_buffer, compression="infer", storage_options=None
    ):  # noqa: PR01
        return cls._read_pickle(
            filepath_or_buffer=filepath_or_buffer,
            compression=compression,
            storage_options=storage_options,
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
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        chunksize=None,
        dtype=None,
    ):  # noqa: PR01
        return cls._read_sql_query(
            sql=sql,
            con=con,
            index_col=index_col,
            coerce_float=coerce_float,
            params=params,
            parse_dates=parse_dates,
            chunksize=chunksize,
            dtype=dtype,
        )

    @classmethod
    @_inherit_docstrings(
        pandas.DataFrame.to_pickle, apilink="pandas.DataFrame.to_pickle"
    )
    def to_pickle(
        cls,
        obj: Any,
        filepath_or_buffer,
        compression="infer",
        protocol: int = pickle.HIGHEST_PROTOCOL,
        storage_options=None,
    ):  # noqa: PR01, D200
        return cls._to_pickle(
            obj,
            filepath_or_buffer,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )
