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
Module houses ``OmnisciOnNativeIO`` class.

``OmnisciOnNativeIO`` is used for storing IO functions implementations with OmniSci storage format and Native engine.
"""

from csv import Dialect
from typing import Union, Sequence, Callable, Dict, Tuple
import inspect
import os

from modin.experimental.core.storage_formats.omnisci.query_compiler import (
    DFAlgQueryCompiler,
)
from modin.core.io import BaseIO
from modin.experimental.core.execution.native.implementations.omnisci_on_native.dataframe.dataframe import (
    OmnisciOnNativeDataframe,
)
from modin.error_message import ErrorMessage
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher

from pyarrow.csv import read_csv, ParseOptions, ConvertOptions, ReadOptions
import pyarrow as pa

import pandas
import pandas._libs.lib as lib
from pandas.io.common import is_url

ReadCsvKwargsType = Dict[
    str,
    Union[
        str,
        int,
        bool,
        dict,
        object,
        Sequence,
        Callable,
        Dialect,
        None,
    ],
]


class ArrowEngineException(Exception):
    """Exception raised in case of Arrow engine-specific incompatibilities are found."""


class OmnisciOnNativeIO(BaseIO, TextFileDispatcher):
    """Class contains IO functions implementations with OmniSci storage format and Native engine."""

    frame_cls = OmnisciOnNativeDataframe
    query_compiler_cls = DFAlgQueryCompiler

    arg_keys = [
        "filepath_or_buffer",
        "sep",
        "delimiter",
        "header",
        "names",
        "index_col",
        "usecols",
        "squeeze",
        "prefix",
        "mangle_dupe_cols",
        "dtype",
        "engine",
        "converters",
        "true_values",
        "false_values",
        "skipinitialspace",
        "skiprows",
        "nrows",
        "na_values",
        "keep_default_na",
        "na_filter",
        "verbose",
        "skip_blank_lines",
        "parse_dates",
        "infer_datetime_format",
        "keep_date_col",
        "date_parser",
        "dayfirst",
        "cache_dates",
        "iterator",
        "chunksize",
        "compression",
        "thousands",
        "decimal",
        "lineterminator",
        "quotechar",
        "quoting",
        "escapechar",
        "comment",
        "encoding",
        "encoding_errors",
        "dialect",
        "error_bad_lines",
        "warn_bad_lines",
        "on_bad_lines",
        "skipfooter",
        "doublequote",
        "delim_whitespace",
        "low_memory",
        "memory_map",
        "float_precision",
        "storage_options",
    ]

    unsupported_args = [
        "decimal",
        "thousands",
        "index_col",
        "prefix",
        "converters",
        "skipfooter",
        "nrows",
        "skipinitialspace",
        "squeeze",
        "mangle_dupe_cols",
        "na_values",
        "keep_default_na",
        "na_filter",
        "verbose",
        "infer_datetime_format",
        "keep_date_col",
        "date_parser",
        "dayfirst",
        "cache_dates",
        "iterator",
        "chunksize",
        "encoding",
        "encoding_errors",
        "lineterminator",
        "dialect",
        "quoting",
        "comment",
        "warn_bad_lines",
        "error_bad_lines",
        "on_bad_lines",
        "low_memory",
        "memory_map",
        "float_precision",
        "storage_options",
    ]

    @classmethod
    def read_csv(
        cls,
        filepath_or_buffer,
        sep=",",
        delimiter=None,
        header="infer",
        names=lib.no_default,
        index_col=None,
        usecols=None,
        squeeze=False,
        prefix=lib.no_default,
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
        cache_dates=True,
        iterator=False,
        chunksize=None,
        compression="infer",
        thousands=None,
        decimal=".",
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
        """
        Read data from `filepath_or_buffer` according to the passed `kwargs` parameters.

        For parameters description please refer to pandas API.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        Reading performed by using of `pyarrow.read_csv` function.
        """
        items = locals().copy()
        mykwargs = {k: items[k] for k in items if k in cls.arg_keys}
        eng = str(engine).lower().strip()
        try:
            if eng in ["pandas", "c"]:
                return cls._read(**mykwargs)

            cls._validate_read_csv_kwargs(mykwargs)
            use_modin_impl, error_message = cls._read_csv_check_support(
                mykwargs,
            )
            if not use_modin_impl:
                raise ArrowEngineException(error_message)
            if isinstance(dtype, dict):
                column_types = {c: cls._dtype_to_arrow(t) for c, t in dtype.items()}
            else:
                column_types = cls._dtype_to_arrow(dtype)

            if (type(parse_dates) is list) and type(column_types) is dict:
                for c in parse_dates:
                    column_types[c] = pa.timestamp("s")

            if names not in [lib.no_default, None] and header == 0:
                skiprows = skiprows + 1 if skiprows is not None else 1

            if delimiter is None and sep is not lib.no_default:
                delimiter = sep

            usecols_md = cls._prepare_pyarrow_usecols(mykwargs)

            po = ParseOptions(
                delimiter="\\s+" if delim_whitespace else delimiter,
                quote_char=quotechar,
                double_quote=doublequote,
                escape_char=escapechar,
                newlines_in_values=False,
                ignore_empty_lines=skip_blank_lines,
            )
            co = ConvertOptions(
                check_utf8=None,
                column_types=column_types,
                null_values=None,
                # we need to add default true/false_values like Pandas does
                true_values=true_values + ["TRUE", "True", "true"]
                if true_values is not None
                else true_values,
                false_values=false_values + ["False", "FALSE", "false"]
                if false_values is not None
                else false_values,
                # timestamp fields should be handled as strings if parse_dates
                # didn't passed explicitly as an array or a dict
                timestamp_parsers=[""] if isinstance(parse_dates, bool) else None,
                strings_can_be_null=None,
                include_columns=usecols_md,
                include_missing_columns=None,
                auto_dict_encode=None,
                auto_dict_max_cardinality=None,
            )
            ro = ReadOptions(
                use_threads=True,
                block_size=None,
                skip_rows=skiprows,
                column_names=names if names is not lib.no_default else None,
                autogenerate_column_names=None,
            )

            at = read_csv(
                filepath_or_buffer,
                read_options=ro,
                parse_options=po,
                convert_options=co,
            )

            return cls.from_arrow(at)
        except (
            pa.ArrowNotImplementedError,
            pa.ArrowInvalid,
            NotImplementedError,
            ArrowEngineException,
        ):
            if eng in ["arrow"]:
                raise

            ErrorMessage.default_to_pandas("`read_csv`")
            return cls._read(**mykwargs)

    @classmethod
    def _dtype_to_arrow(cls, dtype):
        """
        Convert `pandas.read_csv` `dtype` parameter into PyArrow compatible type.

        Parameters
        ----------
        dtype : str, pandas extension or NumPy dtype
            Data type for data or columns, `pandas.read_csv` `dtype` parameter.

        Returns
        -------
        pa.DataType or pa.DictionaryType
            PyArrow compatible type.
        """
        if dtype is None:
            return None
        tname = dtype if isinstance(dtype, str) else dtype.name
        if tname == "category":
            return pa.dictionary(index_type=pa.int32(), value_type=pa.string())
        elif tname == "string":
            return pa.string()
        else:
            return pa.from_numpy_dtype(tname)

    @classmethod
    def _prepare_pyarrow_usecols(cls, read_csv_kwargs):
        """
        Define `usecols` parameter in the way PyArrow can process it.

        Parameters
        ----------
        read_csv_kwargs : dict
            Parameters of read_csv.

        Returns
        -------
        list
            Redefined `usecols` parameter.
        """
        usecols = read_csv_kwargs.get("usecols", None)
        engine = read_csv_kwargs.get("engine", None)
        usecols_md, usecols_names_dtypes = cls._validate_usecols_arg(usecols)
        if usecols_md:
            empty_pd_df = pandas.read_csv(
                **dict(
                    read_csv_kwargs,
                    nrows=0,
                    skipfooter=0,
                    usecols=None,
                    engine=None if engine == "arrow" else engine,
                )
            )
            column_names = empty_pd_df.columns
            if usecols_names_dtypes == "string":
                if usecols_md.issubset(set(column_names)):
                    # columns should be sorted because pandas doesn't preserve columns order
                    usecols_md = [
                        col_name for col_name in column_names if col_name in usecols_md
                    ]
                else:
                    raise NotImplementedError(
                        "values passed in the `usecols` parameter don't match columns names"
                    )
            elif usecols_names_dtypes == "integer":
                # columns should be sorted because pandas doesn't preserve columns order
                usecols_md = sorted(usecols_md)
                if len(column_names) < usecols_md[-1]:
                    raise NotImplementedError(
                        "max usecols value is higher than the number of columns"
                    )
                usecols_md = [column_names[i] for i in usecols_md]
            elif callable(usecols_md):
                usecols_md = [
                    col_name for col_name in column_names if usecols_md(col_name)
                ]
            else:
                raise NotImplementedError("unsupported `usecols` parameter")

        return usecols_md

    read_csv_unsup_defaults = {}
    for k, v in inspect.signature(read_csv.__func__).parameters.items():
        if v.default is not inspect.Parameter.empty and k in unsupported_args:
            read_csv_unsup_defaults[k] = v.default

    @classmethod
    def _read_csv_check_support(
        cls,
        read_csv_kwargs: ReadCsvKwargsType,
    ) -> Tuple[bool, str]:
        """
        Check if passed parameters are supported by current ``modin.pandas.read_csv`` implementation.

        Parameters
        ----------
        read_csv_kwargs : dict
                Parameters of read_csv function.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        str
            Error message that should be raised if user explicitly set `engine="arrow"`.
        """
        filepath_or_buffer = read_csv_kwargs.get("filepath_or_buffer", None)
        header = read_csv_kwargs.get("header", "infer")
        names = read_csv_kwargs.get("names", None)
        engine = read_csv_kwargs.get("engine", None)
        skiprows = read_csv_kwargs.get("skiprows", None)
        delimiter = read_csv_kwargs.get("delimiter", None)
        parse_dates = read_csv_kwargs.get("parse_dates", False)

        if read_csv_kwargs.get("compression", "infer") != "infer":
            return (
                False,
                "read_csv with 'arrow' engine doesn't support explicit compression parameter, compression"
                + " must be inferred automatically (supported compression types are gzip and bz2)",
            )

        if isinstance(filepath_or_buffer, str):
            if not os.path.exists(filepath_or_buffer):
                if cls.file_exists(filepath_or_buffer) or is_url(filepath_or_buffer):
                    return (
                        False,
                        "read_csv with 'arrow' engine supports only local files",
                    )
                else:
                    raise FileNotFoundError("No such file or directory")
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            if hasattr(filepath_or_buffer, "read"):
                return (
                    False,
                    "read_csv with 'arrow' engine doesn't support file-like objects",
                )
            else:
                raise ValueError(
                    f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
                )

        for arg, def_value in cls.read_csv_unsup_defaults.items():
            if read_csv_kwargs[arg] != def_value:
                return (
                    False,
                    f"read_csv with 'arrow' engine doesn't support {arg} parameter",
                )
        if delimiter is not None and read_csv_kwargs.get("delim_whitespace", False):
            raise ValueError(
                "Specified a delimiter with both sep and delim_whitespace=True; you can only specify one."
            )

        parse_dates_unsupported = isinstance(parse_dates, dict) or (
            isinstance(parse_dates, list)
            and any(not isinstance(date, str) for date in parse_dates)
        )
        if parse_dates_unsupported:
            return (
                False,
                (
                    "read_csv with 'arrow' engine supports only bool and "
                    + "flattened list of string column names for the "
                    + "'parse_dates' parameter"
                ),
            )
        if names and names != lib.no_default:
            if header not in [None, 0, "infer"]:
                return (
                    False,
                    "read_csv with 'arrow' engine and provided 'names' parameter supports only 0, None and "
                    + "'infer' header values",
                )
            if isinstance(parse_dates, list) and not set(parse_dates).issubset(names):
                raise ValueError("Missing column provided to 'parse_dates'")

            empty_pandas_df = pandas.read_csv(
                **dict(
                    read_csv_kwargs,
                    nrows=0,
                    skiprows=None,
                    skipfooter=0,
                    usecols=None,
                    index_col=None,
                    names=None,
                    parse_dates=None,
                    engine=None if engine == "arrow" else engine,
                ),
            )
            columns_number = len(empty_pandas_df.columns)
            if columns_number != len(names):
                return (
                    False,
                    "read_csv with 'arrow' engine doesn't support names parameter, which length doesn't match "
                    + "with actual number of columns",
                )
        else:
            if header not in [0, "infer"]:
                return (
                    False,
                    "read_csv with 'arrow' engine without 'names' parameter provided supports only 0 and 'infer' "
                    + "header values",
                )
            if isinstance(parse_dates, list):
                empty_pandas_df = pandas.read_csv(
                    **dict(
                        read_csv_kwargs,
                        nrows=0,
                        skiprows=None,
                        skipfooter=0,
                        usecols=None,
                        index_col=None,
                        engine=None if engine == "arrow" else engine,
                    ),
                )
                if not set(parse_dates).issubset(empty_pandas_df.columns):
                    raise ValueError("Missing column provided to 'parse_dates'")

        if not read_csv_kwargs.get("skip_blank_lines", True):
            # in some corner cases empty lines are handled as '',
            # while pandas handles it as NaNs - issue #3084
            return (
                False,
                "read_csv with 'arrow' engine doesn't support skip_blank_lines = False parameter",
            )

        if skiprows is not None and not isinstance(skiprows, int):
            return (
                False,
                "read_csv with 'arrow' engine doesn't support non-integer skiprows parameter",
            )

        return True, None

    @classmethod
    def _validate_read_csv_kwargs(
        cls,
        read_csv_kwargs: ReadCsvKwargsType,
    ):
        """
        Validate `read_csv` keyword arguments.

        Should be done to mimic `pandas.read_csv` behavior.

        Parameters
        ----------
        read_csv_kwargs : dict
            Parameters of `read_csv` function.
        """
        delimiter = read_csv_kwargs.get("delimiter", None)
        sep = read_csv_kwargs.get("sep", lib.no_default)
        on_bad_lines = read_csv_kwargs.get("on_bad_lines", "error")
        error_bad_lines = read_csv_kwargs.get("error_bad_lines", None)
        warn_bad_lines = read_csv_kwargs.get("warn_bad_lines", None)
        delim_whitespace = read_csv_kwargs.get("delim_whitespace", False)

        if delimiter and (sep is not lib.no_default):
            raise ValueError(
                "Specified a sep and a delimiter; you can only specify one."
            )

        # Alias sep -> delimiter.
        if delimiter is None:
            delimiter = sep

        if delim_whitespace and (delimiter is not lib.no_default):
            raise ValueError(
                "Specified a delimiter with both sep and "
                + "delim_whitespace=True; you can only specify one."
            )
        if on_bad_lines is not None:
            if error_bad_lines is not None or warn_bad_lines is not None:
                raise ValueError(
                    "Both on_bad_lines and error_bad_lines/warn_bad_lines are set. "
                    + "Please only set on_bad_lines."
                )

        if on_bad_lines not in ["error", "warn", "skip", None]:
            raise ValueError(f"Argument {on_bad_lines} is invalid for on_bad_lines.")
