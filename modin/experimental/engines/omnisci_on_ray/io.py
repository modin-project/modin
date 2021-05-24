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

from csv import Dialect
from typing import Union, Sequence, Callable, Dict, Tuple
import inspect
import os

from modin.experimental.backends.omnisci.query_compiler import DFAlgQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.omnisci_on_ray.frame.data import OmnisciOnRayFrame
from modin.error_message import ErrorMessage
from modin.engines.base.io.text.text_file_dispatcher import TextFileDispatcher

from pyarrow.csv import read_csv, ParseOptions, ConvertOptions, ReadOptions
import pyarrow as pa

import pandas
from pandas.io.parsers import _validate_usecols_arg
from pandas._typing import FilePathOrBuffer

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
        FilePathOrBuffer,
        None,
    ],
]


class ArrowEngineException(Exception):
    """Exception should raised in the case if Arrow engine specific incompatibilities is found."""


class OmnisciOnRayIO(RayIO, TextFileDispatcher):

    frame_cls = OmnisciOnRayFrame
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
        "dialect",
        "error_bad_lines",
        "warn_bad_lines",
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
        "true_values",
        "false_values",
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
        "lineterminator",
        "dialect",
        "quoting",
        "comment",
        "warn_bad_lines",
        "error_bad_lines",
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
        dialect=None,
        error_bad_lines=True,
        warn_bad_lines=True,
        skipfooter=0,
        doublequote=True,
        delim_whitespace=False,
        low_memory=True,
        memory_map=False,
        float_precision=None,
        storage_options=None,
    ):
        items = locals().copy()
        mykwargs = {k: items[k] for k in items if k in cls.arg_keys}
        eng = str(engine).lower().strip()
        try:
            if eng in ["pandas", "c"]:
                return cls._read(**mykwargs)
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

            if names and header == 0:
                skiprows = skiprows + 1 if skiprows is not None else 1

            if delimiter is None:
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
                true_values=None,
                false_values=None,
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
                column_names=names,
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
        Define `usecols` parameter in the way pyarrow can process it.
        ----------
        read_csv_kwargs:
                read_csv function parameters.

        Returns
        -------
        usecols_md: list
                Redefined `usecols` parameter.
        """
        usecols = read_csv_kwargs.get("usecols", None)
        engine = read_csv_kwargs.get("engine", None)
        usecols_md, usecols_names_dtypes = _validate_usecols_arg(usecols)
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

        if read_csv_kwargs.get("compression", "infer") != "infer":
            return (
                False,
                "read_csv with 'arrow' engine doesn't support compression parameter, compression inferred"
                " automatically (supported compression types are gzip and bz2)",
            )

        if isinstance(filepath_or_buffer, str):
            if not os.path.exists(filepath_or_buffer):
                from pandas.io.common import is_url

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
                    "read_csv with 'arrow' engine doesn't support file handles",
                )
            else:
                raise ValueError(
                    f"Invalid file path or buffer object type: {type(filepath_or_buffer)}"
                )

        read_csv_signature = inspect.signature(cls.read_csv)
        read_csv_default_kwargs = {
            k: v.default
            for k, v in read_csv_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        for arg in cls.unsupported_args:
            if read_csv_kwargs[arg] != read_csv_default_kwargs[arg]:
                return (
                    False,
                    f"read_csv with 'arrow' engine doesn't support {arg} parameter",
                )
        if delimiter is not None and read_csv_kwargs.get("delim_whitespace", False):
            raise ValueError(
                "Specified a delimiter with both sep and delim_whitespace=True; you can only specify one."
            )

        if names:
            empty_pandas_df = pandas.read_csv(
                **dict(
                    read_csv_kwargs,
                    nrows=0,
                    skiprows=None,
                    skipfooter=0,
                    usecols=None,
                    index_col=None,
                    names=None,
                    engine=None if engine == "arrow" else engine,
                ),
            )
            columns_number = len(empty_pandas_df.columns)
            if header not in [None, 0, "infer"]:
                return (
                    False,
                    "read_csv with 'arrow' engine and provided 'names' parameter supports only 0, None and "
                    "'infer' header values",
                )
            elif columns_number != len(names):
                return (
                    False,
                    "read_csv with 'arrow' engine doesn't support names parameter, which length doesn't match "
                    "with actual number of columns",
                )
        else:
            if header not in [0, "infer"]:
                return (
                    False,
                    "read_csv with 'arrow' engine without 'names' parameter provided supports only 0 and 'infer' "
                    "header values",
                )

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
