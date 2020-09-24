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

from modin.experimental.backends.omnisci.query_compiler import DFAlgQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.omnisci_on_ray.frame.data import OmnisciOnRayFrame
from modin.error_message import ErrorMessage

from pyarrow.csv import read_csv, ParseOptions, ConvertOptions, ReadOptions
import pyarrow as pa


class OmnisciOnRayIO(RayIO):

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
        decimal=b".",
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
    ):
        items = locals().copy()
        mykwargs = {k: items[k] for k in items if k in cls.arg_keys}
        eng = str(engine).lower().strip()
        try:
            if eng in ["pandas", "c"]:
                return cls._read(**mykwargs)

            if isinstance(dtype, dict):
                column_types = {c: cls._dtype_to_arrow(t) for c, t in dtype.items()}
            else:
                column_types = cls._dtype_to_arrow(dtype)

            if (type(parse_dates) is list) and type(column_types) is dict:
                for c in parse_dates:
                    column_types[c] = pa.timestamp("s")

            if names:
                if header == 0:
                    skiprows = skiprows + 1 if skiprows is not None else 1
                elif header is None or header == "infer":
                    pass
                else:
                    raise NotImplementedError(
                        "read_csv with 'arrow' engine and provided 'names' parameter supports only 0, None and 'infer' header values"
                    )
            else:
                if header == 0 or header == "infer":
                    pass
                else:
                    raise NotImplementedError(
                        "read_csv with 'arrow' engine without 'names' parameter provided supports only 0 and 'infer' header values"
                    )

            if delimiter is None:
                delimiter = sep

            if delim_whitespace and delimiter != ",":
                raise ValueError(
                    "Specified a delimiter and delim_whitespace=True; you can only specify one."
                )

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
                strings_can_be_null=None,
                include_columns=None,
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
        except (pa.ArrowNotImplementedError, NotImplementedError):
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
