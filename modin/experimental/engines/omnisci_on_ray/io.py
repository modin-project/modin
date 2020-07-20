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
from modin.experimental.engines.omnisci_on_ray.frame.omnisci_worker import OmnisciServer

class OmnisciOnRayIO(RayIO):

    frame_cls = OmnisciOnRayFrame
    query_compiler_cls = DFAlgQueryCompiler

    #@classmethod
    #def from_arrow(cls, at):
    #    return cls.query_compiler_cls.from_arrow(at, cls.frame_cls)

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
        mykwargs = {k : items[k] for k in items if k in cls.arg_keys}

        try:
            if str(engine).lower().strip() in ['pandas', 'c']:
                return cls._read(**mykwargs)

            from pyarrow.csv import read_csv, timestamp, ParseOptions, ConvertOptions, ReadOptions

            column_types= dtype if dtype is dict else {}
            if( (type(parse_dates) is list) # and (type(parse_dates[0]) is str)  # like parse_dates=["dd",]
                and type(column_types) is dict):
                for c in parse_dates:
                    column_types[c] = timestamp('s') 

            """
            class ParseOptions(delimiter=None, quote_char=None, double_quote=None, escape_char=None, newlines_in_values=None, ignore_empty_lines=None)
            Options for parsing CSV files.

            Parameters
            delimiter: 1-character string, optional (default ',')

                The character delimiting individual cells in the CSV data.  
            quote_char: 1-character string or False, optional (default '"')

                The character used optionally for quoting CSV values  
                (False if quoting is not allowed).  
            double_quote: bool, optional (default True)

                Whether two quotes in a quoted CSV value denote a single quote  
                in the data.  
            escape_char: 1-character string or False, optional (default False)

                The character used optionally for escaping special characters  
                (False if escaping is not allowed).  
            newlines_in_values: bool, optional (default False)

                Whether newline characters are allowed in CSV values.  
                Setting this to True reduces the performance of multi-threaded  
                CSV reading.  
            ignore_empty_lines: bool, optional (default True)

                Whether empty lines are ignored in CSV input.  
                If False, an empty line is interpreted as containing a single empty  
                value (assuming a one-column CSV file).  
            """
            po = ParseOptions(
                    delimiter = sep if sep else '\s+' if delim_whitespace else delimiter,
                    quote_char=quotechar,
                    double_quote=doublequote, 
                    escape_char=escapechar,
                    newlines_in_values=False,
                    ignore_empty_lines=skip_blank_lines
            )


            """
            Options for converting CSV data.

            Parameters
            check_utf8 : bool, optional (default True)

                Whether to check UTF8 validity of string columns.  
            column_types: dict, optional

                Map column names to column types  
                (disabling type inference on those columns).  
            null_values: list, optional

                A sequence of strings that denote nulls in the data  
                (defaults are appropriate in most cases).  
            true_values: list, optional

                A sequence of strings that denote true booleans in the data  
                (defaults are appropriate in most cases).  
            false_values: list, optional

                A sequence of strings that denote false booleans in the data  
                (defaults are appropriate in most cases).  
            strings_can_be_null: bool, optional (default False)

                Whether string / binary columns can have null values.  
                If true, then strings in null_values are considered null for  
                string columns.  
                If false, then all strings are valid string values.  
            auto_dict_encode: bool, optional (default False)

                Whether to try to automatically dict-encode string / binary data.  
                If true, then when type inference detects a string or binary column,  
                it it dict-encoded up to `auto_dict_max_cardinality` distinct values  
                (per chunk), after which it switches to regular encoding.  
                This setting is ignored for non-inferred columns (those in  
                `column_types`).  
            auto_dict_max_cardinality: int, optional

                The maximum dictionary cardinality for `auto_dict_encode`.  
                This value is per chunk.  
            include_columns: list, optional

                The names of columns to include in the Table.  
                If empty, the Table will include all columns from the CSV file.  
                If not empty, only these columns will be included, in this order.  
            include_missing_columns: bool, optional (default False)

                If false, columns in `include_columns` but not in the CSV file will  
                error out.  
                If true, columns in `include_columns` but not in the CSV file will  
                produce a column of nulls (whose type is selected using  
                `column_types`, or null by default).  
                This option is ignored if `include_columns` is empty.  
            """
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
                    auto_dict_max_cardinality=None)

            """
            class ReadOptions(use_threads=None, block_size=None, skip_rows=None, column_names=None, autogenerate_column_names=None)
            Options for reading CSV files.

            Parameters
            use_threads : bool, optional (default True)

                Whether to use multiple threads to accelerate reading  
            block_size : int, optional

                How much bytes to process at a time from the input stream.  
                This will determine multi-threading granularity as well as  
                the size of individual chunks in the Table.  
            skip_rows: int, optional (default 0)

                The number of rows to skip at the start of the CSV data, not  
                including the row of column names (if any).  
            column_names: list, optional

                The column names of the target table.  If empty, fall back on  
                `autogenerate_column_names`.  
            autogenerate_column_names: bool, optional (default False)

                Whether to autogenerate column names if `column_names` is empty.  
                If true, column names will be of the form "f0", "f1"...  
                If false, column names will be read from the first CSV row  
                after `skip_rows`.  
            """
            ro = ReadOptions(
                    use_threads=True,
                    block_size=None,
                    skip_rows=skiprows,
                    column_names=names,
                    autogenerate_column_names=None)

            at = read_csv(filepath_or_buffer, read_options=ro, parse_options=po, convert_options=co)
            
            return cls.from_arrow(at) # can be used to switch between arrow and pandas frames: cls.from_pandas(at.to_pandas())
        except:
            ErrorMessage.default_to_pandas("`read_csv`")
            return cls._read(**mykwargs)

