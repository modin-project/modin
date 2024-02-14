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


"""Module houses experimental Modin parser classes, that are used for data parsing on the workers."""

import warnings
from io import BytesIO

import pandas
from pandas.util._decorators import doc

from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.parsers import (
    PandasCSVParser,
    PandasParser,
    _doc_pandas_parser_class,
    _doc_parse_func,
    _doc_parse_parameters_common,
    _split_result_for_readers,
)


@doc(_doc_pandas_parser_class, data_type="multiple CSV files simultaneously")
class ExperimentalPandasCSVGlobParser(PandasCSVParser):
    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""chunks : list
    List, where each element of the list is a list of tuples. The inner lists
    of tuples contains the data file name of the chunk, chunk start offset, and
    chunk end offsets for its corresponding file.""",
    )
    def parse(chunks, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        index_col = kwargs.get("index_col", None)

        # `single_worker_read` just pass filename via chunks; need check
        if isinstance(chunks, str):
            return pandas.read_csv(chunks, **kwargs)

        # pop `compression` from kwargs because `bio` below is uncompressed
        compression = kwargs.pop("compression", "infer")
        storage_options = kwargs.pop("storage_options", None) or {}
        pandas_dfs = []
        for fname, start, end in chunks:
            if start is not None and end is not None:
                with OpenFile(fname, "rb", compression, **storage_options) as bio:
                    if kwargs.get("encoding", None) is not None:
                        header = b"" + bio.readline()
                    else:
                        header = b""
                    bio.seek(start)
                    to_read = header + bio.read(end - start)
                pandas_dfs.append(pandas.read_csv(BytesIO(to_read), **kwargs))
            else:
                # This only happens when we are reading with only one worker (Default)
                return pandas.read_csv(
                    fname,
                    compression=compression,
                    storage_options=storage_options,
                    **kwargs,
                )

        # Combine read in data.
        if len(pandas_dfs) > 1:
            pandas_df = pandas.concat(pandas_dfs)
        elif len(pandas_dfs) > 0:
            pandas_df = pandas_dfs[0]
        else:
            pandas_df = pandas.DataFrame()

        # Set internal index.
        if index_col is not None:
            index = pandas_df.index
        else:
            # The lengths will become the RangeIndex
            index = len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            index,
            pandas_df.dtypes,
        ]


@doc(_doc_pandas_parser_class, data_type="pickled pandas objects")
class ExperimentalPandasPickleParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = 1
        single_worker_read = kwargs.pop("single_worker_read", None)
        df = pandas.read_pickle(fname, **kwargs)
        if single_worker_read:
            return df
        assert isinstance(
            df, pandas.DataFrame
        ), f"Pickled obj type: [{type(df)}] in [{fname}]; works only with pandas.DataFrame"

        length = len(df)
        width = len(df.columns)

        return _split_result_for_readers(1, num_splits, df) + [length, width]


@doc(_doc_pandas_parser_class, data_type="parquet files")
class ExperimentalPandasParquetParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = 1
        single_worker_read = kwargs.pop("single_worker_read", None)
        df = pandas.read_parquet(fname, **kwargs)
        if single_worker_read:
            return df

        length = len(df)
        width = len(df.columns)

        return _split_result_for_readers(1, num_splits, df) + [length, width]


@doc(_doc_pandas_parser_class, data_type="json files")
class ExperimentalPandasJsonParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = 1
        single_worker_read = kwargs.pop("single_worker_read", None)
        df = pandas.read_json(fname, **kwargs)
        if single_worker_read:
            return df

        length = len(df)
        width = len(df.columns)

        return _split_result_for_readers(1, num_splits, df) + [length, width]


@doc(_doc_pandas_parser_class, data_type="XML files")
class ExperimentalPandasXmlParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = 1
        single_worker_read = kwargs.pop("single_worker_read", None)
        df = pandas.read_xml(fname, **kwargs)
        if single_worker_read:
            return df

        length = len(df)
        width = len(df.columns)

        return _split_result_for_readers(1, num_splits, df) + [length, width]


@doc(_doc_pandas_parser_class, data_type="custom text")
class ExperimentalCustomTextParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        return PandasParser.generic_parse(fname, **kwargs)
