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
from .frame.omnisci_worker import OmnisciServer

class OmnisciOnRayIO(RayIO):

    frame_cls = OmnisciOnRayFrame
    query_compiler_cls = DFAlgQueryCompiler

    def from_arrow(cls, df):
        return cls.query_compiler_cls.from_arrow(df, cls.frame_cls)

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

        try:
            from pyarrow.csv import read_csv
            at = read_csv(filepath_or_buffer)
            # three options, to omnisci or...
            #   omniSession = OmnisciServer()
            #   omniSession... consume
            # ...or leave it as is, in Arrow,
            # or convert to pandas
            print("hello from arrow read_csv:", at)
            # todo like p.frame_id = omniSession.put_pandas_to_omnisci(df)
            return cls.from_arrow(at)
        except:
            ErrorMessage.default_to_pandas("`read_csv`")
            mykwargs = {
                "filepath_or_buffer": filepath_or_buffer,
                "sep": sep,
                "delimiter": delimiter,
                "header": header,
                "names": names,
                "index_col": index_col,
                "usecols": usecols,
                "squeeze": squeeze,
                "prefix": prefix,
                "mangle_dupe_cols": mangle_dupe_cols,
                "dtype": dtype,
                "engine": engine,
                "converters": converters,
                "true_values": true_values,
                "false_values": false_values,
                "skipinitialspace": skipinitialspace,
                "skiprows": skiprows,
                "nrows": nrows,
                "na_values": na_values,
                "keep_default_na": keep_default_na,
                "na_filter": na_filter,
                "verbose": verbose,
                "skip_blank_lines": skip_blank_lines,
                "parse_dates": parse_dates,
                "infer_datetime_format": infer_datetime_format,
                "keep_date_col": keep_date_col,
                "date_parser": date_parser,
                "dayfirst": dayfirst,
                "cache_dates": cache_dates,
                "iterator": iterator,
                "chunksize": chunksize,
                "compression": compression,
                "thousands": thousands,
                "decimal": decimal,
                "lineterminator": lineterminator,
                "quotechar": quotechar,
                "quoting": quoting,
                "escapechar": escapechar,
                "comment": comment,
                "encoding": encoding,
                "dialect": dialect,
                "error_bad_lines": error_bad_lines,
                "warn_bad_lines": warn_bad_lines,
                "skipfooter": skipfooter,
                "doublequote": doublequote,
                "delim_whitespace": delim_whitespace,
                "low_memory": low_memory,
                "memory_map": memory_map,
                "float_precision": float_precision,
            }
            return cls._read(**mykwargs)

