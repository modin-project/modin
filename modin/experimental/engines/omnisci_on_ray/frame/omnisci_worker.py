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

import uuid
import sys
import os

import pyarrow as pa
import numpy as np

prev = sys.getdlopenflags()
sys.setdlopenflags(1 | 256)  # RTLD_LAZY+RTLD_GLOBAL
from dbe import PyDbEngine

sys.setdlopenflags(prev)

from modin.config import OmnisciFragmentSize


class OmnisciServer:
    _server = None

    @classmethod
    def start_server(cls):
        if cls._server is None:
            cls._server = PyDbEngine(
                enable_union=1,
                enable_columnar_output=1,
                enable_lazy_fetch=0,
                null_div_by_zero=1,
            )

    @classmethod
    def stop_server(cls):
        if cls._server is not None:
            cls._server.reset()
            cls._server = None

    def __init__(self):
        self.start_server()

    @classmethod
    def executeDDL(cls, query):
        cls._server.executeDDL(query)

    @classmethod
    def executeDML(cls, query):
        r = cls._server.executeDML(query)
        # todo: assert r
        return r

    @classmethod
    def executeRA(cls, query):
        r = cls._server.executeRA(query)
        # todo: assert r
        return r

    @classmethod
    def _genName(cls, name):
        if not name:
            name = "frame_" + str(uuid.uuid4()).replace("-", "")
        # TODO: reword name in case of caller's mistake
        return name

    @classmethod
    def put_arrow_to_omnisci(cls, table, name=None):
        name = cls._genName(name)

        # Currently OmniSci doesn't support Arrow table import with
        # dictionary columns. Here we cast dictionaries until support
        # is in place.
        # https://github.com/modin-project/modin/issues/1738
        schema = table.schema
        new_schema = schema
        need_cast = False
        new_cols = {}
        for i, field in enumerate(schema):
            if pa.types.is_dictionary(field.type):
                # Conversion for dictionary of null type to string is not supported
                # in Arrow. Build new column for this case for now.
                if pa.types.is_null(field.type.value_type):
                    mask_vals = np.full(table.num_rows, True, dtype=bool)
                    mask = pa.array(mask_vals)
                    new_col_data = np.empty(table.num_rows, dtype=str)
                    new_col = pa.array(new_col_data, pa.string(), mask)
                    new_cols[i] = new_col
                else:
                    need_cast = True
                new_field = pa.field(
                    field.name, pa.string(), field.nullable, field.metadata
                )
                new_schema = new_schema.set(i, new_field)

        for i, col in new_cols.items():
            table = table.set_column(i, new_schema[i], col)

        if need_cast:
            table = table.cast(new_schema)

        fragment_size = OmnisciFragmentSize.get()
        if fragment_size is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                fragment_size = table.num_rows // cpu_count
                fragment_size = min(fragment_size, 2 ** 25)
                fragment_size = max(fragment_size, 2 ** 18)
            else:
                fragment_size = 0
        else:
            fragment_size = int(fragment_size)

        cls._server.importArrowTable(name, table, fragment_size=fragment_size)

        return name

    @classmethod
    def put_pandas_to_omnisci(cls, df, name=None):
        return cls.put_arrow_to_omnisci(pa.Table.from_pandas(df))
