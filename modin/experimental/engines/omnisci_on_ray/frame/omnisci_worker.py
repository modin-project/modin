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
import os
import sys

import pyarrow

prev = sys.getdlopenflags()
sys.setdlopenflags(1 | 256)  # RTLD_LAZY+RTLD_GLOBAL
from dbe import PyDbEngine

sys.setdlopenflags(prev)


class OmnisciServer:
    _server = None

    @classmethod
    def start_server(cls, db_path="data", calc_port=6001):
        if cls._server is None:
            cls._server = PyDbEngine(db_path, calc_port)

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
        has_dicts = False
        for i, field in enumerate(schema):
            if pyarrow.types.is_dictionary(field.type):
                has_dicts = True
                new_field = pyarrow.field(
                    field.name, field.type.value_type, field.nullable, field.metadata
                )
                new_schema = new_schema.set(i, new_field)

        if has_dicts:
            table = table.cast(new_schema)

        cls._server.consumeArrowTable(name, table)

        return name

    @classmethod
    def put_pandas_to_omnisci(cls, df, name=None):
        return cls.put_arrow_to_omnisci(pyarrow.Table.from_pandas(df))
