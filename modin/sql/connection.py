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

from ..pandas import Series, DataFrame


class Connection(object):
    def __init__(self, name):
        self._name = name
        self._cursor = None

    def cursor(self):
        self._cursor = Cursor()
        return self._cursor

    def commit(self):
        pass

    def close(self):
        self._cursor = None


class Cursor(object):
    def __init__(self):
        self._tables = {}

    def execute(self, query):
        split_query = query.split(" ")
        if " ".join(split_query[:2]) == "CREATE TABLE":
            self._create_table(split_query)

        elif " ".join(split_query[:2]) == "INSERT INTO":
            self._insert_into(split_query)
        else:
            raise NotImplementedError(
                "This API is for demonstration purposes only. Coming Soon!"
            )

    def _create_table(self, split_query):
        column_names = (
            " ".join(split_query[3:]).replace("(", "").replace(")", "").split(", ")
        )
        columns = Series(column_names)
        self._tables[split_query[2]] = DataFrame(columns=columns)

    def _insert_into(self, split_query):
        table = self._tables[split_query[2]]
        values = " ".join(split_query[4:]).replace("(", "").replace(")", "").split(", ")
        to_append = Series([eval(i) for i in values], index=table.columns)
        self._tables[split_query[2]] = table.append(to_append, ignore_index=True)


def connect(name):
    return Connection(name)
