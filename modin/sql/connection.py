from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
            raise NotImplementedError("This API is for demonstration purposes "
                                      "only. Coming Soon!")

    def _create_table(self, split_query):
        column_names = " ".join(split_query[3:]) \
            .replace("(", "").replace(")", "").split(", ")
        columns = Series(column_names)
        self._tables[split_query[2]] = DataFrame(columns=columns)

    def _insert_into(self, split_query):
        table = self._tables[split_query[2]]
        values = " ".join(split_query[4:]) \
            .replace("(", "").replace(")", "").split(", ")
        to_append = Series([eval(i) for i in values], index=table.columns)
        self._tables[split_query[2]] = \
            table.append(to_append, ignore_index=True)
        print(self._tables[split_query[2]])


def connect(name):
    return Connection(name)
