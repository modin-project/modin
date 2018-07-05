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
        if split_query[:2] == ["CREATE", "TABLE"]:
            column_names = " ".join(split_query[3:])\
                .replace("(", "").replace(")", "").split(", ")
            columns = Series(column_names)
            self._tables[split_query[2]] = DataFrame(columns=columns)

        elif split_query[:2] == ["INSERT", "INTO"]:
            table = self._tables[split_query[2]]
            values = " ".join(split_query[4:])\
                .replace("(", "").replace(")", "").split(", ")
            to_append = Series([eval(i) for i in values], index=table.columns)
            self._tables[split_query[2]] =\
                table.append(to_append, ignore_index=True)
            print(self._tables[split_query[2]])
        else:
            print("ERROR")


def connect(name):
    return Connection(name)
