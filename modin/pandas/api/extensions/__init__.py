"""
Public API for extending modin objects.
"""

from pandas.core.accessor import _register_accessor
from pandas.api.extensions import *  # noqa: F401, F403


def register_dataframe_accessor(name):
    from modin.pandas import DataFrame

    return _register_accessor(name, DataFrame)


def register_series_accessor(name):
    from modin.pandas import Series

    return _register_accessor(name, Series)


def register_index_accessor(name):
    from modin.pandas import Index

    return _register_accessor(name, Index)
