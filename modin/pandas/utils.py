from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ..data_management.factories import BaseFactory


def from_pandas(df):
    """Converts a pandas DataFrame to a Ray DataFrame.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.

    Returns:
        A new Ray DataFrame object.
    """
    from .dataframe import DataFrame

    return DataFrame(query_compiler=BaseFactory.from_pandas(df))


def to_pandas(df):
    """Converts a Ray DataFrame to a pandas DataFrame/Series.
    Args:
        df (modin.DataFrame): The Ray DataFrame to convert.
    Returns:
        A new pandas DataFrame.
    """
    return df._query_compiler.to_pandas()


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)
        return cls

    return decorator
