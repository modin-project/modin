import pandas

from modin.error_message import ErrorMessage
from .base import BasePandasDataset
from .dataframe import DataFrame
from .utils import to_pandas


def isna(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, BasePandasDataset):
        return obj.isna()
    else:
        return pandas.isna(obj)


isnull = isna


def notna(obj):
    if isinstance(obj, BasePandasDataset):
        return obj.notna()
    else:
        return pandas.notna(obj)


notnull = notna


def merge(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
):
    """Database style join, where common columns in "on" are merged.

    Args:
        left: DataFrame.
        right: DataFrame.
        how: What type of join to use.
        on: The common column name(s) to join on. If None, and left_on and
            right_on  are also None, will default to all commonly named
            columns.
        left_on: The column(s) on the left to use for the join.
        right_on: The column(s) on the right to use for the join.
        left_index: Use the index from the left as the join keys.
        right_index: Use the index from the right as the join keys.
        sort: Sort the join keys lexicographically in the result.
        suffixes: Add this suffix to the common names not in the "on".
        copy: Does nothing in our implementation
        indicator: Adds a column named _merge to the DataFrame with
            metadata from the merge about each row.
        validate: Checks if merge is a specific type.

    Returns:
         A merged Dataframe
        """
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )

    return left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
    )


def merge_ordered(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_by=None,
    right_by=None,
    fill_method=None,
    suffixes=("_x", "_y"),
    how="outer",
):
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )
    ErrorMessage.default_to_pandas("`merge_ordered`")
    if isinstance(right, DataFrame):
        right = to_pandas(right)
    return DataFrame(
        pandas.merge_ordered(
            to_pandas(left),
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_by=left_by,
            right_by=right_by,
            fill_method=fill_method,
            suffixes=suffixes,
            how=how,
        )
    )


def merge_asof(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes=("_x", "_y"),
    tolerance=None,
    allow_exact_matches=True,
    direction="backward",
):
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )
    ErrorMessage.default_to_pandas("`merge_asof`")
    if isinstance(right, DataFrame):
        right = to_pandas(right)
    return DataFrame(
        pandas.merge_asof(
            to_pandas(left),
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            by=by,
            left_by=left_by,
            right_by=right_by,
            suffixes=suffixes,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction,
        )
    )


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
    observed=False,
):
    if not isinstance(data, DataFrame):
        raise ValueError(
            "can not create pivot table with instance of type {}".format(type(data))
        )

    return data.pivot_table(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
    )


def pivot(data, index=None, columns=None, values=None):
    if not isinstance(data, DataFrame):
        raise ValueError("can not pivot with instance of type {}".format(type(data)))
    return data.pivot(index=index, columns=columns, values=values)
