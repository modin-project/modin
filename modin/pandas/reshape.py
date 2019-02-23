from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from .dataframe import DataFrame
from .utils import to_pandas
from modin.error_message import ErrorMessage


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):
    """Convert categorical variable into indicator variables.

    Args:
        data (array-like, Series, or DataFrame): data to encode.
        prefix (string, [string]): Prefix to apply to each encoded column
                                   label.
        prefix_sep (string, [string]): Separator between prefix and value.
        dummy_na (bool): Add a column to indicate NaNs.
        columns: Which columns to encode.
        sparse (bool): Not Implemented: If True, returns SparseDataFrame.
        drop_first (bool): Whether to remove the first level of encoded data.
        dtype: The dtype for the get_dummies call.

    Returns:
        DataFrame or one-hot encoded data.
    """
    if sparse:
        raise NotImplementedError(
            "SparseDataFrame is not implemented. "
            "To contribute to Modin, please visit "
            "github.com/modin-project/modin."
        )
    if not isinstance(data, DataFrame):
        ErrorMessage.default_to_pandas("`get_dummies` on non-DataFrame")
        return DataFrame(
            pandas.get_dummies(
                data,
                prefix=prefix,
                prefix_sep=prefix_sep,
                dummy_na=dummy_na,
                columns=columns,
                sparse=sparse,
                drop_first=drop_first,
                dtype=dtype,
            )
        )
    else:
        new_manager = data._query_compiler.get_dummies(
            columns,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            drop_first=drop_first,
            dtype=dtype,
        )
        return DataFrame(query_compiler=new_manager)


def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
):
    return frame.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
    )


def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name="All",
    dropna=True,
    normalize=False,
):
    ErrorMessage.default_to_pandas("`crosstab`")
    pandas_crosstab = pandas.crosstab(
        index,
        columns,
        values,
        rownames,
        colnames,
        aggfunc,
        margins,
        margins_name,
        dropna,
        normalize,
    )
    return DataFrame(pandas_crosstab)


def lreshape(data, groups, dropna=True, label=None):
    if not isinstance(data, DataFrame):
        raise ValueError("can not lreshape with instance of type {}".format(type(data)))
    ErrorMessage.default_to_pandas("`lreshape`")
    return DataFrame(
        pandas.lreshape(to_pandas(data), groups, dropna=dropna, label=label)
    )


def wide_to_long(df, stubnames, i, j, sep="", suffix=r"\d+"):
    if not isinstance(df, DataFrame):
        raise ValueError(
            "can not wide_to_long with instance of type {}".format(type(df))
        )
    ErrorMessage.default_to_pandas("`wide_to_long`")
    return DataFrame(
        pandas.wide_to_long(to_pandas(df), stubnames, i, j, sep=sep, suffix=suffix)
    )
