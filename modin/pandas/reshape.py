from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.core.dtypes.common import is_list_like

from .dataframe import DataFrame
from modin.error_message import ErrorMessage


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
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
        return pandas.get_dummies(
            data,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            columns=columns,
            sparse=sparse,
            drop_first=drop_first,
        )
    if isinstance(data, DataFrame):
        df = data
    elif is_list_like(data):
        df = DataFrame(data)

    new_manager = df._query_compiler.get_dummies(
        columns,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        drop_first=drop_first,
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
