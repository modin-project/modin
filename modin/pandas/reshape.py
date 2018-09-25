from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.core.dtypes.common import is_list_like

from .dataframe import DataFrame


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
            "To contribute to Pandas on Ray, please visit "
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

    new_manager = df._data_manager.get_dummies(
        columns,
        prefix=prefix,
        prefix_sep=prefix_sep,
        dummy_na=dummy_na,
        drop_first=drop_first,
    )

    return DataFrame(data_manager=new_manager)
