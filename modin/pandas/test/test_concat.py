from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import pandas
import numpy as np
import modin.pandas as pd
from modin.pandas.utils import from_pandas, to_pandas
from .utils import df_equals, test_data_keys, test_data_values, random_state, RAND_LOW, RAND_HIGH

pd.DEFAULT_NPARTITIONS = 4


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
def test_modin_concat(data1, data2):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)

    df_equals(
        pd.concat([modin_df1, modin_df2]), pandas.concat([pandas_df1, pandas_df2])
    )


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
def test_modin_concat_with_series(data1, data2):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)

    # Test axis=0
    pandas_series = pandas.Series(random_state.random_integers(RAND_LOW, RAND_HIGH, pandas_df1.shape[1]))
    modin_series = modin.Series(random_state.random_integers(RAND_LOW, RAND_HIGH, modin_df1.shape[1]))
    df_equals(
        pd.concat([modin_df1, modin_df2, pandas_series], axis=0),
        pandas.concat([pandas_df1, pandas_df2, pandas_series], axis=0),
    )

    # Test axis=1
    pandas_series = pandas.Series(random_state.random_integers(RAND_LOW, RAND_HIGH, pandas_df1.shape[0]))
    modin_series = modin.Series(random_state.random_integers(RAND_LOW, RAND_HIGH, modin_df1.shape[0]))
    df_equals(
        pd.concat([modin_df1, modin_df2, pandas_series], axis=1),
        pandas.concat([pandas_df1, pandas_df2, pandas_series], axis=1),
    )


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
def test_modin_concat_on_index(data1, data2):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)

    df_equals(
        pd.concat([modin_df1, modin_df2], axis="index"),
        pandas.concat([pandas_df1, pandas_df2], axis="index"),
    )

    df_equals(
        pd.concat([modin_df1, modin_df2], axis="rows"),
        pandas.concat([pandas_df, pandas_df2], axis="rows"),
    )

    df_equals(
        pd.concat([modin_df1, modin_df2], axis=0), pandas.concat([pandas_df, pandas_df2], axis=0)
    )


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
def test_ray_concat_on_column(data1, data2):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)

    df_equals(
        pd.concat([modin_df1, modin_df2], axis=1), pandas.concat([pandas_df1, pandas_df2], axis=1)
    )

    df_equals(
        pd.concat([modin_df1, modin_df2], axis="columns"),
        pandas.concat([pandas_df1, pandas_df2], axis="columns"),
    )


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
def test_invalid_axis_errors(data1, data2):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)

    with pytest.raises(ValueError):
        pd.concat([modin_df1, modin_df2], axis=2)


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data3", test_data_values, ids=test_data_keys)
def test_mixed_concat(data1, data2, data3):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    pandas_df3 = pandas.DataFrame(data3)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)
    modin_df3 = pd.DataFrame(data3)

    df_equals(pd.concat([modin_df1, modin_df2, modin_df3]), pandas.concat([pandas_df1, pandas_df2, pandas_df3]))


@pytest.mark.parametrize("data1", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data2", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("data3", test_data_values, ids=test_data_keys)
def test_mixed_inner_concat(data1, data2, data3):
    pandas_df1 = pandas.DataFrame(data1)
    pandas_df2 = pandas.DataFrame(data2)
    pandas_df3 = pandas.DataFrame(data3)
    modin_df1 = pd.DataFrame(data1)
    modin_df2 = pd.DataFrame(data2)
    modin_df3 = pd.DataFrame(data3)

    df_equals(pd.concat([modin_df1, modin_df2, modin_df3], join='inner'), pandas.concat([pandas_df1, pandas_df2, pandas_df3], join='inner'))
