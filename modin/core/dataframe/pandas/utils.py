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


"""Collection of utility functions for the PandasDataFrame."""

import pandas
from pandas.api.types import union_categoricals


def concatenate(dfs):
    """
    Concatenate pandas DataFrames with saving 'category' dtype.

    Parameters
    ----------
    dfs : list
        List of pandas DataFrames to concatenate.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    categoricals_column_names = set.intersection(
        *[set(df.select_dtypes("category").columns.tolist()) for df in dfs]
    )

    for column_name in categoricals_column_names:
        # Build a list of all columns in all dfs with name column_name.
        categorical_columns_with_name = []
        for df in dfs:
            categorical_columns_in_df = df[column_name]
            # Fast path for when the column name is unique.
            if isinstance(categorical_columns_in_df, pandas.Series):
                categorical_columns_with_name.append(categorical_columns_in_df)
            else:
                # If the column name is repeated, df[column_name] gives a
                # a dataframe with all matching columns instead of a series.
                categorical_columns_with_name.extend(
                    col for _, col in categorical_columns_in_df.iteritems()
                )
        # Make a new category unioning all columns with the current name.
        categories = union_categoricals(categorical_columns_with_name).categories
        # Replace all columns having the current name with the new category.
        for df in dfs:
            categorical_columns_in_df = df[column_name]
            # Fast path for when the column name is unique.
            if isinstance(categorical_columns_in_df, pandas.Series):
                df[column_name] = pandas.Categorical(
                    df[column_name], categories=categories
                )
            else:
                for i in range(len(categorical_columns_in_df.columns)):
                    df.iloc[:, i] = pandas.Categorical(
                        df.iloc[:, i], categories=categories
                    )

    return pandas.concat(dfs)
