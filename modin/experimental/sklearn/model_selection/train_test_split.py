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

"""Module holds `train_test_splt` function."""


# FIXME: Change `**options`-->`train_size=0.75`
def train_test_split(df, **options):
    """
    Split input data to train and test data.

    Parameters
    ----------
    df : modin.pandas.DataFrame / modin.pandas.Series
        Data to split.
    **options : dict
        Keyword arguments. If `train_size` key isn't provided
        `train_size` will be 0.75.

    Returns
    -------
    tuple
        A pair of modin.pandas.DataFrame / modin.pandas.Series.
    """
    train_size = options.get("train_size", 0.75)
    train = df.iloc[: int(len(df) * train_size)]
    test = df.iloc[len(train) :]
    return train, test
