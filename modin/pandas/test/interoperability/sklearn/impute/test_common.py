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

import pytest
import numpy as np
from scipy import sparse
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_allclose_dense_sparse
from sklearn.utils._testing import assert_array_equal
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer


def imputers():
    return [IterativeImputer(tol=0.1), KNNImputer(), SimpleImputer()]


def sparse_imputers():
    return [SimpleImputer()]


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputation_missing_value_in_test_array(imputer):
    # [Non Regression Test for issue #13968] Missing value in test set should
    # not throw an error and return a finite dataset
    train = [[1], [2]]
    test = [[3], [np.nan]]
    imputer.set_params(add_indicator=True)
    imputer.fit(train).transform(test)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1, 0])
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_imputers_add_indicator(marker, imputer):
    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imputer.set_params(missing_values=marker, add_indicator=True)

    X_trans = imputer.fit_transform(X)
    assert_allclose(X_trans[:, -4:], X_true_indicator)
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))

    imputer.set_params(add_indicator=False)
    X_trans_no_indicator = imputer.fit_transform(X)
    assert_allclose(X_trans[:, :-4], X_trans_no_indicator)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("marker", [np.nan, -1])
@pytest.mark.parametrize(
    "imputer", sparse_imputers(), ids=lambda x: x.__class__.__name__
)
def test_imputers_add_indicator_sparse(imputer, marker):
    X = sparse.csr_matrix(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    X_true_indicator = sparse.csr_matrix(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    imputer.set_params(missing_values=marker, add_indicator=True)

    X_trans = imputer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans[:, -4:], X_true_indicator)
    assert_array_equal(imputer.indicator_.features_, np.array([0, 1, 2, 3]))

    imputer.set_params(add_indicator=False)
    X_trans_no_indicator = imputer.fit_transform(X)
    assert_allclose_dense_sparse(X_trans[:, :-4], X_trans_no_indicator)


# ConvergenceWarning will be raised by the IterativeImputer
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_pandas_na_integer_array_support(imputer, add_indicator):
    # Test pandas IntegerArray with pd.NA
    pd = pytest.importorskip("modin.pandas")
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    X = np.array(
        [
            [marker, 1, 5, marker, 1],
            [2, marker, 1, marker, 2],
            [6, 3, marker, marker, 3],
            [1, 2, 9, marker, 4],
        ]
    )
    # fit on numpy array
    X_trans_expected = imputer.fit_transform(X)

    # Creates dataframe with IntegerArrays with pd.NA
    X_df = pd.DataFrame(X, dtype="Int16", columns=["a", "b", "c", "d", "e"])

    # fit on pandas dataframe with IntegerArrays
    X_trans = imputer.fit_transform(X_df)

    assert_allclose(X_trans_expected, X_trans)


@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize("add_indicator", [True, False])
def test_imputers_feature_names_out_pandas(imputer, add_indicator):
    """Check feature names out for imputers."""
    pd = pytest.importorskip("modin.pandas")
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)

    X = np.array(
        [
            [marker, 1, 5, 3, marker, 1],
            [2, marker, 1, 4, marker, 2],
            [6, 3, 7, marker, marker, 3],
            [1, 2, 9, 8, marker, 4],
        ]
    )
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d", "e", "f"])
    imputer.fit(X_df)

    names = imputer.get_feature_names_out()

    if add_indicator:
        expected_names = [
            "a",
            "b",
            "c",
            "d",
            "f",
            "missingindicator_a",
            "missingindicator_b",
            "missingindicator_d",
            "missingindicator_e",
        ]
        assert_array_equal(expected_names, names)
    else:
        expected_names = ["a", "b", "c", "d", "f"]
        assert_array_equal(expected_names, names)


@pytest.mark.parametrize("keep_empty_features", [True, False])
@pytest.mark.parametrize("imputer", imputers(), ids=lambda x: x.__class__.__name__)
def test_keep_empty_features(imputer, keep_empty_features):
    """Check that the imputer keeps features with only missing values."""
    X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
    imputer = imputer.set_params(
        add_indicator=False, keep_empty_features=keep_empty_features
    )

    for method in ["fit_transform", "transform"]:
        X_imputed = getattr(imputer, method)(X)
        if keep_empty_features:
            assert X_imputed.shape == X.shape
        else:
            assert X_imputed.shape == (X.shape[0], X.shape[1] - 1)
