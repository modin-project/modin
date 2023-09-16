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

"""Module contains class ``ModinDtypes``."""

import pandas

from modin.error_message import ErrorMessage


class ModinDtypes:
    """
    A class that hides the various implementations of the dtypes needed for optimization.

    Parameters
    ----------
    value : pandas.Series or callable
    """

    def __init__(self, value):
        if value is None:
            raise ValueError(f"ModinDtypes doesn't work with '{value}'")
        self._value = value

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Series)

    def get(self) -> pandas.Series:
        """
        Get the materialized internal representation.

        Returns
        -------
        pandas.Series
        """
        if not self.is_materialized:
            if callable(self._value):
                self._value = self._value()
                if self._value is None:
                    self._value = pandas.Series([])
            else:
                raise NotImplementedError(type(self._value))
        return self._value

    def __len__(self):
        """
        Redirect the 'len' request to the internal representation.

        Returns
        -------
        int

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return len(self._value)

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        The default implementation generates a recursion error. In a short:
        during the construction of the object, `__getattr__` function is called, which
        is not intended to be used in situations where the object is not initialized.
        """
        return (self.__class__, (self._value,))

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.

        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Series``. The main limitations of the current approach are type
        checking and the use of this object where pandas dtypes are supposed to be used.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute.

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return self._value.__getattribute__(name)

    def copy(self) -> "ModinDtypes":
        """
        Copy an object without materializing the internal representation.

        Returns
        -------
        ModinDtypes
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        return ModinDtypes(idx_cache)

    def __getitem__(self, key):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return self._value.__getitem__(key)

    def __setitem__(self, key, item):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        self._value.__setitem__(key, item)

    def __iter__(self):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return iter(self._value)

    def __contains__(self, key):  # noqa: GL08
        if not self.is_materialized:
            self.get()
        return key in self._value


class LazyProxyCategoricalDtype(pandas.CategoricalDtype):
    """
    A lazy proxy representing ``pandas.CategoricalDtype``.

    Parameters
    ----------
    categories : list-like, optional
    ordered : bool, default: False

    Notes
    -----
    Important note! One shouldn't use the class' constructor to instantiate a proxy instance,
    it's intended only for compatibility purposes! In order to create a new proxy instance
    use the appropriate class method `._build_proxy(...)`.
    """

    def __init__(self, categories=None, ordered=False):
        # These will be initialized later inside of the `._build_proxy()` method
        self._parent, self._column_name, self._categories_val, self._materializer = (
            None,
            None,
            None,
            None,
        )
        super().__init__(categories, ordered)

    def _update_proxy(self, parent, column_name):
        """
        Create a new proxy, if either parent or column name are different.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.

        Returns
        -------
        pandas.CategoricalDtype or LazyProxyCategoricalDtype
        """
        if self._is_materialized:
            # The parent has been materialized, we don't need a proxy anymore.
            return pandas.CategoricalDtype(self.categories, ordered=self._ordered)
        elif parent is self._parent and column_name == self._column_name:
            return self
        else:
            return self._build_proxy(parent, column_name, self._materializer)

    @classmethod
    def _build_proxy(cls, parent, column_name, materializer):
        """
        Construct a lazy proxy.

        Parameters
        ----------
        parent : object
            Source object to extract categories on demand.
        column_name : str
            Column name of the categorical column in the source object.
        materializer : callable(parent, column_name) -> pandas.CategoricalDtype
            A function to call in order to extract categorical values.

        Returns
        -------
        LazyProxyCategoricalDtype
        """
        result = cls()
        result._parent = parent
        result._column_name = column_name
        result._materializer = materializer
        return result

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        This object is serialized into a ``pandas.CategoricalDtype`` as an actual proxy can't be
        properly serialized because of the references it stores for its potentially distributed parent.
        """
        return (pandas.CategoricalDtype, (self.categories, self.ordered))

    @property
    def _categories(self):
        """
        Get materialized categorical values.

        Returns
        -------
        pandas.Index
        """
        if not self._is_materialized:
            self._materialize_categories()
        return self._categories_val

    @_categories.setter
    def _categories(self, categories):
        """
        Set new categorical values.

        Parameters
        ----------
        categories : list-like
        """
        self._categories_val = categories
        self._parent = None  # The parent is not required any more
        self._materializer = None

    @property
    def _is_materialized(self) -> bool:
        """
        Check whether categorical values were already materialized.

        Returns
        -------
        bool
        """
        return self._categories_val is not None

    def _materialize_categories(self):
        """Materialize actual categorical values."""
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=self._parent is None,
            extra_log="attempted to materialize categories with parent being 'None'",
        )
        categoricals = self._materializer(self._parent, self._column_name)
        self._categories = categoricals.categories
        self._ordered = categoricals.ordered
