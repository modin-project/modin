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

"""Module contains class ModinIndex."""

import functools
import uuid

import pandas
from pandas.core.indexes.api import ensure_index


class ModinIndex:
    """
    A class that hides the various implementations of the index needed for optimization.

    Parameters
    ----------
    value : sequence, PandasDataframe or callable() -> (pandas.Index, list of ints)
        If a sequence passed this will be considered as the index values.
        If a ``PandasDataframe`` passed then it will be used to lazily extract indices
        when required, note that the `axis` parameter must be passed in this case.
        If a callable passed then it's expected to return a pandas Index and a list of
        partition lengths along the index axis.
    axis : int, optional
        Specifies an axis the object represents, serves as an optional hint. This parameter
        must be passed in case value is a ``PandasDataframe``.
    """

    def __init__(self, value, axis=None):
        from modin.core.dataframe.pandas.dataframe.dataframe import PandasDataframe

        self._is_default_callable = False
        self._axis = axis

        if callable(value):
            self._value = value
        elif isinstance(value, PandasDataframe):
            assert axis is not None
            self._value = self._get_default_callable(value, axis)
            self._is_default_callable = True
        else:
            self._value = ensure_index(value)

        self._lengths_cache = None
        # index/lengths ID's for faster comparison between other ModinIndex objects,
        # these should be propagated to the copies of the index
        self._index_id = uuid.uuid4()
        self._lengths_id = uuid.uuid4()

    @staticmethod
    def _get_default_callable(dataframe_obj, axis):
        """
        Build a callable extracting index labels and partitions lengths for the specified axis.

        Parameters
        ----------
        dataframe_obj : PandasDataframe
        axis : int
            0 - extract indices, 1 - extract columns.

        Returns
        -------
        callable() -> tuple(pandas.Index, list[ints])
        """
        # HACK: for an unknown reason, the 'lambda' approach seems to trigger some strange
        # race conditions in HDK on certain versions of python, causing the tests to fail
        # (python 3.9.* and 3.10.* are the versions where we saw the problem). That's
        # really strange, but practically the same code that uses 'functools.partial'
        # instead of a lambda works absolutely fine.
        # return lambda: dataframe_obj._compute_axis_labels_and_lengths(axis)
        return functools.partial(
            type(dataframe_obj)._compute_axis_labels_and_lengths, dataframe_obj, axis
        )

    def maybe_specify_new_frame_ref(self, value, axis) -> "ModinIndex":
        """
        Set a new reference for a frame used to lazily extract index labels if it's needed.

        The method sets a new reference only if the indices are not yet materialized and
        if a PandasDataframe was originally passed to construct this index (so the ModinIndex
        object holds a reference to it). The reason the reference should be updated is that
        we don't want to hold in memory those frames that are already not needed. Once the
        reference is updated, the old frame will be garbage collected if there are no
        more references to it.

        Parameters
        ----------
        value : PandasDataframe
            New dataframe to reference.
        axis : int
            Axis to extract labels from.

        Returns
        -------
        ModinIndex
            New ModinIndex with the reference updated.
        """
        if not callable(self._value) or not self._is_default_callable:
            return self

        new_index = self.copy(copy_lengths=True)
        new_index._axis = axis
        new_index._value = self._get_default_callable(value, new_index._axis)
        return new_index

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Index)

    def get(self, return_lengths=False) -> pandas.Index:
        """
        Get the materialized internal representation.

        Parameters
        ----------
        return_lengths : bool, default: False
            In some cases, during the index calculation, it's possible to get
            the lengths of the partitions. This flag allows this data to be used
            for optimization.

        Returns
        -------
        pandas.Index
        """
        if not self.is_materialized:
            if callable(self._value):
                index, self._lengths_cache = self._value()
                self._value = ensure_index(index)
            else:
                raise NotImplementedError(type(self._value))
        if return_lengths:
            return self._value, self._lengths_cache
        else:
            return self._value

    def equals(self, other: "ModinIndex") -> bool:
        """
        Check equality of the index values.

        Parameters
        ----------
        other : ModinIndex

        Returns
        -------
        bool
            The result of the comparison.
        """
        if self._index_id == other._index_id:
            return True

        if not self.is_materialized:
            self.get()

        if not other.is_materialized:
            other.get()

        return self._value.equals(other._value)

    def compare_partition_lengths_if_possible(self, other: "ModinIndex"):
        """
        Compare the partition lengths cache for the index being stored if possible.

        The ``ModinIndex`` object may sometimes store the information about partition
        lengths along the axis the index belongs to. If both `self` and `other` have
        this information or it can be inferred from them, the method returns
        a boolean - the result of the comparison, otherwise it returns ``None``
        as an indication that the comparison cannot be made.

        Parameters
        ----------
        other : ModinIndex

        Returns
        -------
        bool or None
            The result of the comparison if both `self` and `other` contain
            the lengths data, ``None`` otherwise.
        """
        if self._lengths_id == other._lengths_id:
            return True

        can_extract_lengths_from_self = self._lengths_cache is not None or callable(
            self._value
        )
        can_extract_lengths_from_other = other._lengths_cache is not None or callable(
            other._value
        )
        if can_extract_lengths_from_self and can_extract_lengths_from_other:
            return self.get(return_lengths=True)[1] == other.get(return_lengths=True)[1]
        return None

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
        return (
            self.__class__,
            (self._value, self._axis),
            {
                "_lengths_cache": self._lengths_cache,
                "_index_id": self._index_id,
                "_lengths_id": self._lengths_id,
                "_is_default_callable": self._is_default_callable,
            },
        )

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.

        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Index``. The main limitations of the current approach are type
        checking and the use of this object where pandas indexes are supposed to be used.

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

    def copy(self, copy_lengths=False) -> "ModinIndex":
        """
        Copy an object without materializing the internal representation.

        Parameters
        ----------
        copy_lengths : bool, default: False
            Whether to copy the stored partition lengths to the
            new index object.

        Returns
        -------
        ModinIndex
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        result = ModinIndex(idx_cache, axis=self._axis)
        result._index_id = self._index_id
        result._is_default_callable = self._is_default_callable
        if copy_lengths:
            result._lengths_cache = self._lengths_cache
            result._lengths_id = self._lengths_id
        return result
