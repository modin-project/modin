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

"""Module houses classes responsible for storing a virtual partition and applying a function to it."""
import math
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas
import ray

from modin.config import MinPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
    BaseDataframeAxisPartition,
)
from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.deferred_execution import (
    DeferredExecution,
    DeferredGetItem,
    MetaList,
    ObjectRefOrDeType,
    ObjectRefType,
)
from modin.core.execution.utils import remote_function
from modin.utils import _inherit_docstrings

from .partition import PandasOnRayDataframePartition


class PandasOnRayDataframeVirtualPartition(BaseDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    data : DeferredExecution or list of PandasOnRayDataframePartition
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    length : ray.ObjectRef or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    num_splits : int, optional
        The number of chunks to split the results on.
    chunk_lengths : list of ints, optional
        The chunk lengths.
    """

    partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    axis = None

    def __init__(
        self,
        data: Union[
            DeferredExecution,
            PandasOnRayDataframePartition,
            List[PandasOnRayDataframePartition],
        ],
        full_axis: bool = True,
        length: Union[int, ObjectRefType] = None,
        width: Union[int, ObjectRefType] = None,
        num_splits=None,
        chunk_lengths=None,
    ):
        self.full_axis = full_axis
        self._meta = MetaList([length, width, None])
        self._meta_offset = 0
        self._chunk_lengths_cache = chunk_lengths

        if isinstance(data, DeferredExecution):
            self._set_data_ref(data)
            self._num_splits = num_splits
            self._list_of_block_partitions = None
            return

        if not isinstance(data, Collection) or len(data) == 1:
            if not isinstance(data, Collection):
                data = [data]
            self._set_data_ref(data[0]._data)
            self._num_splits = 1
            self._list_of_block_partitions = data
            return

        self._num_splits = len(data)
        self._list_of_block_partitions = data

        non_split, lengths, full_concat = self.find_non_split_block(data)
        if non_split is not None:
            if lengths is not None:
                self._chunk_lengths_cache = lengths
            if full_concat:
                self._set_data_ref(non_split)
                return
            # TODO: We have a subset of the same frame here and can just get a single chunk
            # from the original frame instead of concatenating all these chunks.

        data = DeferredExecution([part._data for part in data], self._remote_concat)
        self._set_data_ref(data)

    @staticmethod
    def find_non_split_block(
        partitions: Iterable[PandasOnRayDataframePartition],
    ) -> Tuple[
        Union[ObjectRefOrDeType, None],
        Union[List[Union[ObjectRefType, int]], None],
        bool,
    ]:
        """
        Find a non-split block if there is one.

        The `apply()` method returns the following lazy execution tree:

                        DeferredExecution (apply func)
                                    |
                                    |
                           _DeferredSplit (n)
                             /    ...     \
                            /              \
        _DeferredGetChunk (0)     ...       _DeferredGetChunk (n - 1)

        If we need to concatenate all the `partitions` and the `partitions` are the
        complete sequence of `_DeferredGetChunk`, then we can just get the root of
        this tree and avoid the concatenation.

        Parameters
        ----------
        partitions : list of PandasOnRayDataframePartition

        Returns
        -------
        ObjectRefOrDeType or None
            The non-split block or None if not found.
        list of lengths or None
            Estimated chunk lengths, that could be different form the real ones.
        bool
            Whether the specified partitions represent the full block or just the
            first part of this block.
        """
        refs = [part._data_ref for part in partitions]
        if (
            isinstance(refs[0], _DeferredGetChunk)
            and isinstance(split := refs[0].data, _DeferredSplit)
            and (refs[0].index == 0)
            and all(prev.is_next_chunk(next) for prev, next in zip(refs[:-1], refs[1:]))
        ):
            lengths = [ref.length for ref in refs if ref.length is not None]
            return (
                split.data,
                lengths if len(lengths) == len(refs) else None,
                split.num_splits == refs[-1].index + 1,
            )
        return None, None, False

    def _set_data_ref(self, data: Union[DeferredExecution, ObjectRefType]):
        """
        Set the `_data_ref` property.

        Parameters
        ----------
        data : DeferredExecution or ObjectRefType
        """
        if isinstance(data, DeferredExecution):
            data.subscribe()
        self._data_ref = data

    def __del__(self):
        """Unsubscribe from DeferredExecution."""
        if isinstance(self._data_ref, DeferredExecution):
            self._data_ref.unsubscribe()

    @_inherit_docstrings(BaseDataframeAxisPartition.apply)
    def apply(
        self,
        func,
        *args,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        lengths=None,
        manual_partition=False,
        **kwargs,
    ) -> Union[List[PandasOnRayDataframePartition], PandasOnRayDataframePartition]:
        if not manual_partition:
            if not self.full_axis:
                # If this is not a full axis partition, it already contains a subset of
                # the full axis, so we shouldn't split the result further.
                num_splits = 1
            elif num_splits is None:
                num_splits = self._num_splits

            if (
                num_splits == 1
                or not maintain_partitioning
                or num_splits != self._num_splits
            ):
                lengths = None
            elif lengths is None:
                lengths = self._chunk_lengths

        if other_axis_partition is not None:
            if isinstance(other_axis_partition, Collection):
                if len(other_axis_partition) == 1:
                    other_part = other_axis_partition[0]._data
                else:
                    concat_fn = (
                        PandasOnRayDataframeColumnPartition
                        if self.axis
                        else PandasOnRayDataframeRowPartition
                    )._remote_concat
                    other_part = DeferredExecution(
                        [p._data for p in other_axis_partition], concat_fn
                    )
            else:
                other_part = other_axis_partition._data
            args = [other_part] + list(args)

        de = self._apply(func, args, kwargs)
        if num_splits > 1:
            de = _DeferredSplit(de, self.axis, num_splits, lengths)
            if lengths is not None and len(lengths) != num_splits:
                lengths = None
            result = [
                PandasOnRayDataframePartition(
                    _DeferredGetChunk(
                        de, i, lengths[i] if lengths is not None else None
                    )
                )
                for i in range(num_splits)
            ]
        else:
            result = [PandasOnRayDataframePartition(de)]
        if self.full_axis or other_axis_partition is not None:
            return result
        else:
            # If this is not a full axis partition, just take out the single split in the result.
            return result[0]

    @_inherit_docstrings(PandasDataframeAxisPartition.add_to_apply_calls)
    def add_to_apply_calls(self, func, *args, length=None, width=None, **kwargs):
        de = self._apply(func, args, kwargs)
        return type(self)(
            de, self.full_axis, length, width, self._num_splits, self._chunk_lengths
        )

    @_inherit_docstrings(PandasDataframeAxisPartition.split)
    def split(
        self, split_func, num_splits, f_args=None, f_kwargs=None, extract_metadata=False
    ) -> List[PandasOnRayDataframePartition]:
        chunks, meta, offsets = DeferredExecution(
            self._data_ref,
            split_func,
            args=f_args,
            kwargs=f_kwargs,
            num_returns=num_splits,
        ).exec()
        return [
            PandasOnRayDataframePartition(chunks[i], meta=meta, meta_offset=offsets[i])
            for i in range(num_splits)
        ]

    @property
    def _length_cache(self) -> Union[int, None, ObjectRefType]:
        """
        Get the cached length of the partition.

        Returns
        -------
        int or None or ObjectRefType
        """
        return self._meta[self._meta_offset]

    def length(self, materialize=True) -> Union[int, ObjectRefType]:
        """
        Get the length of the partition.

        Parameters
        ----------
        materialize : bool, default: True

        Returns
        -------
        int or ObjectRefType
        """
        if self._length_cache is None:
            self._calculate_lengths(materialize)
        elif materialize:
            self._meta.materialize()
        return self._length_cache

    @property
    def _width_cache(self) -> Union[int, None, ObjectRefType]:
        """
        Get the cached width of the partition.

        Returns
        -------
        int or None or ObjectRefType
        """
        return self._meta[self._meta_offset + 1]

    def width(self, materialize=True) -> Union[int, ObjectRefType]:
        """
        Get the width of the partition.

        Parameters
        ----------
        materialize : bool, default: True

        Returns
        -------
        int or ObjectRefType
        """
        if self._width_cache is None:
            self._calculate_lengths(materialize)
        elif materialize:
            self._meta.materialize()
        return self._width_cache

    def _calculate_lengths(self, materialize=True):
        """
        Calculate the length and width of the partition.

        Parameters
        ----------
        materialize : bool, default: True
        """
        if self._list_of_block_partitions is not None:
            from . import PandasOnRayDataframePartitionManager

            lengths = [part.length(False) for part in self._list_of_block_partitions]
            widths = [part.width(False) for part in self._list_of_block_partitions]
            materialized = PandasOnRayDataframePartitionManager.materialize_futures(
                lengths + widths
            )
            self._meta[self._meta_offset] = sum(materialized[: len(lengths)])
            self._meta[self._meta_offset + 1] = sum(materialized[len(lengths) :])
        else:
            self.force_materialization()
            if materialize:
                self._meta.materialize()

    @_inherit_docstrings(PandasDataframeAxisPartition.drain_call_queue)
    def drain_call_queue(self, num_splits=None):
        if num_splits:
            self._num_splits = num_splits

    @_inherit_docstrings(PandasDataframeAxisPartition.force_materialization)
    def force_materialization(self, get_ip=False):
        self._data  # Trigger execution
        self._num_splits = 1
        self._chunk_lengths_cache = None
        self._list_of_block_partitions = None
        return self

    @_inherit_docstrings(PandasDataframeAxisPartition.wait)
    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        RayWrapper.wait(self._data)

    @_inherit_docstrings(PandasDataframeAxisPartition.to_pandas)
    def to_pandas(self):
        return RayWrapper.materialize(self._data)

    @_inherit_docstrings(PandasDataframeAxisPartition.to_numpy)
    def to_numpy(self):
        return self.to_pandas().to_numpy()

    @_inherit_docstrings(PandasDataframeAxisPartition.mask)
    def mask(self, row_indices, col_indices):
        part = PandasOnRayDataframePartition(self._data_ref).mask(
            row_indices, col_indices
        )
        return type(self)(part, False)

    @property
    @_inherit_docstrings(BaseDataframeAxisPartition.list_of_blocks)
    def list_of_blocks(self):
        return [part._data for part in self.list_of_block_partitions]

    @property
    @_inherit_docstrings(PandasDataframeAxisPartition.list_of_block_partitions)
    def list_of_block_partitions(self) -> list:
        if self._list_of_block_partitions is not None:
            return self._list_of_block_partitions

        data = self._data_ref
        num_splits = self._num_splits
        if num_splits > 1:
            lengths = self._chunk_lengths
            data = _DeferredSplit(data, self.axis, num_splits, lengths)
            if lengths is not None and len(lengths) != num_splits:
                lengths = None
            self._list_of_block_partitions = [
                PandasOnRayDataframePartition(
                    _DeferredGetChunk(
                        data, i, lengths[i] if lengths is not None else None
                    )
                )
                for i in range(num_splits)
            ]
        else:
            self._list_of_block_partitions = [PandasOnRayDataframePartition(data)]
        return self._list_of_block_partitions

    @property
    def list_of_ips(self):
        """
        Return the list of IP worker addresses.

        Returns
        -------
        list of str
        """
        if (ip := self._meta[self._meta_offset + 2]) is not None:
            return [ip]
        if self._list_of_block_partitions is not None:
            return [part.ip() for part in self._list_of_block_partitions]
        return []

    @property
    def _data(self) -> ObjectRefType:
        """
        Get the data wrapped by the partition.

        If the data is a `DeferredExecution`, the execution is triggered
        and the result is returned.

        Returns
        -------
        ObjectRefType
        """
        data = self._data_ref
        if isinstance(data, DeferredExecution):
            data, self._meta, self._meta_offset = data.exec()
            self._data_ref = data
        return data

    @property
    def _chunk_lengths(self) -> List[int]:
        """
        Calculate the partition chunk lengths.

        Returns
        -------
        list of int
        """
        if (
            self._chunk_lengths_cache is None
            and self._list_of_block_partitions is not None
        ):
            attr = "length" if self.axis == 0 else "width"
            self._chunk_lengths_cache = [
                getattr(p, attr)(materialize=False)
                for p in self._list_of_block_partitions
            ]
        return self._chunk_lengths_cache

    def _apply(
        self, apply_fn: Union[Callable, ObjectRefType], args: List, kwargs: Dict
    ) -> DeferredExecution:
        """
        Apply the function to this partition.

        Parameters
        ----------
        apply_fn : callable or ObjectRefType
        args : list
        kwargs : dict

        Returns
        -------
        DeferredExecution
        """
        return DeferredExecution(self._data_ref, apply_fn, args, kwargs)


class _DeferredSplit(DeferredExecution):
    """
    Split the DataFrame along the specified axis into the `num_splits` chunks.

    Parameters
    ----------
    obj : ObjectRefOrDeType
    axis : int
    num_splits : int
    lengths : list of int or None
    """

    def __init__(
        self,
        obj: ObjectRefOrDeType,
        axis: int,
        num_splits: int,
        lengths: Union[List[Union[ObjectRefType, int]], None],
    ):
        self.num_splits = num_splits
        self.skip_chunks = set()
        args = [axis, num_splits, MinPartitionSize.get(), self.skip_chunks]
        if lengths and (len(lengths) == num_splits):
            args.extend(lengths)
        super().__init__(obj, self._split, args, num_returns=num_splits)

    @remote_function
    def _split(
        df: pandas.DataFrame,
        axis: int,
        num_splits: int,
        min_chunk_len: int,
        skip_chunks: Set[int],
        *lengths: Optional[List[int]],
    ):  # pragma: no cover  # noqa: GL08
        if not lengths or (sum(lengths) != df.shape[axis]):
            length = df.shape[axis]
            chunk_len = max(math.ceil(length / num_splits), min_chunk_len)
            lengths = [chunk_len] * num_splits

        result = []
        start = 0
        for i in range(num_splits):
            if i in skip_chunks:
                result.append(None)
                start += lengths[i]
                continue

            end = start + lengths[i]
            chunk = df.iloc[start:end] if axis == 0 else df.iloc[:, start:end]
            start = end
            result.append(chunk)
            if isinstance(chunk.axes[axis], pandas.MultiIndex):
                chunk.set_axis(
                    chunk.axes[axis].remove_unused_levels(),
                    axis=axis,
                    copy=False,
                )

        return result


class _DeferredGetChunk(DeferredGetItem):
    """
    Get the chunk with the specified index from the split.

    Parameters
    ----------
    split : _DeferredSplit
    index : int
    length : int, optional
    """

    def __init__(self, split: _DeferredSplit, index: int, length: Optional[int] = None):
        super().__init__(split, index)
        self.length = length

    def __del__(self):
        """Remove this chunk from _DeferredSplit if it's not executed yet."""
        if isinstance(self.data, _DeferredSplit):
            self.data.skip_chunks.add(self.index)

    def is_next_chunk(self, other):
        """
        Check if the other chunk is the next chunk of the same split.

        Parameters
        ----------
        other : object

        Returns
        -------
        bool
        """
        return (
            isinstance(other, _DeferredGetChunk)
            and (self.data is other.data)
            and (other.index == self.index + 1)
        )


@_inherit_docstrings(PandasOnRayDataframeVirtualPartition.__init__)
class PandasOnRayDataframeColumnPartition(PandasOnRayDataframeVirtualPartition):
    axis = 0

    @remote_function
    def _remote_concat(dfs):  # pragma: no cover  # noqa: GL08
        return pandas.concat(dfs, axis=0, copy=False)


@_inherit_docstrings(PandasOnRayDataframeVirtualPartition.__init__)
class PandasOnRayDataframeRowPartition(PandasOnRayDataframeVirtualPartition):
    axis = 1

    @remote_function
    def _remote_concat(dfs):  # pragma: no cover  # noqa: GL08
        return pandas.concat(dfs, axis=1, copy=False)
