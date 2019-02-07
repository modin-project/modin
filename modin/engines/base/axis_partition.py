import pandas
from modin.data_management.utils import split_result_of_axis_func_pandas


class BaseAxisPartition(object):  # pragma: no cover
    """This abstract class represents the Parent class for any
        `ColumnPartition` or `RowPartition` class. This class is intended to
        simplify the way that operations are performed

        Note 0: The procedures that use this class and its methods assume that
            they have some global knowledge about the entire axis. This may
            require the implementation to use concatenation or append on the
            list of block partitions in this object.

        Note 1: The `BaseBlockPartitions` object that controls these objects
            (through the API exposed here) has an invariant that requires that
            this object is never returned from a function. It assumes that
            there will always be `BaseRemotePartition` object stored and structures
            itself accordingly.

        The abstract methods that need implemented are `apply` and `shuffle`.
        The children classes must also implement `instance_type` and `partition_type`
        (see below).
    """

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs
    ):
        """Applies a function to a full axis.

        Note: The procedures that invoke this method assume full axis
            knowledge. Implement this method accordingly.

        Important: You must return a list of `BaseRemotePartition` objects from
            this method. See Note 1 for this class above for more information.

        Args:
            func: The function to apply. This will be preprocessed according to
                the corresponding `RemotePartitions` object.
            num_splits: The number of objects to return, the number of splits
                for the resulting object. It is up to this method to choose the
                splitting at this time.
            other_axis_partition: Another `BaseAxisPartition` object to be applied
                to func. This is for operations that are between datasets.
            maintain_partitioning: Whether or not to keep the partitioning in the same
                orientation as it was previously. This is important because we may be
                operating on an individual AxisPartition and not touching the rest.
                In this case, we have to return the partitioning to its previous
                orientation (the lengths will remain the same). This is ignored between
                two axis partitions.

        Returns:
            A list of `BaseRemotePartition` objects.
        """
        raise NotImplementedError("Must be implemented in children classes")

    def shuffle(self, func, lengths, **kwargs):
        """Shuffle the order of the data in this axis based on the `lengths`.

        Args:
            func: The function to apply before splitting.
            lengths: The list of partition lengths to split the result into.

        Returns:
            A list of RemotePartition objects split by `lengths`.
        """
        raise NotImplementedError("Must be implemented in children classes")

    # Child classes must have these in order to correctly subclass.
    instance_type = None
    partition_type = None

    def _wrap_partitions(self, partitions):
        if isinstance(partitions, self.instance_type):
            return [self.partition_type(partitions)]
        else:
            return [self.partition_type(obj) for obj in partitions]


class PandasOnXAxisPartition(BaseAxisPartition):
    """This abstract class is created to simplify and consolidate the code for
        AxisPartitions that run pandas. Because much of the code is similar, this allows
        us to reuse this code.

        Subclasses must implement `list_of_blocks` which unwraps the `RemotePartition`
        objects and creates something interpretable as a pandas DataFrame.

        See `modin.engines.ray.pandas_on_ray.axis_partition.PandasOnRayAxisPartition`
        for an example on how to override/use this class when the implementation needs
        to be augmented.
    """

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs
    ):
        """Applies func to the object in the plasma store.

        See notes in Parent class about this method.

        Args:
            func: The function to apply.
            num_splits: The number of times to split the result object.
            other_axis_partition: Another `PandasOnRayAxisPartition` object to apply to
                func with this one.
            maintain_partitioning: Whether or not to keep the partitioning in the same
                orientation as it was previously. This is important because we may be
                operating on an individual AxisPartition and not touching the rest.
                In this case, we have to return the partitioning to its previous
                orientation (the lengths will remain the same). This is ignored between
                two axis partitions.

        Returns:
            A list of `RayRemotePartition` objects.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return self._wrap_partitions(
                self.deploy_func_between_two_axis_partitions(
                    self.axis,
                    func,
                    num_splits,
                    len(self.list_of_blocks),
                    kwargs,
                    *tuple(self.list_of_blocks + other_axis_partition.list_of_blocks)
                )
            )
        args = [self.axis, func, num_splits, kwargs, maintain_partitioning]
        args.extend(self.list_of_blocks)
        return self._wrap_partitions(self.deploy_axis_func(*args))

    def shuffle(self, func, lengths, **kwargs):
        """Shuffle the order of the data in this axis based on the `lengths`.

        Extends `BaseAxisPartition.shuffle`.

        Args:
            func: The function to apply before splitting.
            lengths: The list of partition lengths to split the result into.

        Returns:
            A list of RemotePartition objects split by `lengths`.
        """
        num_splits = len(lengths)
        # We add these to kwargs and will pop them off before performing the operation.
        kwargs["manual_partition"] = True
        kwargs["_lengths"] = lengths
        args = [self.axis, func, num_splits, kwargs, False]
        args.extend(self.list_of_blocks)
        return self._wrap_partitions(self.deploy_axis_func(*args))

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        """Deploy a function along a full axis in Ray.

            Args:
                axis: The axis to perform the function along.
                func: The function to perform.
                num_splits: The number of splits to return
                    (see `split_result_of_axis_func_pandas`)
                kwargs: A dictionary of keyword arguments.
                maintain_partitioning: If True, keep the old partitioning if possible.
                    If False, create a new partition layout.
                partitions: All partitions that make up the full axis (row or column)

            Returns:
                A list of Pandas DataFrames.
            """
        # Pop these off first because they aren't expected by the function.
        manual_partition = kwargs.pop("manual_partition", False)
        lengths = kwargs.pop("_lengths", None)

        dataframe = pandas.concat(partitions, axis=axis, copy=False)
        result = func(dataframe, **kwargs)
        if isinstance(result, pandas.Series):
            if num_splits == 1:
                return result
            return [result] + [pandas.Series([]) for _ in range(num_splits - 1)]

        if manual_partition:
            # The split function is expecting a list
            lengths = list(lengths)
        # We set lengths to None so we don't use the old lengths for the resulting partition
        # layout. This is done if the number of splits is changing or we are told not to
        # keep the old partitioning.
        elif num_splits != len(partitions) or not maintain_partitioning:
            lengths = None
        else:
            if axis == 0:
                lengths = [len(part) for part in partitions]
                if sum(lengths) != len(result):
                    lengths = None
            else:
                lengths = [len(part.columns) for part in partitions]
                if sum(lengths) != len(result.columns):
                    lengths = None
        return split_result_of_axis_func_pandas(axis, num_splits, result, lengths)

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        """Deploy a function along a full axis between two data sets in Ray.

        Args:
            axis: The axis to perform the function along.
            func: The function to perform.
            num_splits: The number of splits to return
                (see `split_result_of_axis_func_pandas`).
            len_of_left: The number of values in `partitions` that belong to the
                left data set.
            kwargs: A dictionary of keyword arguments.
            partitions: All partitions that make up the full axis (row or column)
                for both data sets.

        Returns:
            A list of Pandas DataFrames.
        """
        lt_frame = pandas.concat(list(partitions[:len_of_left]), axis=axis, copy=False)
        rt_frame = pandas.concat(list(partitions[len_of_left:]), axis=axis, copy=False)

        result = func(lt_frame, rt_frame, **kwargs)
        return split_result_of_axis_func_pandas(axis, num_splits, result)
