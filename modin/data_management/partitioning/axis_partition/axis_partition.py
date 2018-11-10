class BaseAxisPartition(object):
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

        The only abstract method needed to implement is the `apply` method.
    """

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
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

        Returns:
            A list of `BaseRemotePartition` objects.
        """
        raise NotImplementedError("Must be implemented in children classes")

    def shuffle(self, func, num_splits=None, **kwargs):
        """Shuffle the order of the data in this axis based on the `func`.

        Args:
            func:
            num_splits:
            kwargs:

        Returns:
             A list of `BaseRemotePartition` objects.
        """
        raise NotImplementedError("Must be implemented in children classes")
