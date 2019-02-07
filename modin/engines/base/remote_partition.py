class BaseRemotePartition(object):  # pragma: no cover
    """This abstract class holds the data and metadata for a single partition.
        The methods required for implementing this abstract class are listed in
        the section immediately following this.

        The API exposed by the children of this object is used in
        `BaseBlockPartitions`.

        Note: These objects are treated as immutable by `BaseBlockPartitions`
        subclasses. There is no logic for updating inplace.
    """

    # Abstract methods and fields. These must be implemented in order to
    # properly subclass this object. There are also some abstract classmethods
    # to implement.
    def get(self):
        """Return the object wrapped by this one to the original format.

        Note: This is the opposite of the classmethod `put`.
            E.g. if you assign `x = BaseRemotePartition.put(1)`, `x.get()` should
            always return 1.

        Returns:
            The object that was `put`.
        """
        raise NotImplementedError("Must be implemented in child class")

    def apply(self, func, **kwargs):
        """Apply some callable function to the data in this partition.

        Note: It is up to the implementation how kwargs are handled. They are
            an important part of many implementations. As of right now, they
            are not serialized.

        Args:
            func: The lambda to apply (may already be correctly formatted)

        Returns:
             A new `BaseRemotePartition` containing the object that has had `func`
             applied to it.
        """
        raise NotImplementedError("Must be implemented in child class")

    def add_to_apply_calls(self, func, **kwargs):
        """Add the function to the apply function call stack.

        This function will be executed when apply is called. It will be executed
        in the order inserted; apply's func operates the last and return
        """
        raise NotImplementedError("Must be implemented in child class")

    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Note: If the underlying object is a Pandas DataFrame, this will likely
            only need to call `get`

        Returns:
            A Pandas DataFrame.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def put(cls, obj):
        """A factory classmethod to format a given object.

        Args:
            obj: An object.

        Returns:
            A `RemotePartitions` object.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def preprocess_func(cls, func):
        """Preprocess a function before an `apply` call.

        Note: This is a classmethod because the definition of how to preprocess
            should be class-wide. Also, we may want to use this before we
            deploy a preprocessed function to multiple `BaseRemotePartition`
            objects.

        Args:
            func: The function to preprocess.

        Returns:
            An object that can be accepted by `apply`.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def length_extraction_fn(cls):
        """The function to compute the length of the object in this partition.

        Returns:
            A callable function.
        """
        raise NotImplementedError("Must be implemented in child class")

    @classmethod
    def width_extraction_fn(cls):
        """The function to compute the width of the object in this partition.

        Returns:
            A callable function.
        """
        raise NotImplementedError("Must be implemented in child class")

    _length_cache = None
    _width_cache = None

    def length(self):
        if self._length_cache is None:
            cls = type(self)
            func = cls.length_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)

            self._length_cache = self.apply(preprocessed_func)
        return self._length_cache

    def width(self):
        if self._width_cache is None:
            cls = type(self)
            func = cls.width_extraction_fn()
            preprocessed_func = cls.preprocess_func(func)

            self._width_cache = self.apply(preprocessed_func)
        return self._width_cache

    @classmethod
    def empty(cls):
        raise NotImplementedError("To be implemented in the child class!")
