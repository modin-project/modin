import os


def get_execution_engine():
    # In the future, when there are multiple engines and different ways of
    # backing the DataFrame, there will have to be some changed logic here to
    # decide these things. In the meantime, we will use the currently supported
    # execution engine + backing (Pandas + Ray).
    if "MODIN_ENGINE" in os.environ:
        engine = os.environ["MODIN_ENGINE"]
    else:
        engine = "Ray" if "MODIN_DEBUG" not in os.environ else "Python"
    return engine


def get_partition_format():
    # See note above about engine + backing.
    return "Pandas"


__version__ = "0.2.4"
__execution_engine__ = get_execution_engine()
__partition_format__ = get_partition_format()

# We don't want these used outside of this file.
del get_execution_engine
del get_partition_format
