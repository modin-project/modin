import os
import warnings


def custom_formatwarning(msg, category, *args, **kwargs):
    # ignore everything except the message
    return "{}: {}\n".format(category.__name__, msg)


warnings.formatwarning = custom_formatwarning
# Filter numpy version warnings because they are not relevant
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def get_execution_engine():
    # In the future, when there are multiple engines and different ways of
    # backing the DataFrame, there will have to be some changed logic here to
    # decide these things. In the meantime, we will use the currently supported
    # execution engine + backing (Pandas + Ray).
    if "MODIN_ENGINE" in os.environ:
        # .title allows variants like ray, RAY, Ray
        engine = os.environ["MODIN_ENGINE"].title()
    else:
        if "MODIN_DEBUG" in os.environ:
            engine = "Python"
        else:
            try:
                import ray

                engine = "Ray"
            except ImportError:
                try:
                    import dask
                    import distributed

                    engine = "Dask"
                except ImportError:
                    raise ImportError(
                        "Please `pip install modin[ray] or modin[dask] to install an engine"
                    )
                else:
                    if (
                        str(dask.__version__) < "2.1.0"
                        or str(distributed.__version__) < "2.3.2"
                    ):
                        raise ImportError(
                            "Please `pip install modin[dask] to install compatible Dask version."
                        )
            else:
                if ray.__version__ != "0.7.3":
                    raise ImportError(
                        "Please `pip install modin[ray] to install compatible Ray version."
                    )
    return engine


def get_partition_format():
    # See note above about engine + backing.
    return os.environ.get("MODIN_BACKEND", "Pandas").title()


__version__ = "0.6.1"
__execution_engine__ = get_execution_engine()
__partition_format__ = get_partition_format()

# We don't want these used outside of this file.
del get_execution_engine
del get_partition_format
