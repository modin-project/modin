from pandas import plotting as pdplot

from .utils import to_pandas
from .dataframe import DataFrame


def instancer(cls):
    """This will create a dummy instance each time this is imported.

    This serves the purpose of allowing us to use all of pandas plotting methods without
        aliasing and writing each of them ourselves.
    """
    return cls()


@instancer
class Plotting(object):
    def __dir__(self):
        """This allows tab completion of plotting library"""
        return dir(pdplot)

    def __getattribute__(self, item):
        """This method will override the parameters passed and convert any Modin
            DataFrames to pandas so that they can be plotted normally
        """
        if hasattr(pdplot, item):
            func = getattr(pdplot, item)
            if callable(func):

                def wrap_func(*args, **kwargs):
                    """Convert Modin DataFrames to pandas then call the function"""
                    args = tuple(
                        arg if not isinstance(arg, DataFrame) else to_pandas(arg)
                        for arg in args
                    )
                    kwargs = {
                        kwd: val if not isinstance(val, DataFrame) else to_pandas(val)
                        for kwd, val in kwargs.items()
                    }
                    return func(*args, **kwargs)

                return wrap_func
            else:
                return func
        else:
            return object.__getattribute__(self, item)
