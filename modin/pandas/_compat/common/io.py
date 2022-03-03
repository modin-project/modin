import pandas

from modin.utils import Engine
from ... import _update_engine, DataFrame


def _read(**kwargs):
    """
    Read csv file from local disk.
    Parameters
    ----------
    **kwargs : dict
        Keyword arguments in pandas.read_csv.
    Returns
    -------
    modin.pandas.DataFrame
    """
    Engine.subscribe(_update_engine)
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    squeeze = kwargs.pop("squeeze", False)
    pd_obj = FactoryDispatcher.read_csv(**kwargs)
    # This happens when `read_csv` returns a TextFileReader object for iterating through
    if isinstance(pd_obj, pandas.io.parsers.TextFileReader):
        reader = pd_obj.read
        pd_obj.read = lambda *args, **kwargs: DataFrame(
            query_compiler=reader(*args, **kwargs)
        )
        return pd_obj
    result = DataFrame(query_compiler=pd_obj)
    if squeeze:
        return result.squeeze(axis=1)
    return result
