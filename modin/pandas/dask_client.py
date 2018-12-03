import dask.distributed


def _get_client():
    """Setup and get dask.distributed client"""
    # TODO add more parameters or better implement as a Client class
    return dask.distributed.Client()


try:
    from functools import lru_cache

    # the lru_cache prevents creating more clients
    @lru_cache()
    def get_client():
        return _get_client()
# Python2 does not support lru_cache
# TODO add a way to cache in Python2
except AttributeError:
    get_client = _get_client()
