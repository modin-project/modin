import functools

import dask.distributed


# the lru_cache prevents creating more clients
@functools.lru_cache()
def get_client():
    """Setup and get dask.distributed client"""
    # TODO add more parameters or better implement as a Client class
    return dask.distributed.Client()
