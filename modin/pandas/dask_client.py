import dask.distributed


# TODO add a way to only initalize Client one time
# Note: functools.lru_cache() is not supported in Python2
def get_client():
    """Setup and get dask.distributed client"""
    # TODO add more parameters or better implement as a Client class
    return dask.distributed.Client()
