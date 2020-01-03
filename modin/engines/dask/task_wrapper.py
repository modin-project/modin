from modin import __execution_engine__

if __execution_engine__ == "Dask":
    from distributed.client import _get_global_client


class DaskTask:
    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        client = _get_global_client()
        remote_task_future = client.submit(func, **kwargs)
        return [
            client.submit(lambda l, i: l[i], remote_task_future, i)
            for i in range(num_return_vals)
        ]

    @classmethod
    def materialize(cls, future):
        client = _get_global_client()
        return client.gather(future)
