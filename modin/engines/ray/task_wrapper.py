from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray

    @ray.remote
    def deploy_ray_func(func, args):  # pragma: no cover
        return func(**args)


class RayTask:
    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        return deploy_ray_func._remote(
            args=(func, kwargs), num_return_vals=num_return_vals
        )

    @classmethod
    def materialize(cls, obj_id):
        return ray.get(obj_id)
