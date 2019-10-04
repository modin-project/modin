from .function import Function


class MapReduceFunction(Function):
    @classmethod
    def call(cls, map_function, reduce_function, **call_kwds):
        def caller(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map_reduce(
                    call_kwds.get("axis")
                    if "axis" in call_kwds
                    else kwargs.get("axis"),
                    lambda x: map_function(x, *args, **kwargs),
                    lambda y: reduce_function(y, *args, **kwargs),
                )
            )

        return caller

    @classmethod
    def register(cls, map_function, reduce_function, **kwargs):
        return cls.call(map_function, reduce_function, **kwargs)
