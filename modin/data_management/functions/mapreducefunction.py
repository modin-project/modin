from .function import Function


class MapReduceFunction(Function):
    @classmethod
    def call(cls, map_function, reduce_function, *call_args, **call_kwds):
        def caller(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map_reduce(
                    call_kwds.get("axis", None) or kwargs.get("axis"),
                    lambda x: map_function(x, *args, **kwargs),
                    lambda y: reduce_function(y, *args, **kwargs),
                    *call_args,
                    **call_kwds
                )
            )

        return caller

    @classmethod
    def register(cls, function, *args, **kwargs):
        return cls.call(function, *args, **kwargs)
