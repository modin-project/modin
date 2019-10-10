from .function import Function


class MapFunction(Function):
    @classmethod
    def call(cls, function, *call_args, **call_kwds):
        def caller(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map(
                    lambda x: function(x, *args, **kwargs), *call_args, **call_kwds
                )
            )

        return caller
