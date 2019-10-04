from .function import Function


class ReductionFunction(Function):
    @classmethod
    def call(cls, reduction_function, **call_kwds):
        def caller(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(
                query_compiler._modin_frame._fold_reduce(
                    call_kwds.get("axis")
                    if "axis" in call_kwds
                    else kwargs.get("axis"),
                    lambda x: reduction_function(x, *args, **kwargs),
                )
            )

        return caller
