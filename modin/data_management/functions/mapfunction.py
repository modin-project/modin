from .function import Function


class MapFunction(Function):

    @classmethod
    def call(cls, function, **call_kwds):

        def caller(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(query_compiler._data_obj._map_partitions(lambda x: function(x, *args, **kwargs), **call_kwds))

        return caller

    @classmethod
    def register(cls, function, *args, **kwargs):
        return cls.call(function, *args, **kwargs)
