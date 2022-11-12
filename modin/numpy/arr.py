import numpy

class array(object):

    def __init__(self, object=None, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None, query_compiler=None):
        if query_compiler is not None:
            self._query_compiler = query_compiler
        else:
            arr = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like)
            pass

    def _sum(self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
        result = self._query_compiler.sum(axis=axis)
        if dtype is not None:
            result.astype(dtype)

    def __repr__(self):
        return repr(self._query_compiler.to_numpy())
