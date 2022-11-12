import numpy

class array(object):

    def __init__(self, object, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None, query_compiler=None):
        if query_compiler is not None:
            pass
        arr = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndim=ndim, like=like)
        pass

    def _sum(self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
        return self._query_compiler.sum(axis=axis)

