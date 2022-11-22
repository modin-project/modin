import numpy

class array(object):

    def __init__(self, object=None, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None, query_compiler=None):
        if query_compiler is not None:
            self._query_compiler = query_compiler
        else:
            arr = numpy.array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin, like=like)
            pass

    def _absolute(self, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
        result = self._query_compiler.abs().to_numpy()
        return result

    def _add(self, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
        result = self._query_compiler.add(x2._query_compiler).to_numpy()
        return result

    def _all(self, axis=None, out=None, keepdims=None, where=None):
        result = self._query_compiler.all(axis=axis).to_numpy()
        return result

    def _subtract(self, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
        result = self._query_compiler.sub(x2._query_compiler).to_numpy()
        return result

    def _sum(self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
        result = self._query_compiler.sum(axis=axis)
        if dtype is not None:
            result = result.astype(dtype)
        if out is not None:
            out._query_compiler = result
            return
        return array(query_compiler=result)

    def __repr__(self):
        return repr(self._query_compiler.to_numpy())
