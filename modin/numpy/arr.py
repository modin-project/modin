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

    def _get_shape(self):
        return (len(self._query_compiler.index), len(self._query_compiler.columns))   
    
    def _set_shape(self, new_shape):
        if not (isinstance(new_shape, int)) and not isinstance(new_shape, tuple):
            raise TypeError(f"expected a sequence of integers or a single integer, got '{new_shape}'")
        elif isinstance(new_shape, tuple):
            for dim in new_shape:
                if not isinstance(dim, int):
                    raise TypeError(f"'{type(dim)}' object cannot be interpreted as an integer")
        from math import prod
        new_dimensions = new_shape if isinstance(new_shape, int) else prod(new_shape)
        if new_dimensions != prod(self._get_shape()):
            raise ValueError(f"cannot reshape array of size {prod(self._get_shape)} into {new_shape if isinstance(new_shape, tuple) else (new_shape,)}")
        if isinstance(new_shape, int):
            qcs = []
            for index_val in self._query_compiler.index[1:]:
                qcs.append(self._query_compiler.getitem_row_array([index_val]).reset_index(drop=True))
            self._query_compiler = self._query_compiler.getitem_row_array([self._query_compiler.index[0]]).reset_index(drop=True).concat(1, qcs, ignore_index=True)
        else:
            raise NotImplementedError("Reshaping from a 2D object to a 2D object is not currently supported!")
    
    shape = property(_get_shape, _set_shape)
            

    def __repr__(self):
        return repr(self._query_compiler.to_numpy())
