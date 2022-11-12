import numpy

def absolute(x, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if hasattr(x, "_absolute"):
        return x._absolute(out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    return numpy.absolute(x, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)

abs = absolute


def add(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if hasattr(x1, "_add"):
        return x1._add(x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    return numpy.add(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)


def all(a, axis=None, out=None, keepdims=None, where=None):
    if hasattr(a, "_all"):
        return a._all(axis=axis, out=out, keepdims=keepdims, where=where)
    return numpy.all(a, axis=axis, out=out, keepdims=keepdims, where=where)


def subtract(x1, x2, out=None, where=True, casting='same_kind', order='K', dtype=None, subok=True):
    if hasattr(x1, "_subtract"):
        return x1._subtract(x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)
    return numpy.subtract(x1, x2, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok)


def sum(arr, axis):
    if hasattr(arr, "_sum"):
        return arr._sum(axis)
    else:
        return numpy.sum(arr)
