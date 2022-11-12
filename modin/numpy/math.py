import numpy


def sum(arr, axis):
    if hasattr(arr, "_sum"):
        return arr._sum(axis)
    else:
        return numpy.sum(arr)
