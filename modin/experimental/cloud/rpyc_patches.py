import numpy
from rpyc.core import netref


def apply_pathes():
    def fixed_make_method(name, doc, orig=netref._make_method):
        if name == "__array__":

            def __array__(self, dtype=None):
                # Note that protocol=-1 will only work between python
                # interpreters of the same version.
                res = netref.pickle.loads(
                    netref.syncreq(self, netref.consts.HANDLE_PICKLE, -1)
                )

                if dtype is not None:
                    res = numpy.asarray(res, dtype=dtype)

                return res

            __array__.__doc__ = doc
            return __array__
        return orig(name, doc)

    netref._make_method = fixed_make_method
