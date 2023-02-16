import numpy

from .arr import array
from modin.error_message import ErrorMessage


def _dispatch_logic(operator_name):
    def call(x, *args, **kwargs):
        if not isinstance(x, array):
            ErrorMessage.single_warning(
                f"Modin NumPy only supports objects of modin.numpy.array types for add, not {type(x)}. Defaulting to NumPy."
            )
            return getattr(numpy, operator_name)(x, *args, **kwargs)
        return getattr(x, f"_{operator_name}")(*args, **kwargs)
    return call


all = _dispatch_logic("all")
any = _dispatch_logic("any")
isfinite = _dispatch_logic("isfinite")
isinf = _dispatch_logic("isinf")
isnan = _dispatch_logic("isnan")
isnat = _dispatch_logic("isnat")
isneginf = _dispatch_logic("isneginf")
isposinf = _dispatch_logic("isposinf")
iscomplex = _dispatch_logic("iscomplex")
isreal = _dispatch_logic("isreal")


def isscalar(e):
    if isinstance(e, array):
        return False
    return numpy.isscalar(e)


logical_not = _dispatch_logic("logical_not")
logical_and = _dispatch_logic("logical_and")
logical_or = _dispatch_logic("logical_or")
logical_xor = _dispatch_logic("logical_xor")
greater = _dispatch_logic("greater")
greater_equal = _dispatch_logic("greater_equal")
less = _dispatch_logic("less")
less_equal = _dispatch_logic("less_equal")
equal = _dispatch_logic("equal")
not_equal = _dispatch_logic("not_equal")
array_equal = _dispatch_logic("array_equal")
