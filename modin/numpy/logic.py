import numpy
import pandas


def _dispatch_unary(pandas_f, np_f, use_axis_arg=True):
    def call(*args, **kwargs):
        if hasattr(args[0], "_unary_op"):
            return args[0]._unary_op(pandas_f, *args[1:], use_axis_arg=use_axis_arg, **kwargs)
        return np_f
    return call


def _dispatch_binary(pandas_f, np_f):
    def call(*args, **kwargs):
        if hasattr(args[0], "_binary_op"):
            return args[0]._binary_op(pandas_f, *args[1:], **kwargs)
    return call


all = _dispatch_unary(pandas.DataFrame.all, numpy.all)
any = _dispatch_unary(pandas.DataFrame.any, numpy.any)
# use np operator for isfinite, isinf, etc. since there's no corresponding pandas function
isfinite = _dispatch_unary(numpy.isfinite, numpy.isfinite, use_axis_arg=False)
isinf = _dispatch_unary(numpy.isinf, numpy.isinf, use_axis_arg=False)
isnan = _dispatch_unary(pandas.DataFrame.isna, numpy.isnan, use_axis_arg=False)
isnat = _dispatch_unary(numpy.isnat, numpy.isnat, use_axis_arg=False)
isneginf = _dispatch_unary(numpy.isneginf, numpy.isneginf, use_axis_arg=False)
isposinf = _dispatch_unary(numpy.isposinf, numpy.isposinf, use_axis_arg=False)
iscomplex = _dispatch_unary(numpy.iscomplex, numpy.iscomplex, use_axis_arg=False)
isreal = _dispatch_unary(numpy.isreal, numpy.isreal, use_axis_arg=False)
isscalar = numpy.isscalar
# use np operator for logical operators since pandas only has bitwise ones
logical_not = _dispatch_unary(numpy.logical_not, numpy.logical_not, use_axis_arg=False)
logical_and = _dispatch_binary(numpy.logical_and, numpy.logical_and)
logical_or = _dispatch_binary(numpy.logical_or, numpy.logical_or)
logical_xor = _dispatch_binary(numpy.logical_xor, numpy.logical_xor)
greater = _dispatch_binary(pandas.DataFrame.gt, numpy.greater)
greater_equal = _dispatch_binary(pandas.DataFrame.ge, numpy.greater_equal)
less = _dispatch_binary(pandas.DataFrame.lt, numpy.less)
less_equal = _dispatch_binary(pandas.DataFrame.le, numpy.less_equal)
equal = _dispatch_binary(pandas.DataFrame.eq, numpy.equal)
not_equal = _dispatch_binary(pandas.DataFrame.ne, numpy.not_equal)


def array_equal(x1, x2):
    if hasattr(x1, "_array_equal"):
        return x1._array_equal(x2)
    return numpy.array_equal(x1, x2)
