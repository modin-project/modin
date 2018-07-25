from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import modin.pandas as pd


@pytest.fixture
def create_test_series():
    return pd.Series(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_T():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.T


@pytest.mark.skip(reason="Using pandas Series.")
def test___abs__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__abs__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___add__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__add__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___and__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__and__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___array__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__array__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___array_prepare__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__array_prepare__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___array_priority__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__array_priority__


@pytest.mark.skip(reason="Using pandas Series.")
def test___array_wrap__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__array_wrap__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___bool__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__bool__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___bytes__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__bytes__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___class__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__class__(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___contains__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__contains__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___copy__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__copy__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___deepcopy__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__deepcopy__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___delitem__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__delitem__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___div__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__div__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___divmod__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__divmod__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___doc__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__doc__


@pytest.mark.skip(reason="Using pandas Series.")
def test___eq__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__eq__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___finalize__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__finalize__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___float__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__float__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___floordiv__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__floordiv__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___ge__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__ge__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___getitem__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__getitem__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___getstate__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__getstate__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___gt__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__gt__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___iadd__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__iadd__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___imul__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__imul__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___int__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__int__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___invert__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__invert__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___ipow__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__ipow__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___isub__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__isub__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___iter__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__iter__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___itruediv__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__itruediv__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___le__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__le__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___len__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__len__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___long__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__long__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___lt__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__lt__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___mod__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__mod__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___mul__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__mul__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___ne__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__ne__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___neg__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__neg__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___nonzero__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__nonzero__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___or__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__or__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___pow__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__pow__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___repr__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__repr__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___round__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__round__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___setitem__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__setitem__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___setstate__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__setstate__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___sizeof__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__sizeof__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___str__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__str__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___sub__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__sub__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___truediv__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__truediv__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___xor__():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.__xor__(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_abs():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.abs()


@pytest.mark.skip(reason="Using pandas Series.")
def test_add():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.add(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_add_prefix():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.add_prefix(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_add_suffix():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.add_suffix(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_agg():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.agg(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_aggregate():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.aggregate(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_align():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.align(None, None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_all():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.all(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_any():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.any(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_append():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.append(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_apply():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.apply(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_argmax():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.argmax(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_argmin():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.argmin(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_argsort():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.argsort(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_as_blocks():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.as_blocks(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_as_matrix():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.as_matrix(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_asfreq():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.asfreq(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_asobject():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.asobject


@pytest.mark.skip(reason="Using pandas Series.")
def test_asof():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.asof(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_astype():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.astype(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_at():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.at(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_at_time():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.at_time(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_autocorr():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.autocorr(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_axes():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.axes


@pytest.mark.skip(reason="Using pandas Series.")
def test_base():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.base


@pytest.mark.skip(reason="Using pandas Series.")
def test_between():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.between(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_between_time():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.between_time(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_bfill():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.bfill(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_blocks():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.blocks


@pytest.mark.skip(reason="Using pandas Series.")
def test_bool():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.bool()


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.clip(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip_lower():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.clip_lower(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip_upper():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.clip_upper(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_combine():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.combine(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_combine_first():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.combine_first(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_compound():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.compound(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_compress():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.compress(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_consolidate():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.consolidate(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_convert_objects():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.convert_objects(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_copy():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.copy(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_corr():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.corr(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_count():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.count(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cov():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.cov(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cummax():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.cummax(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cummin():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.cummin(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cumprod():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.cumprod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cumsum():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.cumsum(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_data():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.data


@pytest.mark.skip(reason="Using pandas Series.")
def test_describe():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.describe(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_diff():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.diff(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_div():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.div(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_divide():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.divide(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dot():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.dot(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.drop(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop_duplicates():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.drop_duplicates(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dropna():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.dropna(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dtype():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.dtype


@pytest.mark.skip(reason="Using pandas Series.")
def test_dtypes():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.dtypes


@pytest.mark.skip(reason="Using pandas Series.")
def test_duplicated():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.duplicated(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_empty():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.empty


@pytest.mark.skip(reason="Using pandas Series.")
def test_eq():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.eq(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_equals():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.equals(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ewm():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ewm(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_expanding():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.expanding(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_factorize():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.factorize(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ffill():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ffill(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_fillna():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.fillna(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_filter():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.filter(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_first():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.first(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_first_valid_index():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.first_valid_index()


@pytest.mark.skip(reason="Using pandas Series.")
def test_flags():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.flags


@pytest.mark.skip(reason="Using pandas Series.")
def test_floordiv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.floordiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_from_array():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.from_array(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_from_csv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.from_csv(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ftype():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ftype


@pytest.mark.skip(reason="Using pandas Series.")
def test_ftypes():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ftypes


@pytest.mark.skip(reason="Using pandas Series.")
def test_ge():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ge(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.get(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_dtype_counts():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.get_dtype_counts()


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_ftype_counts():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.get_ftype_counts()


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_value():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.get_value(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_values():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.get_values()


@pytest.mark.skip(reason="Using pandas Series.")
def test_groupby():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.groupby(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_gt():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.gt(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_hasnans():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.hasnans


@pytest.mark.skip(reason="Using pandas Series.")
def test_head():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.head(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_hist():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.hist(None, None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_iat():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.iat(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_idxmax():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.idxmax(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_idxmin():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.idxmin(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_iloc():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.iloc(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_imag():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.imag


@pytest.mark.skip(reason="Using pandas Series.")
def test_index():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.index


@pytest.mark.skip(reason="Using pandas Series.")
def test_interpolate():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.interpolate(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_copy():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.is_copy


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic_decreasing():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic_decreasing


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic_increasing():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.is_monotonic_increasing


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_unique():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.is_unique


@pytest.mark.skip(reason="Using pandas Series.")
def test_isin():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.isin(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_isnull():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.isnull()


@pytest.mark.skip(reason="Using pandas Series.")
def test_item():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.item()


@pytest.mark.skip(reason="Using pandas Series.")
def test_items():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.items()


@pytest.mark.skip(reason="Using pandas Series.")
def test_itemsize():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.itemsize


@pytest.mark.skip(reason="Using pandas Series.")
def test_iteritems():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.iteritems()


@pytest.mark.skip(reason="Using pandas Series.")
def test_ix():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ix(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_keys():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.keys()


@pytest.mark.skip(reason="Using pandas Series.")
def test_kurt():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.kurt(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_kurtosis():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.kurtosis(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_last():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.last(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_last_valid_index():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.last_valid_index()


@pytest.mark.skip(reason="Using pandas Series.")
def test_le():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.le(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_loc():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.loc(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_lt():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.lt(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mad():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mad(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_map():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.map(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mask():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mask(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_max():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.max(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mean():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mean(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_median():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.median(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_memory_usage():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.memory_usage(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_min():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.min(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mod():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mode():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mode()


@pytest.mark.skip(reason="Using pandas Series.")
def test_mul():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.mul(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_multiply():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.multiply(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_name():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.name


@pytest.mark.skip(reason="Using pandas Series.")
def test_nbytes():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.nbytes


@pytest.mark.skip(reason="Using pandas Series.")
def test_ndim():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ndim


@pytest.mark.skip(reason="Using pandas Series.")
def test_ne():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ne(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nlargest():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.nlargest(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nonzero():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.nonzero()


@pytest.mark.skip(reason="Using pandas Series.")
def test_notnull():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.notnull()


@pytest.mark.skip(reason="Using pandas Series.")
def test_nsmallest():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.nsmallest(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nunique():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.nunique(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pct_change():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.pct_change(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pipe():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.pipe(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_plot():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.plot(None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None, None, None, None, None,
                        None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pop():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.pop(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pow():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.pow(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_prod():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.prod(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_product():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.product(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ptp():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ptp(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_put():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.put(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_quantile():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.quantile(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_radd():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.radd(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rank():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rank(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ravel():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.ravel(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rdiv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rdiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_real():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.real


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reindex(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex_axis():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reindex_axis(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex_like():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reindex_like(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rename():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rename(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rename_axis():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rename_axis(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reorder_levels():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reorder_levels(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_repeat():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.repeat(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_replace():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.replace(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_resample():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.resample(None, None, None, None, None, None, None, None,
                            None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reset_index():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reset_index(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reshape():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.reshape(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rfloordiv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rfloordiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rmod():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rmod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rmul():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rmul(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rolling():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rolling(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_round():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.round(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rpow():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rpow(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rsub():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rsub(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rtruediv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.rtruediv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sample():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sample(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_searchsorted():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.searchsorted(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_select():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.select(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sem():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sem(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_set_axis():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.set_axis(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_set_value():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.set_value(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_shape():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.shape


@pytest.mark.skip(reason="Using pandas Series.")
def test_shift():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.shift(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_size():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.size


@pytest.mark.skip(reason="Using pandas Series.")
def test_skew():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.skew(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_slice_shift():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.slice_shift(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sort_index():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sort_index(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sort_values():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sort_values(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sortlevel():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sortlevel(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_squeeze():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.squeeze(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_std():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.std(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_strides():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.strides


@pytest.mark.skip(reason="Using pandas Series.")
def test_sub():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sub(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_subtract():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.subtract(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sum():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.sum(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_swapaxes():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.swapaxes(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_swaplevel():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.swaplevel(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tail():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.tail(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_take():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.take(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_clipboard():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_clipboard(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_csv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_csv(None, None, None, None, None, None, None, None, None,
                          None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_dense():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_dense()


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_dict():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_dict()


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_excel():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_excel(None, None, None, None, None, None, None, None,
                            None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_frame():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_frame(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_hdf():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_hdf(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_json():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_json(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_latex():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_latex(None, None, None, None, None, None, None, None,
                            None, None, None, None, None, None, None, None,
                            None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_msgpack():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_msgpack(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_period():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_period(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_pickle():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_pickle(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_sparse():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_sparse(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_sql():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_sql(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_string():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_string(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_timestamp():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_timestamp(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_xarray():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.to_xarray()


@pytest.mark.skip(reason="Using pandas Series.")
def test_tolist():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.tolist()


@pytest.mark.skip(reason="Using pandas Series.")
def test_transform():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.transform(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_transpose():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.transpose(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_truediv():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.truediv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_truncate():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.truncate(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tshift():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.tshift(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tz_convert():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.tz_convert(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tz_localize():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.tz_localize(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_unique():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.unique()


@pytest.mark.skip(reason="Using pandas Series.")
def test_unstack():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.unstack(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_update():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.update(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_valid():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.valid(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_value_counts():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.value_counts(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_values():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.values


@pytest.mark.skip(reason="Using pandas Series.")
def test_var():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.var(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_view():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.view(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_where():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.where(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_xs():
    ray_series = create_test_series()

    with pytest.raises(NotImplementedError):
        ray_series.xs(None, None, None)
