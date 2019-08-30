import pandas
from .dataframe import DataFrame
from .series import Series


def concat(
    objs,
    axis=0,
    join="outer",
    join_axes=None,
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=None,
    copy=True,
):
    if isinstance(objs, (pandas.Series, Series, DataFrame, str, pandas.DataFrame)):
        raise TypeError(
            "first argument must be an iterable of pandas "
            "objects, you passed an object of type "
            '"{name}"'.format(name=type(objs).__name__)
        )
    axis = pandas.DataFrame()._get_axis_number(axis)
    objs = list(objs)
    if len(objs) == 0:
        raise ValueError("No objects to concatenate")

    objs = [obj for obj in objs if obj is not None]

    if len(objs) == 0:
        raise ValueError("All objects passed were None")
    try:
        type_check = next(
            obj
            for obj in objs
            if not isinstance(obj, (pandas.Series, Series, pandas.DataFrame, DataFrame))
        )
    except StopIteration:
        type_check = None
    if type_check is not None:
        raise ValueError(
            'cannot concatenate object of type "{0}"; only '
            "pandas.Series, pandas.DataFrame, "
            "and modin.pandas.DataFrame objs are "
            "valid",
            type(type_check),
        )
    all_series = all(isinstance(obj, Series) for obj in objs)
    if all_series and axis == 0:
        return Series(
            query_compiler=objs[0]._query_compiler.concat(
                axis,
                [o._query_compiler for o in objs[1:]],
                join=join,
                join_axes=None,
                ignore_index=ignore_index,
                keys=None,
                levels=None,
                names=None,
                verify_integrity=False,
                copy=True,
                sort=sort,
            )
        )
    if isinstance(objs, dict):
        raise NotImplementedError("Obj as dicts not implemented.")
    if join not in ["inner", "outer"]:
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )
    # We have the weird Series and axis check because, when concatenating a
    # dataframe to a series on axis=0, pandas ignores the name of the series,
    # and this check aims to mirror that (possibly buggy) functionality
    objs = [
        obj
        if isinstance(obj, DataFrame)
        else DataFrame(obj.rename())
        if isinstance(obj, (pandas.Series, Series)) and axis == 0
        else DataFrame(obj)
        for obj in objs
    ]
    objs = [obj._query_compiler for obj in objs if len(obj.index) or len(obj.columns)]
    if keys is not None:
        if all_series:
            new_idx = keys
        else:
            objs = [objs[i] for i in range(min(len(objs), len(keys)))]
            new_idx_labels = {
                k: v.index if axis == 0 else v.columns for k, v in zip(keys, objs)
            }
            tuples = [(k, o) for k, obj in new_idx_labels.items() for o in obj]
            new_idx = pandas.MultiIndex.from_tuples(tuples)
    else:
        new_idx = None
    new_query_compiler = objs[0].concat(
        axis,
        objs[1:],
        join=join,
        join_axes=None,
        ignore_index=ignore_index,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        copy=True,
        sort=sort,
    )
    result_df = DataFrame(query_compiler=new_query_compiler)
    if new_idx is not None:
        if axis == 0:
            result_df.index = new_idx
        else:
            result_df.columns = new_idx
    return result_df
