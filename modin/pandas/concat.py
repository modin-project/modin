from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from .dataframe import DataFrame


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
    copy=True,
    sort=False,
):
    if keys is not None:
        objs = [objs[k] for k in keys]
    else:
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
            if not isinstance(obj, (pandas.Series, pandas.DataFrame, DataFrame))
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
    all_series = all(isinstance(obj, pandas.Series) for obj in objs)
    if all_series:
        return DataFrame(
            pandas.concat(
                objs,
                axis,
                join,
                join_axes,
                ignore_index,
                keys,
                levels,
                names,
                verify_integrity,
                copy,
                sort,
            )
        )
    if isinstance(objs, dict):
        raise NotImplementedError(
            "Obj as dicts not implemented. To contribute to "
            "Modin, please visit github.com/ray-project/ray."
        )
    axis = pandas.DataFrame()._get_axis_number(axis)

    if join not in ["inner", "outer"]:
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the" " other axis"
        )
    # We have the weird Series and axis check because, when concatenating a
    # dataframe to a series on axis=0, pandas ignores the name of the series,
    # and this check aims to mirror that (possibly buggy) functionality
    objs = [
        obj
        if isinstance(obj, DataFrame)
        else DataFrame(obj.rename())
        if isinstance(obj, pandas.Series) and axis == 0
        else DataFrame(obj)
        for obj in objs
    ]
    df = objs[0]
    objs = [obj._query_compiler for obj in objs]
    new_manager = df._query_compiler.concat(
        axis,
        objs[1:],
        join=join,
        join_axes=None,
        ignore_index=False,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        copy=True,
        sort=False,
    )
    return DataFrame(query_compiler=new_manager)
