from typing import Any


# Technically, it would be possible to correctly type hint this function
# with a tonne of overloads, but for now, it' not worth it, just use Any
def validate_comparand(left: Any, right: Any) -> Any:
    """Validate comparand, raising if it can't be compared with `left`.

    If `left` and `right` are derived from the same dataframe, then return
    the underlying object of `right`.

    If the comparison isn't supported, return `NotImplemented` so that the
    "right-hand-side" operation (e.g. `__radd__`) can be tried.
    """
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):  # pragma: no cover
        # Technically, currently unreachable - but, keeping this in case it
        # becomes reachable in the future.
        msg = "Cannot compare different dataframe objects - please join them first"
        raise ValueError(msg)
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__column_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot compare Column with DataFrame it was not derived from."
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__dataframe_namespace__") and hasattr(
        right,
        "__scalar_namespace__",
    ):
        if right.parent_dataframe is not None and right.parent_dataframe is not left:
            msg = "Cannot compare Scalar with DataFrame it was not derived from."
            raise ValueError(msg)
        return right.scalar

    if hasattr(left, "__column_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__column_namespace__") and hasattr(right, "__column_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot compare Columns from different dataframes"
            raise ValueError(msg)
        return right.column
    if hasattr(left, "__column_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot compare Column and Scalar if they don't share the same parent dataframe"
            raise ValueError(msg)
        return right.scalar

    if hasattr(left, "__scalar_namespace__") and hasattr(
        right,
        "__dataframe_namespace__",
    ):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__column_namespace__"):
        return NotImplemented
    if hasattr(left, "__scalar_namespace__") and hasattr(right, "__scalar_namespace__"):
        if (
            right.parent_dataframe is not None
            and right.parent_dataframe is not left.parent_dataframe
        ):
            msg = "Cannot combine Scalars from different dataframes"
            raise ValueError(msg)
        return right.scalar

    return right
