import inspect

from modin.core.execution.client.query_compiler import ClientQueryCompiler
from modin.core.storage_formats.base import BaseQueryCompiler


KNOWN_MISSING = frozenset(
    [
        # conj no longer exists in pandas
        "conj",
        # no need to make service implement get_axis, which I think
        # is used internally in a few places.
        "get_axis",
        # we don't need to forward these two either
        "get_index_name",
        "get_index_names",
        # Base QC can do this for us
        "has_multiindex",
        # Base QC can do this for us
        "is_series_like",
    ]
)

print(
    "base query compiler methods that are not in client query compiler, but should be:"
)
for name, f in inspect.getmembers(ClientQueryCompiler, predicate=inspect.isfunction):
    if name in KNOWN_MISSING:
        continue
    if f == getattr(BaseQueryCompiler, name, None):
        print(name)
