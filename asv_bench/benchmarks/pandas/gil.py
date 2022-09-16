from functools import wraps
import threading

import numpy as np

from modin.pandas import (
    DataFrame,
    Series,
    date_range,
    factorize,
    read_csv,
)

try:
    from modin.pandas import (
        rolling_kurt,
        rolling_max,
        rolling_mean,
        rolling_median,
        rolling_min,
        rolling_skew,
        rolling_std,
        rolling_var,
    )

    have_rolling_methods = True
except ImportError:
    have_rolling_methods = False


from .pandas_vb_common import BaseIO  # isort:skip


def test_parallel(num_threads=2, kwargs_list=None):
    """
    Decorator to run the same function multiple times in parallel.

    Parameters
    ----------
    num_threads : int, optional
        The number of times the function is run in parallel.
    kwargs_list : list of dicts, optional
        The list of kwargs to update original
        function kwargs on different threads.

    Notes
    -----
    This decorator does not pass the return value of the decorated function.

    Original from scikit-image:

    https://github.com/scikit-image/scikit-image/pull/1519

    """
    assert num_threads > 0
    has_kwargs_list = kwargs_list is not None
    if has_kwargs_list:
        assert len(kwargs_list) == num_threads

    def wrapper(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if has_kwargs_list:
                update_kwargs = lambda i: dict(kwargs, **kwargs_list[i])
            else:
                update_kwargs = lambda i: kwargs
            threads = []
            for i in range(num_threads):
                updated_kwargs = update_kwargs(i)
                thread = threading.Thread(target=func, args=args, kwargs=updated_kwargs)
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        return inner

    return wrapper


class ParallelGroupbyMethods:

    params = ([2, 4, 8], ["count", "last", "max", "mean", "min", "prod", "sum", "var"])
    param_names = ["threads", "method"]

    def setup(self, threads, method):

        N = 10**6
        ngroups = 10**3
        df = DataFrame(
            {"key": np.random.randint(0, ngroups, size=N), "data": np.random.randn(N)}
        )

        @test_parallel(num_threads=threads)
        def parallel():
            getattr(df.groupby("key")["data"], method)()

        self.parallel = parallel

        def loop():
            getattr(df.groupby("key")["data"], method)()

        self.loop = loop

    def time_parallel(self, threads, method):
        self.parallel()

    def time_loop(self, threads, method):
        for i in range(threads):
            self.loop()


class ParallelGroups:

    params = [2, 4, 8]
    param_names = ["threads"]

    def setup(self, threads):

        size = 2**22
        ngroups = 10**3
        data = Series(np.random.randint(0, ngroups, size=size))

        @test_parallel(num_threads=threads)
        def get_groups():
            data.groupby(data).groups

        self.get_groups = get_groups

    def time_get_groups(self, threads):
        self.get_groups()


class ParallelDatetimeFields:
    def setup(self):

        N = 10**6
        self.dti = date_range("1900-01-01", periods=N, freq="T")
        self.period = self.dti.to_period("D")

    def time_datetime_field_year(self):
        @test_parallel(num_threads=2)
        def run(dti):
            dti.year

        run(self.dti)

    def time_datetime_field_day(self):
        @test_parallel(num_threads=2)
        def run(dti):
            dti.day

        run(self.dti)

    def time_datetime_field_daysinmonth(self):
        @test_parallel(num_threads=2)
        def run(dti):
            dti.days_in_month

        run(self.dti)

    def time_datetime_field_normalize(self):
        @test_parallel(num_threads=2)
        def run(dti):
            dti.normalize()

        run(self.dti)

    def time_datetime_to_period(self):
        @test_parallel(num_threads=2)
        def run(dti):
            dti.to_period("S")

        run(self.dti)

    def time_period_to_datetime(self):
        @test_parallel(num_threads=2)
        def run(period):
            period.to_timestamp()

        run(self.period)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
