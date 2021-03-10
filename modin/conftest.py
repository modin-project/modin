# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import os
import sys
import pytest
import pandas
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

import modin
import modin.config
from modin.config import IsExperimental

from modin.backends import PandasQueryCompiler, BaseQueryCompiler
from modin.engines.python.pandas_on_python.io import PandasOnPythonIO
from modin.data_management.factories import factories
from modin.utils import get_current_backend
from modin.pandas.test.utils import (
    _make_csv_file,
    get_unique_filename,
    teardown_test_files,
    NROWS,
    IO_OPS_DATA_DIR,
)

# create test data dir if it is not exists yet
if not os.path.exists(IO_OPS_DATA_DIR):
    os.mkdir(IO_OPS_DATA_DIR)


def pytest_addoption(parser):
    parser.addoption(
        "--simulate-cloud",
        action="store",
        default="off",
        help="simulate cloud for testing: off|normal|experimental",
    )
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="specifies backend to run tests on",
    )
    parser.addoption(
        "--extra-test-parameters",
        action="store_true",
        help="activate extra test parameter combinations",
    )


class Patcher:
    def __init__(self, conn, *pairs):
        self.pairs = pairs
        self.originals = None
        self.conn = conn

    def __wrap(self, func):
        def wrapper(*a, **kw):
            return func(
                *(tuple(self.conn.obtain(x) for x in a)),
                **({k: self.conn.obtain(v) for k, v in kw.items()}),
            )

        return func, wrapper

    def __enter__(self):
        self.originals = []
        for module, attrname in self.pairs:
            orig, wrapped = self.__wrap(getattr(module, attrname))
            self.originals.append((module, attrname, orig))
            setattr(module, attrname, wrapped)
        return self

    def __exit__(self, *a, **kw):
        for module, attrname, orig in self.originals:
            setattr(module, attrname, orig)


def set_experimental_env(mode):
    from modin.config import IsExperimental

    IsExperimental.put(mode == "experimental")


@pytest.fixture(scope="session", autouse=True)
def simulate_cloud(request):
    mode = request.config.getoption("--simulate-cloud").lower()
    if mode == "off":
        yield
        return

    if mode not in ("normal", "experimental"):
        raise ValueError(f"Unsupported --simulate-cloud mode: {mode}")
    assert IsExperimental.get(), "Simulated cloud must be started in experimental mode"

    from modin.experimental.cloud import create_cluster, get_connection
    import pandas._testing
    import pandas._libs.testing as cyx_testing

    with create_cluster("local", cluster_type="local"):
        get_connection().teleport(set_experimental_env)(mode)
        with Patcher(
            get_connection(),
            (pandas._testing, "assert_class_equal"),
            (cyx_testing, "assert_almost_equal"),
        ):
            yield


@pytest.fixture(scope="session", autouse=True)
def enforce_config():
    """
    A fixture that ensures that all checks for MODIN_* variables
    are done using modin.config to prevent leakage
    """
    orig_env = os.environ
    modin_start = os.path.dirname(modin.__file__)
    modin_exclude = [os.path.dirname(modin.config.__file__)]

    class PatchedEnv:
        @staticmethod
        def __check_var(name):
            if name.upper().startswith("MODIN_"):
                frame = sys._getframe()
                try:
                    # get the path to module where caller of caller is defined;
                    # caller of this function is inside PatchedEnv, and we're
                    # interested in whomever called a method on PatchedEnv
                    caller_file = frame.f_back.f_back.f_code.co_filename
                finally:
                    del frame
                pkg_name = os.path.dirname(caller_file)
                if pkg_name.startswith(modin_start):
                    assert any(
                        pkg_name.startswith(excl) for excl in modin_exclude
                    ), "Do not access MODIN_ environment variable bypassing modin.config"

        def __getitem__(self, name):
            self.__check_var(name)
            return orig_env[name]

        def __setitem__(self, name, value):
            self.__check_var(name)
            orig_env[name] = value

        def __delitem__(self, name):
            self.__check_var(name)
            del orig_env[name]

        def pop(self, name, default=object()):
            self.__check_var(name)
            return orig_env.pop(name, default)

        def get(self, name, default=None):
            self.__check_var(name)
            return orig_env.get(name, default)

        def __contains__(self, name):
            self.__check_var(name)
            return name in orig_env

        def __getattr__(self, name):
            return getattr(orig_env, name)

        def __iter__(self):
            return iter(orig_env)

    os.environ = PatchedEnv()
    yield
    os.environ = orig_env


BASE_BACKEND_NAME = "BaseOnPython"


class TestQC(BaseQueryCompiler):
    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    def finalize(self):
        self._modin_frame.finalize()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    def free(self):
        pass

    to_pandas = PandasQueryCompiler.to_pandas
    default_to_pandas = PandasQueryCompiler.default_to_pandas


class BaseOnPythonIO(PandasOnPythonIO):
    query_compiler_cls = TestQC


class BaseOnPythonFactory(factories.BaseFactory):
    @classmethod
    def prepare(cls):
        cls.io_cls = BaseOnPythonIO


def set_base_backend(name=BASE_BACKEND_NAME):
    setattr(factories, f"{name}Factory", BaseOnPythonFactory)
    modin.set_backends(engine="python", partition=name.split("On")[0])


def pytest_configure(config):
    if config.option.extra_test_parameters is not None:
        import modin.pandas.test.utils as utils

        utils.extra_test_parameters = config.option.extra_test_parameters

    backend = config.option.backend

    if backend is None:
        return

    if backend == BASE_BACKEND_NAME:
        set_base_backend(BASE_BACKEND_NAME)
    else:
        partition, engine = backend.split("On")
        modin.set_backends(engine=engine, partition=partition)


def pytest_runtest_call(item):
    custom_markers = ["xfail", "skip"]

    # dynamicly adding custom markers to tests
    for custom_marker in custom_markers:
        for marker in item.iter_markers(name=f"{custom_marker}_backends"):
            backends = marker.args[0]
            if not isinstance(backends, list):
                backends = [backends]

            current_backend = get_current_backend()
            reason = marker.kwargs.pop("reason", "")

            item.add_marker(
                getattr(pytest.mark, custom_marker)(
                    condition=current_backend in backends,
                    reason=f"Backend {current_backend} does not pass this test. {reason}",
                    **marker.kwargs,
                )
            )


@pytest.fixture(scope="class")
def TestReadCSVFixture():
    filenames = []
    files_ids = [
        "test_read_csv_regular",
        "test_read_csv_blank_lines",
        "test_read_csv_yes_no",
        "test_read_csv_nans",
        "test_read_csv_bad_lines",
    ]
    # each xdist worker spawned in separate process with separate namespace and dataset
    pytest.csvs_names = {file_id: get_unique_filename() for file_id in files_ids}
    # test_read_csv_col_handling, test_read_csv_parsing
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_read_csv_regular"],
    )
    # test_read_csv_parsing
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_read_csv_yes_no"],
        additional_col_values=["Yes", "true", "No", "false"],
    )
    # test_read_csv_col_handling
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_read_csv_blank_lines"],
        add_blank_lines=True,
    )
    # test_read_csv_nans_handling
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_read_csv_nans"],
        add_blank_lines=True,
        additional_col_values=["<NA>", "N/A", "NA", "NULL", "custom_nan", "73"],
    )
    # test_read_csv_error_handling
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_read_csv_bad_lines"],
        add_bad_lines=True,
    )

    yield
    # Delete csv files that were created
    teardown_test_files(filenames)


@pytest.fixture
def make_csv_file():
    """Pytest fixture factory that makes temp csv files for testing.
    Yields:
        Function that generates csv files
    """
    filenames = []

    yield _make_csv_file(filenames)

    # Delete csv files that were created
    teardown_test_files(filenames)


@pytest.fixture
def make_parquet_file():
    """Pytest fixture factory that makes a parquet file/dir for testing.

    Yields:
        Function that generates a parquet file/dir
    """
    filenames = []

    def _make_parquet_file(
        filename,
        row_size=NROWS,
        force=True,
        directory=False,
        partitioned_columns=[],
    ):
        """Helper function to generate parquet files/directories.

        Args:
            filename: The name of test file, that should be created.
            row_size: Number of rows for the dataframe.
            force: Create a new file/directory even if one already exists.
            directory: Create a partitioned directory using pyarrow.
            partitioned_columns: Create a partitioned directory using pandas.
            Will be ignored if directory=True.
        """
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        if os.path.exists(filename) and not force:
            pass
        elif directory:
            if os.path.exists(filename):
                shutil.rmtree(filename)
            else:
                os.mkdir(filename)
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table, root_path=filename)
        elif len(partitioned_columns) > 0:
            df.to_parquet(filename, partition_cols=partitioned_columns)
        else:
            df.to_parquet(filename)

        filenames.append(filename)

    # Return function that generates csv files
    yield _make_parquet_file

    # Delete parquet file that was created
    for path in filenames:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


@pytest.fixture
def make_sql_connection():
    """Sets up sql connections and takes them down after the caller is done.

    Yields:
        Factory that generates sql connection objects
    """
    filenames = []

    def _sql_connection(filename, table=""):
        # Remove file if exists
        if os.path.exists(filename):
            os.remove(filename)
        filenames.append(filename)
        # Create connection and, if needed, table
        conn = "sqlite:///{}".format(filename)
        if table:
            df = pandas.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4, 5, 6],
                    "col2": [7, 8, 9, 10, 11, 12, 13],
                    "col3": [14, 15, 16, 17, 18, 19, 20],
                    "col4": [21, 22, 23, 24, 25, 26, 27],
                    "col5": [0, 0, 0, 0, 0, 0, 0],
                }
            )
            df.to_sql(table, conn)
        return conn

    yield _sql_connection

    # Teardown the fixture
    teardown_test_files(filenames)


@pytest.fixture(scope="class")
def TestReadGlobCSVFixture():
    filenames = []

    base_name = get_unique_filename(extension="")
    pytest.glob_path = "{}_*.csv".format(base_name)
    pytest.files = ["{}_{}.csv".format(base_name, i) for i in range(11)]
    for fname in pytest.files:
        # Glob does not guarantee ordering so we have to remove the randomness in the generated csvs.
        _make_csv_file(filenames)(fname, row_size=11, remove_randomness=True)

    yield

    teardown_test_files(filenames)
