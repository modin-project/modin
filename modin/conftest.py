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
from pandas.util._decorators import doc
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

assert (
    "modin.utils" not in sys.modules
), "Do not import modin.utils before patching, or tests could fail"
# every import under this assert has to be postfixed with 'noqa: E402'
# as flake8 complains about that... but we _have_ to make sure we
# monkey-patch at the right spot, otherwise testing doc URLs might
# not catch all of them
import modin.utils  # noqa: E402

_generated_doc_urls = set()


def _saving_make_api_url(token, _make_api_url=modin.utils._make_api_url):
    url = _make_api_url(token)
    _generated_doc_urls.add(url)
    return url


modin.utils._make_api_url = _saving_make_api_url

import modin  # noqa: E402
import modin.config  # noqa: E402
from modin.config import IsExperimental, TestRayClient  # noqa: E402
import uuid  # noqa: E402

from modin.core.storage_formats import (  # noqa: E402
    PandasQueryCompiler,
    BaseQueryCompiler,
)
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
    PandasOnPythonIO,
)
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.utils import get_current_execution  # noqa: E402
from modin.pandas.test.utils import (  # noqa: E402
    _make_csv_file,
    get_unique_filename,
    make_default_file,
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
        "--execution",
        action="store",
        default=None,
        help="specifies execution to run tests on",
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
    import modin.pandas.test.utils

    with create_cluster("local", cluster_type="local"):
        get_connection().teleport(set_experimental_env)(mode)
        with Patcher(
            get_connection(),
            (modin.pandas.test.utils, "assert_index_equal"),
            (modin.pandas.test.utils, "assert_series_equal"),
            (modin.pandas.test.utils, "assert_frame_equal"),
            (modin.pandas.test.utils, "assert_extension_array_equal"),
            (modin.pandas.test.utils, "assert_empty_frame_equal"),
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


BASE_EXECUTION_NAME = "BaseOnPython"


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

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        raise NotImplementedError(
            "The selected execution does not implement the DataFrame exchange protocol."
        )

    @classmethod
    def from_dataframe(cls, df, data_cls):
        raise NotImplementedError(
            "The selected execution does not implement the DataFrame exchange protocol."
        )

    to_pandas = PandasQueryCompiler.to_pandas
    default_to_pandas = PandasQueryCompiler.default_to_pandas


class BaseOnPythonIO(PandasOnPythonIO):
    query_compiler_cls = TestQC


class BaseOnPythonFactory(factories.BaseFactory):
    @classmethod
    def prepare(cls):
        cls.io_cls = BaseOnPythonIO


def set_base_execution(name=BASE_EXECUTION_NAME):
    setattr(factories, f"{name}Factory", BaseOnPythonFactory)
    modin.set_execution(engine="python", storage_format=name.split("On")[0])


@pytest.fixture(scope="function")
def get_unique_base_execution():
    """Setup unique execution for a single function and yield its QueryCompiler that's suitable for inplace modifications."""
    # It's better to use decimal IDs rather than hex ones due to factory names formatting
    execution_id = int(uuid.uuid4().hex, 16)
    format_name = f"Base{execution_id}"
    engine_name = "Python"
    execution_name = f"{format_name}On{engine_name}"

    # Dynamically building all the required classes to form a new execution
    base_qc = type(format_name, (TestQC,), {})
    base_io = type(
        f"{execution_name}IO", (BaseOnPythonIO,), {"query_compiler_cls": base_qc}
    )
    base_factory = type(
        f"{execution_name}Factory",
        (BaseOnPythonFactory,),
        {"prepare": classmethod(lambda cls: setattr(cls, "io_cls", base_io))},
    )

    # Setting up the new execution
    setattr(factories, f"{execution_name}Factory", base_factory)
    old_engine, old_format = modin.set_execution(
        engine=engine_name, storage_format=format_name
    )
    yield base_qc

    # Teardown the new execution
    modin.set_execution(engine=old_engine, storage_format=old_format)
    try:
        delattr(factories, f"{execution_name}Factory")
    except AttributeError:
        pass


def pytest_configure(config):
    if config.option.extra_test_parameters is not None:
        import modin.pandas.test.utils as utils

        utils.extra_test_parameters = config.option.extra_test_parameters

    execution = config.option.execution

    if execution is None:
        return

    if execution == BASE_EXECUTION_NAME:
        set_base_execution(BASE_EXECUTION_NAME)
        config.addinivalue_line(
            "filterwarnings", "default:.*defaulting to pandas.*:UserWarning"
        )
    else:
        partition, engine = execution.split("On")
        modin.set_execution(engine=engine, storage_format=partition)


def pytest_runtest_call(item):
    custom_markers = ["xfail", "skip"]

    # dynamicly adding custom markers to tests
    for custom_marker in custom_markers:
        for marker in item.iter_markers(name=f"{custom_marker}_executions"):
            executions = marker.args[0]
            if not isinstance(executions, list):
                executions = [executions]

            current_execution = get_current_execution()
            reason = marker.kwargs.pop("reason", "")

            item.add_marker(
                getattr(pytest.mark, custom_marker)(
                    condition=current_execution in executions,
                    reason=f"Execution {current_execution} does not pass this test. {reason}",
                    **marker.kwargs,
                )
            )


_doc_pytest_fixture = """
Pytest fixture factory that makes temp {file_type} files for testing.

Yields:
    Function that generates {file_type} files
"""


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
@doc(_doc_pytest_fixture, file_type="csv")
def make_csv_file():
    filenames = []

    yield _make_csv_file(filenames)

    # Delete csv files that were created
    teardown_test_files(filenames)


def create_fixture(file_type):
    @doc(_doc_pytest_fixture, file_type=file_type)
    def fixture():
        func, filenames = make_default_file(file_type=file_type)
        yield func
        teardown_test_files(filenames)

    return fixture


for file_type in ("json", "html", "excel", "feather", "stata", "hdf", "pickle", "fwf"):

    fixture = create_fixture(file_type)
    fixture.__name__ = f"make_{file_type}_file"
    globals()[fixture.__name__] = pytest.fixture(fixture)


@pytest.fixture
def make_parquet_file():
    """Pytest fixture factory that makes a parquet file/dir for testing.

    Yields:
        Function that generates a parquet file/dir
    """
    filenames = []

    def _make_parquet_file(
        filename,
        nrows=NROWS,
        ncols=2,
        force=True,
        directory=False,
        partitioned_columns=[],
    ):
        """Helper function to generate parquet files/directories.

        Args:
            filename: The name of test file, that should be created.
            nrows: Number of rows for the dataframe.
            ncols: Number of cols for the dataframe.
            force: Create a new file/directory even if one already exists.
            directory: Create a partitioned directory using pyarrow.
            partitioned_columns: Create a partitioned directory using pandas.
            Will be ignored if directory=True.
        """
        if force or not os.path.exists(filename):
            df = pandas.DataFrame(
                {f"col{x + 1}": np.arange(nrows) for x in range(ncols)}
            )
            if directory:
                if os.path.exists(filename):
                    shutil.rmtree(filename)
                else:
                    os.makedirs(filename)
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(table, root_path=filename)
            elif len(partitioned_columns) > 0:
                df.to_parquet(filename, partition_cols=partitioned_columns)
            else:
                df.to_parquet(filename)
            filenames.append(filename)

    # Return function that generates parquet files
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


@pytest.fixture
def get_generated_doc_urls():
    return lambda: _generated_doc_urls


ray_client_server = None


def pytest_sessionstart(session):
    if TestRayClient.get():
        import ray
        import ray.util.client.server.server as ray_server

        addr = "localhost:50051"
        global ray_client_server
        ray_client_server = ray_server.serve(addr)
        ray.util.connect(addr)


def pytest_sessionfinish(session, exitstatus):
    if TestRayClient.get():
        import ray

        ray.util.disconnect()
        if ray_client_server:
            ray_client_server.stop(0)
