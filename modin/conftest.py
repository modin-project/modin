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

# We turn off mypy type checks in this file because it's not imported anywhere
# type: ignore

import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional

import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc

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

import uuid  # noqa: E402

import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
    AsyncReadMode,
    BenchmarkMode,
    GithubCI,
    IsExperimental,
    MinRowPartitionSize,
    NPartitions,
)
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
    PandasOnPythonIO,
)
from modin.core.storage_formats import (  # noqa: E402
    BaseQueryCompiler,
    PandasQueryCompiler,
)
from modin.tests.pandas.utils import (  # noqa: E402
    NROWS,
    _make_csv_file,
    get_unique_filename,
    make_default_file,
)


def pytest_addoption(parser):
    parser.addoption(
        "--execution",
        action="store",
        default=None,
        help="specifies execution to run tests on",
    )


def set_experimental_env(mode):
    IsExperimental.put(mode == "experimental")


@pytest.fixture(scope="session", autouse=True)
def enforce_config():
    """
    A fixture that ensures that all checks for MODIN_* variables
    are done using modin.config to prevent leakage
    """
    orig_env = os.environ
    modin_start = os.path.dirname(modin.__file__)
    modin_exclude = [
        os.path.dirname(modin.config.__file__),
        os.path.dirname(modin.tests.config.__file__),
    ]

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

    storage_format = property(
        lambda self: "Base", doc=BaseQueryCompiler.storage_format.__doc__
    )
    engine = property(lambda self: "Python", doc=BaseQueryCompiler.engine.__doc__)

    def finalize(self):
        self._modin_frame.finalize()

    def execute(self):
        self.finalize()
        self._modin_frame.wait_computations()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    def free(self):
        pass

    def to_interchange_dataframe(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ):
        raise NotImplementedError(
            "The selected execution does not implement the DataFrame exchange protocol."
        )

    @classmethod
    def from_interchange_dataframe(cls, df, data_cls):
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

            current_execution = modin.utils.get_current_execution()
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
def TestReadCSVFixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("TestReadCSVFixture")

    creator = _make_csv_file(data_dir=tmp_path)
    # each xdist worker spawned in separate process with separate namespace and dataset
    pytest.csvs_names = {}
    # test_read_csv_col_handling, test_read_csv_parsing
    pytest.csvs_names["test_read_csv_regular"] = creator()
    # test_read_csv_parsing
    pytest.csvs_names["test_read_csv_yes_no"] = creator(
        additional_col_values=["Yes", "true", "No", "false"],
    )
    # test_read_csv_col_handling
    pytest.csvs_names["test_read_csv_blank_lines"] = creator(
        add_blank_lines=True,
    )
    # test_read_csv_nans_handling
    pytest.csvs_names["test_read_csv_nans"] = creator(
        add_blank_lines=True,
        additional_col_values=["<NA>", "N/A", "NA", "NULL", "custom_nan", "73"],
    )
    # test_read_csv_error_handling
    pytest.csvs_names["test_read_csv_bad_lines"] = creator(
        add_bad_lines=True,
    )
    yield


@pytest.fixture
@doc(_doc_pytest_fixture, file_type="csv")
def make_csv_file(tmp_path):
    yield _make_csv_file(data_dir=tmp_path)


def create_fixture(file_type):
    @doc(_doc_pytest_fixture, file_type=file_type)
    def fixture(tmp_path):
        yield make_default_file(file_type=file_type, data_dir=tmp_path)

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
        range_index_start=0,
        range_index_step=1,
        range_index_name=None,
        partitioned_columns=[],
        row_group_size: Optional[int] = None,
    ):
        """Helper function to generate parquet files/directories.

        Args:
            filename: The name of test file, that should be created.
            nrows: Number of rows for the dataframe.
            ncols: Number of cols for the dataframe.
            force: Create a new file/directory even if one already exists.
            partitioned_columns: Create a partitioned directory using pandas.
            row_group_size: Maximum size of each row group.
        """
        if force or not os.path.exists(filename):
            df = pandas.DataFrame(
                {f"col{x + 1}": np.arange(nrows) for x in range(ncols)}
            )
            index = pandas.RangeIndex(
                start=range_index_start,
                stop=range_index_start + (nrows * range_index_step),
                step=range_index_step,
                name=range_index_name,
            )
            if (
                range_index_start == 0
                and range_index_step == 1
                and range_index_name is None
            ):
                assert df.index.equals(index)
            else:
                df.index = index
            if len(partitioned_columns) > 0:
                df.to_parquet(
                    filename,
                    partition_cols=partitioned_columns,
                    row_group_size=row_group_size,
                )
            else:
                df.to_parquet(filename, row_group_size=row_group_size)
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

    def _sql_connection(filename, table=""):
        # Remove file if exists
        if os.path.exists(filename):
            os.remove(filename)
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


@pytest.fixture(scope="class")
def TestReadGlobCSVFixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("TestReadGlobCSVFixture")

    base_name = get_unique_filename(extension="")
    pytest.glob_path = str(tmp_path / "{}_*.csv".format(base_name))
    pytest.files = [str(tmp_path / "{}_{}.csv".format(base_name, i)) for i in range(11)]
    for fname in pytest.files:
        # Glob does not guarantee ordering so we have to remove the randomness in the generated csvs.
        _make_csv_file(data_dir=tmp_path)(fname, row_size=11, remove_randomness=True)

    yield


@pytest.fixture
def get_generated_doc_urls():
    return lambda: _generated_doc_urls


@pytest.fixture
def set_num_partitions(request):
    old_num_partitions = NPartitions.get()
    NPartitions.put(request.param)
    yield
    NPartitions.put(old_num_partitions)


@pytest.fixture()
def set_benchmark_mode(request):
    old_benchmark_mode = BenchmarkMode.get()
    BenchmarkMode.put(request.param)
    yield
    BenchmarkMode.put(old_benchmark_mode)


@pytest.fixture
def set_async_read_mode(request):
    old_async_read_mode = AsyncReadMode.get()
    AsyncReadMode.put(request.param)
    yield
    AsyncReadMode.put(old_async_read_mode)


@pytest.fixture
def set_min_row_partition_size(request):
    old_min_row_partition_size = MinRowPartitionSize.get()
    MinRowPartitionSize.put(request.param)
    yield
    MinRowPartitionSize.put(old_min_row_partition_size)


ray_client_server = None


@pytest.fixture
def s3_storage_options(worker_id):
    # # copied from pandas conftest.py:
    # https://github.com/pandas-dev/pandas/blob/32f789fbc5d5a72d9d1ac14935635289eeac9009/pandas/tests/io/conftest.py#L45
    # worker_id is a pytest fixture
    if GithubCI.get():
        url = "http://localhost:5000/"
    else:
        # If we hit this else-case, this test is being run locally. In that case, we want
        # each worker to point to a different port for its mock S3 service. The easiest way
        # to do that is to use the `worker_id`, which is unique, to determine what port to point
        # to. We arbitrarily assign `5` as a worker id to the master worker, since we need a number
        # for each worker, and we never run tests with more than `pytest -n 4`.
        worker_id = "5" if worker_id == "master" else worker_id.lstrip("gw")
        url = f"http://127.0.0.1:555{worker_id}/"
    return {"client_kwargs": {"endpoint_url": url}}


@pytest.fixture(scope="session")
def monkeysession():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="session")
def s3_base(worker_id, monkeysession):
    """
    Fixture for mocking S3 interaction.

    Sets up moto server in separate process locally.

    Yields
    ------
    str
        URL for motoserver/moto CI service.
    """
    # copied from pandas conftest.py
    # still need access keys for https://github.com/getmoto/moto/issues/1924
    monkeysession.setenv("AWS_ACCESS_KEY_ID", "foobar_key")
    monkeysession.setenv("AWS_SECRET_ACCESS_KEY", "foobar_secret")
    monkeysession.setenv("AWS_REGION", "us-west-2")
    if GithubCI.get():
        if sys.platform in ("darwin", "win32", "cygwin") or (
            platform.machine() in ("arm64", "aarch64")
            or platform.machine().startswith("armv")
        ):
            # pandas comments say:
            # DO NOT RUN on Windows/macOS/ARM, only Ubuntu
            # - subprocess in CI can cause timeouts
            # - GitHub Actions do not support
            #   container services for the above OSs
            pytest.skip(
                "S3 tests do not have a corresponding service in Windows, macOS "
                + "or ARM platforms"
            )
        else:
            # assume CI has started moto in docker container:
            # https://docs.getmoto.org/en/latest/docs/server_mode.html#run-using-docker
            # It would be nice to start moto on another thread as in the
            # instructions here:
            # https://docs.getmoto.org/en/latest/docs/server_mode.html#start-within-python
            # but that gives 403 forbidden error when we try to create the bucket
            yield "http://localhost:5000"
    else:
        # Launching moto in server mode, i.e., as a separate process
        # with an S3 endpoint on localhost

        # If we hit this else-case, this test is being run locally. In that case, we want
        # each worker to point to a different port for its mock S3 service. The easiest way
        # to do that is to use the `worker_id`, which is unique, to determine what port to point
        # to.
        endpoint_port = (
            5500 if worker_id == "master" else (5550 + int(worker_id.lstrip("gw")))
        )
        endpoint_uri = f"http://127.0.0.1:{endpoint_port}/"

        # pipe to null to avoid logging in terminal
        # TODO any way to throw the error from here? e.g. i had an annoying problem
        # where I didn't have flask-cors and moto just failed .if there's an error
        # in the popen command and we throw an error within the body of the context
        # manager, the test just hangs forever.
        with subprocess.Popen(
            ["moto_server", "s3", "-p", str(endpoint_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        ) as proc:
            for _ in range(50):
                try:
                    # OK to go once server is accepting connections
                    if requests.get(endpoint_uri).ok:
                        break
                except Exception:
                    # try again while we still have retries
                    time.sleep(0.1)
            else:
                proc.terminate()
                _, errs = proc.communicate()
                raise RuntimeError(
                    "Could not connect to moto server after 50 tries. "
                    + f"See stderr for extra info: {errs}"
                )
            yield endpoint_uri

            proc.terminate()


@pytest.fixture
def s3_resource(s3_base):
    """
    Set up S3 bucket with contents. The primary bucket name is "modin-test".

    When running locally, this function should be safe even if there are multiple pytest
    workers running in parallel because each worker gets its own endpoint. When running
    in CI, we use a single endpoint for all workers, so we can't have multiple pytest
    workers running in parallel.
    """
    bucket = "modin-test"
    conn = boto3.resource("s3", endpoint_url=s3_base)
    cli = boto3.client("s3", endpoint_url=s3_base)

    # https://github.com/getmoto/moto/issues/3292
    # without location, I get
    # botocore.exceptions.ClientError: An error occurred
    # (IllegalLocationConstraintException) when calling the CreateBucket operation:
    # The unspecified location constraint is incompatible for the region specific
    # endpoint this request was sent to.
    # even if I delete os.environ['AWS_REGION'] but somehow pandas can get away with
    # this.
    try:
        cli.create_bucket(
            Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
        )
    except Exception as e:
        # OK if bucket already exists, but want to raise other exceptions.
        # The exception raised by `create_bucket` is made using a factory,
        # so we need to check using this method of reading the response rather
        # than just checking the type of the exception.
        response = getattr(e, "response", {})
        error_code = response.get("Error", {}).get("Code", "")
        if error_code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            raise
    for _ in range(20):
        # We want to wait until bucket creation is finished.
        if cli.list_buckets()["Buckets"]:
            break
        time.sleep(0.1)
    if not cli.list_buckets()["Buckets"]:
        raise RuntimeError("Could not create bucket")

    s3fs.S3FileSystem.clear_instance_cache()

    s3 = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_base})

    test_s3_files = [
        ("modin-bugs/multiple_csv/", "modin/tests/pandas/data/multiple_csv/"),
        (
            "modin-bugs/test_data_dir.parquet/",
            "modin/tests/pandas/data/test_data_dir.parquet/",
        ),
        ("modin-bugs/test_data.parquet", "modin/tests/pandas/data/test_data.parquet"),
        ("modin-bugs/test_data.json", "modin/tests/pandas/data/test_data.json"),
        ("modin-bugs/test_data.fwf", "modin/tests/pandas/data/test_data.fwf"),
        ("modin-bugs/test_data.feather", "modin/tests/pandas/data/test_data.feather"),
        ("modin-bugs/issue5159.parquet/", "modin/tests/pandas/data/issue5159.parquet/"),
    ]
    for s3_key, file_name in test_s3_files:
        s3.put(file_name, f"{bucket}/{s3_key}", recursive=s3_key.endswith("/"))

    yield conn

    s3.rm(bucket, recursive=True)
    for _ in range(20):
        # We want to wait until the deletion finishes.
        if not cli.list_buckets()["Buckets"]:
            break
        time.sleep(0.1)


@pytest.fixture
def modify_config(request):
    values = request.param
    old_values = {}

    for key, value in values.items():
        old_values[key] = key.get()
        key.put(value)

    yield  # waiting for the test to be completed
    # restoring old parameters
    for key, value in old_values.items():
        try:
            key.put(value)
        except ValueError as e:
            # sometimes bool env variables have 'None' as a default value, which
            # causes a ValueError when we try to set this value back, as technically,
            # only bool values are allowed (and 'None' is not a bool), in this case
            # we try to set 'False' instead
            if key.type == bool and value is None:
                key.put(False)
            else:
                raise e
