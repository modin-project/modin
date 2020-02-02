#!/usr/bin/env bash

arg=$1
shift

if [ "$arg" == "" ]; then
    arg="all"
fi

if [ "$arg" == "ray" ] || [ "$arg" == "all" ]; then
    echo "Running Ray tests"
    aws s3 cp  --no-sign-request s3://modin-testing/testmondata_ray .
    mv testmondata_ray .testmondata
    MODIN_ENGINE=ray pytest --testmon-forceselect modin/pandas/test/ $@
    # This happens on sqlite error from testmon if too many files were changed.
    if [ "$?" -eq 3 ]; then
      rm .testmondata
      MODIN_ENGINE=ray pytest --testmon-forceselect modin/pandas/test/ $@
    fi
fi
if [ "$arg" == "python" ] || [ "$arg" == "all" ]; then
    echo "Running Python tests"
    aws s3 cp  --no-sign-request s3://modin-testing/testmondata_python .
    mv testmondata_python .testmondata
    MODIN_ENGINE=python pytest --testmon-forceselect modin/pandas/test/ $@
    # This happens on sqlite error from testmon if too many files were changed.
    if [ "$?" -eq 3 ]; then
      rm .testmondata
      MODIN_ENGINE=python pytest --testmon-forceselect modin/pandas/test/ $@
    fi
fi
if [ "$arg" == "dask" ] || [ "$arg" == "all" ]; then
    echo "Running Dask tests"
    aws s3 cp  --no-sign-request s3://modin-testing/testmondata_dask .
    mv testmondata_dask .testmondata
    MODIN_ENGINE=dask pytest --testmon-forceselect modin/pandas/test/ $@
    # This happens on sqlite error from testmon if too many files were changed.
    if [ "$?" -eq 3 ]; then
      rm .testmondata
      MODIN_ENGINE=dask pytest --testmon-forceselect modin/pandas/test/ $@
    fi
fi
if [ "$arg" == "pyarrow" ] || [ "$arg" == "all" ]; then
    echo "Running Pyarrow tests"
    aws s3 cp  --no-sign-request s3://modin-testing/testmondata_pyarrow .
    mv testmondata_pyarrow .testmondata
    MODIN_BACKEND=pyarrow MODIN_EXPERIMENTAL=1 pytest --testmon-forceselect modin/pandas/test/test_io.py::test_from_csv $@
fi
