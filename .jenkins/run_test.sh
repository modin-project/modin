set -ex
python -c "import ray; ray.init()"
python -m pytest --disable-pytest-warnings modin/pandas/test/test_dataframe.py
python -m pytest --disable-pytest-warnings modin/pandas/test/test_concat.py
python -m pytest --disable-pytest-warnings modin/pandas/test/test_io.py
python -m pytest --disable-pytest-warnings modin/pandas/test/test_groupby.py
