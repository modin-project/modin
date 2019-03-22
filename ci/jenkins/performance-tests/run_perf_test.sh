set -x

source activate py3

# wget http://noaa-ghcn-pds.s3.amazonaws.com/csv/2017.csv
python -c "import ray; ray.init()"
MODIN_ENGINE=dask pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
MODIN_ENGINE=python pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py

sha_tag=`git rev-parse --verify --short HEAD`
# save the results to S3
aws s3 cp .benchmarks/*/*.json s3://modin-jenkins-result/${sha_tag}-perf-${BUCKET_SUFFIX}/ --acl public-read

rm -rf .benchmarks
