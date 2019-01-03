set -x

source activate py3

# wget http://noaa-ghcn-pds.s3.amazonaws.com/csv/2017.csv
python -c "import ray; ray.init()"

run_once() {
    MODIN_ENGINE=dask pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
    MODIN_ENGINE=python pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
    pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
}

sha_tag=`git rev-parse --verify --short HEAD`
run_once
# save the results to S3
aws s3 cp "$(ls .benchmarks/*/*.json)" s3://modin-jenkins-result/${sha_tag}-perf-${BUCKET_SUFFIX}/PR/ --acl public-read
rm -rf .benchmarks

master_tag=`git rev-parse master`
git checkout "${master_tag}"
run_once
# save the results to S3
aws s3 cp "$(ls .benchmarks/*/*.json)" s3://modin-jenkins-result/${master_tag}-perf-${BUCKET_SUFFIX}/master/ --acl public-read
rm -rf .benchmarks


