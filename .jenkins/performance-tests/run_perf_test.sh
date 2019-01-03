set -x

source activate py3

# wget http://noaa-ghcn-pds.s3.amazonaws.com/csv/2017.csv
python -c "import ray; ray.init()"

run_once() {
    MODIN_ENGINE=dask pytest --benchmark-autosave --benchmark-compare --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
    MODIN_ENGINE=python pytest --benchmark-autosave --benchmark-compare --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
    pytest --benchmark-autosave --benchmark-compare --disable-pytest-warnings modin/pandas/test/performance-tests/test_performance.py
}

sha_tag=`git rev-parse --verify --short HEAD`
master_tag=`git rev-parse master`

git checkout "${master_tag}"
run_once
master_results=$(ls .benchmarks/*/*.json)
aws s3 cp "${master_results}" s3://modin-jenkins-result/${master_tag}-perf-${BUCKET_SUFFIX}/master/ --acl public-read

git checkout "${sha_tag}"
run_once
rm $master_results
aws s3 cp "$(ls .benchmarks/*/*.json)" s3://modin-jenkins-result/${sha_tag}-perf-${BUCKET_SUFFIX}/PR/ --acl public-read

rm -rf .benchmarks





