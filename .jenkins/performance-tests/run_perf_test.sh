set -x

pip install awscli
python .jenkins/inject_aws_credentials.py

# wget http://noaa-ghcn-pds.s3.amazonaws.com/csv/2017.csv
python -c "import ray; ray.init()"
pytest --benchmark-autosave --disable-pytest-warnings modin/pandas/test/test_performance.py

sha_tag=`git rev-parse --verify --short HEAD`
# save the results to S3
# aws s3 cp .benchmark/*/*.json s3://modin-jenkins-performance-result/${sha_tag}/

# delete the .benchmarks directory
rm -rf .benchmarks
