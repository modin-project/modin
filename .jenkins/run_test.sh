set -ex

pip install awscli pytest-html

python -c "import ray; ray.init()"
pytest --html=test_dataframe.html --self-contained-html modin/pandas/test/test_dataframe.py
pytest --html=test_concat.html --self-contained-html modin/pandas/test/test_concat.py
pytest --html=test_io.html --self-contained-html modin/pandas/test/test_io.py
pytest --html=test_groupby.html --self-contained-html modin/pandas/test/test_groupby.py

sha_tag=`git rev-parse --verify --short HEAD`
aws cp test_dataframe.html s3://modin-jenkins-result/${sha_tag}/
aws cp test_concat.html s3://modin-jenkins-result/${sha_tag}/
aws cp test_io.html s3://modin-jenkins-result/${sha_tag}/
aws cp test_groupby.html s3://modin-jenkins-result/${sha_tag}/
