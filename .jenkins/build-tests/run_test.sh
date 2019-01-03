set -x

source activate py3

python -c "import ray; ray.init()"
pytest -n auto --html=test_dataframe.html --self-contained-html --disable-pytest-warnings modin/pandas/test/test_dataframe.py
pytest --html=test_concat.html --self-contained-html --disable-pytest-warnings modin/pandas/test/test_concat.py
pytest --html=test_io.html --self-contained-html --disable-pytest-warnings modin/pandas/test/test_io.py
pytest --html=test_groupby.html --self-contained-html --disable-pytest-warnings modin/pandas/test/test_groupby.py

sha_tag=`git rev-parse --verify --short HEAD`
aws s3 cp test_dataframe.html s3://modin-jenkins-result/${sha_tag}/ --acl public-read
aws s3 cp test_concat.html s3://modin-jenkins-result/${sha_tag}/ --acl public-read
aws s3 cp test_io.html s3://modin-jenkins-result/${sha_tag}/ --acl public-read
aws s3 cp test_groupby.html s3://modin-jenkins-result/${sha_tag}/ --acl public-read
