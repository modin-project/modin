set -x

# How does it work?
# - This script will run all tests in the TESTS array, assuming the
#   pytest python file is name with prefix test_; and then it will
#   upload the html result to s3 bucket.
# - `any_test_failed` variable is updated if any of the test does not
#   exit normally. This script will exit with status code from this variable.

sha_tag=$(git rev-parse --verify --short HEAD)

any_test_failed=0

TESTS=(
	"modin/pandas/test/test_dataframe" 
	"modin/pandas/test/test_concat" 
	"modin/pandas/test/test_io" 
	"modin/pandas/test/test_groupby"
	"modin/experimental/pandas/test/test_io_exp.py"
)
TESTS_FAILED=()

test_and_upload_result() {
    test_path=$1
    pytest -n auto --html=test_"$test_path".html --self-contained-html --disable-pytest-warnings --cov-config=.coveragerc --cov=modin --cov-append $test_path
    test_status=$?

    aws s3 cp test_"$test_name".html s3://modin-jenkins-result/"$sha_tag"/ --acl public-read

    if [ $test_status -ne 0 ]; then
        any_test_failed=$test_status
        TESTS_FAILED+=("$test_name")
    fi;
}

curl -s https://codecov.io/bash > codecov.sh

for test in "${TESTS[@]}"; do
    test_and_upload_result $test
done

python .jenkins/build-tests/post_comments.py --sha "$sha_tag" --tests "${TESTS_FAILED[@]}"

bash codecov.sh

exit $any_test_failed
