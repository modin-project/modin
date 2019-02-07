set -x

# How does it work?
# - This script will run all tests in the TESTS array, assuming the
#   pytest python file is name with prefix test_; and then it will
#   upload the html result to s3 bucket.
# - `any_test_failed` variable is updated if any of the test does not
#   exit normally. This script will exit with status code from this variable.

sha_tag=$(git rev-parse --verify --short HEAD)

any_test_failed=0

TESTS=("dataframe" "concat" "io" "groupby")
TESTS_FAILED=()

test_and_upload_result() {
    test_name=$1
    pytest --html=test_"$test_name".html --self-contained-html --disable-pytest-warnings modin/pandas/test/test_"$test_name".py
    test_status=$?

    aws s3 cp test_"$test_name".html s3://modin-jenkins-result/"$sha_tag"/ --acl public-read

    if [ $test_status -ne 0 ]; then
        any_test_failed=$test_status
        TESTS_FAILED+=("$test_name")
    fi;
}

for test in "${TESTS[@]}"; do
    test_and_upload_result $test
done

python .jenkins/build-tests/post_comments.py --sha "$sha_tag" --tests "${TESTS_FAILED[@]}"

exit $any_test_failed