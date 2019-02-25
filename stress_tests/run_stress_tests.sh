#!/usr/bin/env bash

# Show explicitly which commands are currently running.
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
RESULT_FILE=$ROOT_DIR/results-$(date '+%Y-%m-%d_%H-%M-%S').log
echo "Logging to" $RESULT_FILE
touch $RESULT_FILE

run_test(){
    local test_name=$1

    echo "Try running $test_name."
    {
        pytest -vls "$test_name.py" >> $RESULT_FILE
    } || echo "FAIL: $test_name" >> $RESULT_FILE
}

pushd "$ROOT_DIR"
    run_test test_kaggle_ipynb
popd

cat $RESULT_FILE
[ ! -s $RESULT_FILE ] || exit 1
