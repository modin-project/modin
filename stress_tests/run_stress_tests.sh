#!/usr/bin/env bash

# Show explicitly which commands are currently running.
set -x

# TODO (williamma12): Once we use clusters, make sure to download latest wheels
# from s3 bucket instead of building ray
# Ray directory
RAY_DIR=${1}

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
RESULT_FILE=$ROOT_DIR/results-$(date '+%Y-%m-%d_%H-%M-%S').log
echo "Logging to" $RESULT_FILE
touch $RESULT_FILE

setup_environment(){
    pushd "$ROOT_DIR"
    # Create a virtual environment for the stress tests
    python -m virtualenv stress_tests_env >> $RESULT_FILE
    source stress_tests_env/bin/activate >> $RESULT_FILE

    # Install ray from source if available
    if [[ ! -z "$RAY_DIR" ]]; then
        pushd "$RAY_DIR"
        pip install -e . --verbose >> $RESULT_FILE
        popd
    fi

    # Install modin from source to virtual environment
    pushd "$ROOT_DIR/.."
    pip install -e . >> $RESULT_FILE
    popd

    # Install basic data science packages
    pip install matplotlib numpy seaborn scipy >> $RESULT_FILE

    # Install machine learning packages
    pip install scikit-learn xgboost lightgbm keras >> $RESULT_FILE

    # Install packages for kaggle18
    pip install nltk wordcloud plotly bokeh pyLDAvis >> $RESULT_FILE

    popd
}

teardown_environment(){
    pushd "$ROOT_DIR"
    rm -rf stress_tests_env >> $RESULT_FILE
    popd
}

run_test(){
    local test_name=$1

    echo "Try running $test_name."
    {
        pytest -vls "$test_name.py" >> $RESULT_FILE
    } || echo "FAIL: $test_name" >> $RESULT_FILE
}

pushd "$ROOT_DIR"
    setup_environment
    run_test test_kaggle_ipynb
    teardown_environment
popd

cat $RESULT_FILE
[ ! -s $RESULT_FILE ] || exit 1
