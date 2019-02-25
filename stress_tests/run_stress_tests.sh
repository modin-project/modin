#!/usr/bin/env bash

# Show explicitly which commands are currently running.
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
RESULT_FILE=$ROOT_DIR/results-$(date '+%Y-%m-%d_%H-%M-%S').log
echo "Logging to" $RESULT_FILE
touch $RESULT_FILE

setup_environment(){
    pushd "$ROOT_DIR"
    # Create a virtual environment for the stress tests
    python -m virtualenv stress_tests_env
    source stress_tests_env/bin/activate

    # Install modin from source to virtual environment
    pushd "../$ROOT_DIR"
    pip install -e .
    popd

    # Install basic data science packages
    pip install matplotlib numpy seaborn scipy

    # Install machine learning packages
    pip install scikit-learn xgboost lightgbm keras

    # Install packages for kaggle18
    pip install nltk wordcloud plotly bokeh pyLDAvis

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
popd

cat $RESULT_FILE
[ ! -s $RESULT_FILE ] || exit 1
