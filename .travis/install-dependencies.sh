#!/usr/bin/env bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# There was a pandas + numpy installation version issue
# https://github.com/pandas-dev/pandas/issues/20775
PIP_NO_BUILD_ISOLATION=false pip install -r requirements.txt
pip install -q pytest flake8 flake8-comprehensions yapf feather-format lxml openpyxl xlrd numpy
