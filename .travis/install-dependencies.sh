#!/usr/bin/env bash

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
pip install -r requirements.txt
pip install -q pytest flake8 flake8-comprehensions yapf pandas==0.22 feather-format lxml openpyxl xlrd
