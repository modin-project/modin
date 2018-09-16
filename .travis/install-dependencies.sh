#!/usr/bin/env bash

if [[ "$PYTHON" == "2.7" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"

elif [[ "$PYTHON" == "3.6" ]] || [[ "$LINT" == "1" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  conda install -y python==3.6.5
fi

pip install -r requirements.txt
pip install -q pytest flake8 flake8-comprehensions yapf feather-format lxml openpyxl xlrd numpy
