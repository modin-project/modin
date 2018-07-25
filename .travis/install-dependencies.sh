#!/usr/bin/env bash

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)

echo "PYTHON is $PYTHON"

platform="unknown"
unamestr="$(uname)"
if [[ "$unamestr" == "Linux" ]]; then
  echo "Platform is linux."
  platform="linux"
elif [[ "$unamestr" == "Darwin" ]]; then
  echo "Platform is macosx."
  platform="macosx"
else
  echo "Unrecognized platform."
  exit 1
fi

if [[ "$PYTHON" == "2.7" ]] && [[ "$platform" == "linux" ]]; then
  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  pip install -q pandas==0.22 feather-format lxml openpyxl xlrd
  # Install ray from its latest wheels
  pip install -q -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.4.0-cp27-cp27mu-manylinux1_x86_64.whl
elif [[ "$PYTHON" == "3.6" ]] && [[ "$platform" == "linux" ]]; then
  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  pip install -q pandas==0.22 feather-format lxml openpyxl xlrd
  # Install ray from its latest wheels
  pip install -q -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.4.0-cp36-cp36m-manylinux1_x86_64.whl
elif [[ "$PYTHON" == "2.7" ]] && [[ "$platform" == "macosx" ]]; then
  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  pip install -q pandas==0.22 feather-format lxml openpyxl xlrd
  # Install ray from its latest wheels
  pip install -q -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.4.0-cp27-cp27m-macosx_10_6_intel.whl
elif [[ "$PYTHON" == "3.6" ]] && [[ "$platform" == "macosx" ]]; then
  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  pip install -q pandas==0.22 feather-format lxml openpyxl xlrd
  # Install ray from its latest wheels
  pip install -q -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-0.4.0-cp36-cp36m-macosx_10_6_intel.whl
elif [[ "$LINT" == "1" ]]; then
  # Install miniconda.
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  # Install Python linting tools.
  pip install -q flake8 flake8-comprehensions yapf
else
  echo "Unrecognized environment."
  exit 1
fi
