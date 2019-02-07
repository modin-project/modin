#!/usr/bin/env bash

# - Travis for lint only
#

set -e
set -x

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE:-$0}")"; pwd)
echo "PYTHON is $PYTHON"

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda install -y python==3.6.5

pip install black flake8 flake8-comprehensions
  
