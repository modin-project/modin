#!/usr/bin/env bash
set -e
set -x


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
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"

elif [[ "$PYTHON" == "3.6" ]] && [[ "$platform" == "linux" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  conda install -y python==3.6.5

elif [[ "$PYTHON" == "2.7" ]] && [[ "$platform" == "macosx" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"

elif [[ "$PYTHON" == "3.6" ]] && [[ "$platform" == "macosx" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  conda install -y python==3.6.5

elif [[ "$LINT" == "1" ]]; then
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -nv
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  conda install -y python==3.6.5
  pip install black flake8 flake8-comprehensions
else
  echo "Unrecognized environment."
  exit 1
fi

pip install -r requirements.txt
# Upgrade ray to 0.6.4
if [[ "$PYTHON" == "2.7" ]]; then
  pip install https://s3-us-west-2.amazonaws.com/ray-wheels/fa8c07dd19f2f5a36b7e57a81e0364a7c556053a/ray-0.6.4-cp27-cp27mu-manylinux1_x86_64.whl
else
  pip install https://s3-us-west-2.amazonaws.com/ray-wheels/fa8c07dd19f2f5a36b7e57a81e0364a7c556053a/ray-0.6.4-cp36-cp36m-manylinux1_x86_64.whl
fi
# This causes problems with execution in the tests
pip uninstall -y numexpr
