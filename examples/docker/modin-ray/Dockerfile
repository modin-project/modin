# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Build image from this dockerfile like this:
# docker build -t modin-ray:latest .

FROM ubuntu:20.04

# Proxy settings
ENV http_proxy=${http_proxy}
ENV https_proxy=${https_proxy}
ENV no_proxy=${no_proxy}

RUN apt-get update --yes \
    && apt-get install wget --yes \
    && rm -rf /var/lib/apt/lists/*

ENV USER modin
ENV UID 1000
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --home $HOME \
    $USER

# Conda settings
ENV CONDA_DIR=${HOME}/miniconda
ENV CONDA_ENV_NAME=modin-ray
ENV PATH="${CONDA_DIR}/bin:${PATH}"

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh

RUN conda update -n base -c defaults conda -y \
    && conda create -n ${CONDA_ENV_NAME} --yes -c conda-forge --strict-channel-priority \
        modin-ray \
        ray-dashboard \
        scikit-learn \
        scikit-learn-intelex \
        xgboost \
    && conda clean --all --yes

# Activate ${CONDA_ENV_NAME} for interactive shells
RUN echo "source ${CONDA_DIR}/bin/activate ${CONDA_ENV_NAME}" >> "${HOME}/.bashrc"
# Activate ${CONDA_ENV_NAME} for non-interactive shells
# The following line comments out line that prevents ~/.bashrc execution in
# non-interactive mode.
RUN sed -e 's,\(^[[:space:]]\+[*]) return;;$\),# \1,' -i "${HOME}/.bashrc"
ENV BASH_ENV="${HOME}/.bashrc"

# Set up benchmark scripts
COPY nyc-taxi.py "${HOME}"
COPY census.py "${HOME}"
COPY plasticc.py "${HOME}"
RUN mkdir /dataset
WORKDIR ${HOME}

# Clean up proxy settings to publish on Docker Hub
ENV http_proxy=
ENV https_proxy=
ENV no_proxy=

# Set entrypoint with arguments expansion
ENTRYPOINT ["/bin/bash", "-c", "exec $0 $*"]
