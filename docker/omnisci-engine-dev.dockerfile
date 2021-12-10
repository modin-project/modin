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

FROM ubuntu:18.04
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

RUN apt-get update --yes \
    && apt-get install wget git --yes \
    #
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR ${HOME}/miniconda

SHELL ["/bin/bash", "--login", "-c"]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh \
    && bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u \
    && "${CONDA_DIR}/bin/conda" init bash \
    && rm -f /tmp/miniconda3.sh \
    && echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

# define `gh_username` can be useful in case of using modin fork
ARG gh_username=modin-project
ARG modin_dir="${HOME}/modin"

# Clone modin repo
RUN mkdir "$modin_dir" \
    && git clone "https://github.com/$gh_username/modin.git" "$modin_dir" \
    && cd "$modin_dir" \
    && git remote add upstream "https://github.com/modin-project/modin.git"

# install modin dependencies
RUN conda env create -n modin -f "$modin_dir/requirements/env_omnisci.yml"

# install modin
RUN cd "$modin_dir" \
    && conda activate modin \
    && pip install -e . --no-deps

# setup environments for modin on omnisci engine work
ENV MODIN_ENGINE "native"
ENV MODIN_STORAGE_FORMAT "omnisci"
ENV MODIN_EXPERIMENTAL "true"

# To work properly, run the following command in the container:
# conda activate modin
WORKDIR $modin_dir
