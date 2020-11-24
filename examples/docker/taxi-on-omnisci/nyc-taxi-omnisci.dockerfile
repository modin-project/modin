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
ENV MODIN_BACKEND "omnisci"
ENV MODIN_EXPERIMENTAL "true"

RUN apt-get update --yes \
    && apt-get install wget --yes && \
    rm -rf /var/lib/apt/lists/*

ENV USER modin
ENV UID 1000
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --home $HOME \
    $USER

ENV CONDA_DIR ${HOME}/miniconda

SHELL ["/bin/bash", "--login", "-c"]

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y && \
    conda create -n modin --yes --no-default-packages && \
    conda activate modin && \
    conda install -c intel/label/modin -c conda-forge modin "ray>=1.0.0" && \
    conda clean --all --yes

COPY trips_xaa.csv "${HOME}/trips_xaa.csv"
COPY nyc-taxi-omnisci.py "${HOME}/nyc-taxi-omnisci.py"

CMD ["/bin/bash", "--login", "-c", "conda activate modin && python ${HOME}/nyc-taxi-omnisci.py"]
