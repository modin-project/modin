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
ENV no_proxy ${no_proxy}
ENV MODIN_BACKEND "omnisci"
ENV MODIN_EXPERIMENTAL "true"

ARG conda_extra_channel
ENV add_extra_channel=${conda_extra_channel:+"-c ${conda_extra_channel}"}

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

RUN wget -nv https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda3.sh && \
    bash /tmp/miniconda3.sh -b -p "${CONDA_DIR}" -f -u && \
    "${CONDA_DIR}/bin/conda" init bash && \
    rm -f /tmp/miniconda3.sh && \
    echo ". '${CONDA_DIR}/etc/profile.d/conda.sh'" >> "${HOME}/.profile"

RUN conda update -n base -c defaults conda -y && \
    conda config --set channel_priority strict && \
    conda create -n modin --yes --no-default-packages && \
    conda activate modin && \
    conda install -c intel/label/modin -c conda-forge "numpy<1.20.0" omniscidbe4py "ray-core>=1.0" \
        "ray-autoscaler>=1.0" pandas==1.1.5 cloudpickle==1.4.1 rpyc==4.1.5 "dask>=2.12.0" && \
    conda install -c intel/label/modin -c conda-forge modin==0.8.3

RUN conda activate modin && \
    conda install -c intel/label/modin -c conda-forge -c intel ${add_extra_channel} \
        "daal4py>=2021.1" dpcpp_cpp_rt && \
    conda install -c conda-forge scikit-learn==0.23.2 xgboost && \
    conda clean --all --yes

COPY training_set.csv "${HOME}/training_set.csv"
COPY test_set_skiprows.csv "${HOME}/test_set_skiprows.csv"
COPY training_set_metadata.csv "${HOME}/training_set_metadata.csv"
COPY test_set_metadata.csv "${HOME}/test_set_metadata.csv"

COPY plasticc-omnisci.py "${HOME}/plasticc-omnisci.py"

CMD ["/bin/bash", "--login", "-c", "conda activate modin && python -u ${HOME}/plasticc-omnisci.py"]
