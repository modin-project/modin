FROM ubuntu:18.04
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

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
    conda create --name modin --yes python==3.7.6 && \
    conda activate modin && \
    pip install --no-cache-dir modin[ray] && \
    conda clean --all --yes

RUN wget https://modin-datasets.s3.amazonaws.com/trips_data.csv -O "${HOME}/trips_data.csv"

COPY nyc-taxi.py "${HOME}/nyc-taxi.py"

ENTRYPOINT ["/bin/bash", "--login", "-c", "conda run \"$@\"", "/bin/bash", "-n", "modin", "/usr/bin/env", "--"]
CMD ["python", "${HOME}/nyc-taxi.py"]
