# Create images from this container like this (in modin repo root):
#
# git rev-parse HEAD > ci/teamcity/git-rev
#
# tar cf ci/teamcity/modin.tar .
#
# docker build --build-arg ENVIRONMENT=environment-dev.yml -t modin-project/teamcity-ci:${BUILD_NUMBER} -f ci/teamcity/Dockerfile.teamcity-ci ci/teamcity

FROM rayproject/ray:latest

ARG ENVIRONMENT=environment-dev.yml

ADD modin.tar /modin
ADD git-rev /modin/git-rev

WORKDIR /modin
RUN sudo chown -R ray /modin

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Initialize conda in bash config files:
RUN conda init bash
ENV PATH /home/ray/anaconda3/envs/modin/bin:$PATH

RUN conda config --set channel_priority strict
RUN conda update python -y
RUN conda env create -f ${ENVIRONMENT}
RUN conda install curl PyGithub

# Activate the environment, and make sure it's activated:
# The following line also removed conda initialization from
# ~/.bashrc so conda starts complaining that it should be
# initialized for bash. But it is necessary to do it because
# activation is not always executed when "docker exec" is used
# and then conda initialization overwrites PATH with its base
# environment where python doesn't have any packages installed.
RUN echo "conda activate modin" > ~/.bashrc
RUN echo "Make sure environment is activated"
RUN conda list -n modin
