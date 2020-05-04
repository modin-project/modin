ARG BASE_IMAGE=ray-project/deploy

FROM ${BASE_IMAGE}

ARG REQUIREMENTS=requirements.txt

ADD modin.tar /modin
ADD git-rev /modin/git-rev
WORKDIR /modin
RUN pip install -U pip setuptools
RUN pip install -U -r ${REQUIREMENTS}
RUN pip install -U pytest-remotedata
RUN pip install -e .[all]
