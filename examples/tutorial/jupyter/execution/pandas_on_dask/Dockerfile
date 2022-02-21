FROM continuumio/miniconda3

COPY ./requirements.txt ./requirements.txt

RUN conda install -c conda-forge psutil setproctitle
RUN pip install -r requirements.txt

