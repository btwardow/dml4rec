FROM continuumio/miniconda:4.7.10

ENV PATH /usr/local/cuda/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

RUN mkdir /dml4rec
WORKDIR /dml4rec
ADD environment.yml .

RUN apt-get update && apt-get install -y gcc
RUN conda env create -f environment.yml
RUN echo "source activate dml4rec" > ~/.bashrc
ENV PATH /opt/conda/envs/dml4rec/bin:$PATH


