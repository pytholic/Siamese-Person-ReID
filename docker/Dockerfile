FROM nvcr.io/nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

COPY . /home/app/code
WORKDIR /home/app/code

RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' \
      /etc/apt/apt.conf.d/docker-clean

RUN apt-get update && apt-get install -y python3-pip git curl wget vim

RUN pip install -r requirements.txt
