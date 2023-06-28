## You should change below region code to the region you used, here sample is use us-west-2
From 763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-training:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04 
#From pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN  apt-get update
RUN  echo "Y"|apt-get install ffmpeg
