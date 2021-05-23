FROM tensorflow/tensorflow:latest-gpu-jupyter
# FROM python:3.8.3-slim-buster

LABEL maintainer="Vincent Gauthier <vincent.gauthier99@gmail.com>"

RUN pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

WORKDIR /aux
# # WORKDIR /usr/src/app/ 

# # set environment varibles
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir fastapi

# install system dependencies
RUN apt-get update && apt-get clean

# install python dependencies
COPY ./requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
RUN pip install torch torchvision torchaudio
# # RUN pip install git+https://github.com/tortoise/tortoise-orm.git
# RUN python -m spacy download en_core_web_sm

COPY . ./

# # ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
