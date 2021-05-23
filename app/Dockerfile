FROM jupyter/minimal-notebook:python-3.9.2

USER root
LABEL maintainer="Vincent Gauthier <vincent.gauthier99@gmail.com>"

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get clean

RUN pip3 install --upgrade pip
RUN pip3 install jupyterlab

COPY requirements.txt .
RUN pip3 install -r requirements.txt

EXPOSE 8888
# pydebug
EXPOSE 5678

COPY . ./

CMD jupyter-lab --ip=0.0.0.0 --no-browser --allow-root
