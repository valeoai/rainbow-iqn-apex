FROM nvidia/cuda:9.0-base as base_image

RUN apt-get update && apt-get install -y wget bzip2 build-essential tcl make tar libglib2.0-0 libsm6 libxext6 libxrender-dev

#Install Redis
RUN wget http://download.redis.io/redis-stable.tar.gz
RUN tar xzvf redis-stable.tar.gz
RUN cd redis-stable && make && make install

#Install miniconda, python3, torch and all other python dependencies
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda config --set always_yes yes
RUN conda install python=3.6

RUN conda install pytorch=0.4.1 cuda90 torchvision -c pytorch
RUN pip install atari-py redlock-py plotly opencv-python
COPY ./ ./rainbow-iqn-apex
RUN pip install -e ./rainbow-iqn-apex

WORKDIR ./rainbow-iqn-apex
