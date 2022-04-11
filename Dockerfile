FROM continuumio/anaconda3

# Install utils needed for starspace
RUN apt-get update && apt-get install -y apt-utils build-essential && \
    wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.tar.gz && \
    tar -xvzf boost_1_77_0.tar.gz && \
    mv boost_1_77_0 /usr/local/bin

# Install starspace
RUN git clone https://github.com/narendramukherjee/StarSpace && cd StarSpace && make 

# Install CPU version of pytorch
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other required packages
RUN pip install sbi hyperopt mlflow tqdm arviz black nb_black pyreadr mypy

# Make a data folder which will be connected to the host
RUN mkdir /data
