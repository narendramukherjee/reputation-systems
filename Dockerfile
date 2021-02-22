FROM continuumio/anaconda3
LABEL maintainer="Narendra Mukherjee <narendra.mukherjee@gmail.com>"

# Install CPU version of pytorch
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other required packages
RUN pip install sbi hyperopt mlflow tqdm arviz black nb_black pyreadr

# Make a data folder which will be connected to the host
RUN mkdir /data
