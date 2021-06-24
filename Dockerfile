FROM tensorflow/tensorflow:latest-jupyter
LABEL maintainer="tnsgh0101 <tnsgh0101@gamil.com>"

EXPOSE 8888/tcp
EXPOSE 5000/tcp

RUN pip install --upgrade pip
RUN git clone https://github.com/sooooner/Human-Detection-Model.git

WORKDIR /tf/Human-Detection-Model/
RUN pip install -r requirements.txt
RUN apt-get update 
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /tf
RUN rm -r tensorflow-tutorials/


