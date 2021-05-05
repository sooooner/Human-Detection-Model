FROM tensorflow/tensorflow:latest-jupyter
MAINTAINER tnsgh0101 <tnsgh0101@gamil.com>

EXPOSE 8888/tcp

RUN pip install --upgrade pip
RUN git clone https://github.com/sooooner/Faster-RCNN.git

WORKDIR /tf/Faster-RCNN/
RUN pip install -r requirements.txt
RUN apt-get -y install libgl1-mesa-glx

WORKDIR /tf
RUN rm -r tensorflow-tutorials/


