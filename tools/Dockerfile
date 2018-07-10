FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Cheng-Yueh Liu <cyliustack@gmail.com>
RUN apt-get update && apt-get install -y git wget zip unzip vim python3 python3-pip 
RUN wget https://github.com/cyliustack/sofa/archive/master.zip && unzip master.zip && cd sofa-master && ./tools/prerequisite.sh && ./install.sh /opt/sofa && echo "source /opt/sofa/tools/activate.sh" >> ~/.bashrc 
RUN mkdir -p /tmp/program 
RUN mkdir -p /tmp/data 




