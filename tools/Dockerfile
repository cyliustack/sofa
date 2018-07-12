FROM tensorflow/tensorflow:latest-gpu
MAINTAINER Cheng-Yueh Liu <cyliustack@gmail.com>
RUN apt-get update && apt-get install -y git wget zip unzip vim python3 python3-pip 
RUN wget https://github.com/cyliustack/sofa/archive/master.zip && unzip master.zip && rm master.zip  && cd sofa-master && ./tools/prerequisite.sh && ./install.sh /opt/sofa && echo "source /opt/sofa/tools/activate.sh" >> ~/.bashrc 
RUN wget https://github.com/cyliustack/scout/archive/master.zip && unzip master.zip.1 && rm master.zip && cd scout-master && ./tools/prepare.sh  
RUN source ~/.bashrc && cd scout && sofa record "./scout t-bench resnet50" && sofa report 
RUN mkdir -p /tmp/program 
RUN mkdir -p /tmp/data 




