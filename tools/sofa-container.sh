#!/bin/bash
WITH_SUDO="" 
if [[ "$(whoami)" != "root"  ]]; then 
    WITH_SUDO="sudo"
fi 

if [[ $(which yum) ]]; then 
    # If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
    $WITH_SUDO docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
    $WITH_SUDO yum remove nvidia-docker
    
    # Add the package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
          $WITH_SUDO tee /etc/yum.repos.d/nvidia-docker.repo
    
    # Install nvidia-docker2 and reload the Docker daemon configuration
    $WITH_SUDO yum install -y nvidia-docker2
    $WITH_SUDO pkill -SIGHUP dockerd
    
    # Test nvidia-smi with the latest official CUDA image
    $WITH_SUDO docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
elif [[ $(which apt) ]]; then 
    $WITH_SUDO apt-get install -y nvidia-docker2=2.0.1+docker1.12.6-1 nvidia-container-runtime=1.1.0+docker1.12.6-1

    # If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
    $WITH_SUDO docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
    $WITH_SUDO apt-get purge -y nvidia-docker
    
    # Add the package repositories
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
          $WITH_SUDO apt-key add -
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
          $WITH_SUDO tee /etc/apt/sources.list.d/nvidia-docker.list
    $WITH_SUDO apt-get update
    
    # Install nvidia-docker2 and reload the Docker daemon configuration
    $WITH_SUDO apt-get install -y nvidia-docker2
    $WITH_SUDO pkill -SIGHUP dockerd
    
    # Test nvidia-smi with the latest official CUDA image
    $WITH_SUDO docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
else
    echo "Not supported platform!"
fi

mkdir -p ~/program 
mkdir -p ~/data
$WITH_SUDO docker build -t tfsofa -f Dockerfile .
$WITH_SUDO docker run --runtime=nvidia -it --privileged -p 8000:8000 -p 8888:8888 -c 8 -v ~/program:/tmp/program:rw -v ~/data:/tmp/data:rw  tfsofa bash

