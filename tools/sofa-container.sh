#!/bin/bash
WITH_SUDO="" 
if [[ "$(whoami)" != "root"  ]]; then 
    WITH_SUDO="sudo"
fi 

if [[ $(which yum) ]]; then 
    $WITH_SUDO yum remove docker \
                      docker-client \
                      docker-client-latest \
                      docker-common \
                      docker-latest \
                      docker-latest-logrotate \
                      docker-logrotate \
                      docker-selinux \
                      docker-engine-selinux \
                      docker-engine
    $WITH_SUDO yum install -y yum-utils \
      device-mapper-persistent-data \
      lvm2
    $WITH_SUDO yum-config-manager \
        --add-repo \
        https://download.docker.com/linux/centos/docker-ce.repo
    $WITH_SUDO yum install docker-ce
    $WITH_SUDO systemctl start docker
    $WITH_SUDO docker run hello-world
    
    # If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
    docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
    $WITH_SUDO yum remove nvidia-docker
    
    # Add the package repositories
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | \
          $WITH_SUDO tee /etc/yum.repos.d/nvidia-docker.repo
    
    # Install nvidia-docker2 and reload the Docker daemon configuration
    $WITH_SUDO yum install -y nvidia-docker2
    $WITH_SUDO pkill -SIGHUP dockerd
    
    # Test nvidia-smi with the latest official CUDA image
    docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi
elif [[ $(which apt) ]]; then 
	$WITH_SUDO apt-get remove docker docker-engine docker.io containerd runc
	$WITH_SUDO apt-get update
	$WITH_SUDO apt-get install \
	    apt-transport-https \
	    ca-certificates \
	    curl \
	    gnupg-agent \
	    software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | $WITH_SUDO apt-key add -
	$WITH_SUDO apt-key fingerprint 0EBFCD88
	$WITH_SUDO add-apt-repository \
	   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
	   $(lsb_release -cs) \
	   stable"
	$WITH_SUDO apt-get update
	$WITH_SUDO apt-get install docker-ce docker-ce-cli containerd.io
	
	# If you have nvidia-docker 1.0 installed: we need to remove it and all existing GPU containers
	docker volume ls -q -f driver=nvidia-docker | xargs -r -I{} -n1 docker ps -q -a -f volume={} | xargs -r docker rm -f
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
	
	$WITH_SUDO groupadd docker
	
	$WITH_SUDO usermod -aG docker $USER
	
	# Test nvidia-smi with the latest official CUDA image
	$WITH_SUDO docker run --runtime=nvidia --rm nvidia/cuda:9.0-base nvidia-smi

else
    echo "Not supported platform!"
fi


mkdir -p ~/program 
mkdir -p ~/data 
$WITH_SUDO docker build -t tfsofa -f Dockerfile .
$WITH_SUDO docker run --runtime=nvidia -it --privileged -p 8000:8000 -p 8888:8888 -c 8 -v $(pwd):/home/workspace tfsofa bash
