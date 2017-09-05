#!/bin/bash

if [ -f /etc/os-release ]; then
    # freedesktop.org and systemd
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    # linuxbase.org
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
elif [ -f /etc/lsb-release ]; then
    # For some versions of Debian/Ubuntu without lsb_release command
    . /etc/lsb-release
    OS=$DISTRIB_ID
    VER=$DISTRIB_RELEASE
elif [ -f /etc/debian_version ]; then
    # Older Debian/Ubuntu/etc.
    OS=Debian
    VER=$(cat /etc/debian_version)
elif [ -f /etc/SuSe-release ]; then
    # Older SuSE/etc.
    ...
elif [ -f /etc/redhat-release ]; then
    # Older Red Hat, CentOS, etc.
    ...
else
    # Fall back to uname, e.g. "Linux <version>", also works for BSD, etc.
    OS=$(uname -s)
    VER=$(uname -r)
fi


echo $OS
echo $ARCH
echo $VERSION

os_found=true

if [[ $OS == "Ubuntu" ]]; then
	echo "This is Ubuntu "
	sudo apt-get install libboost-dev libpcap-dev libconfig-dev libconfig++-dev linux-tools-common linux-tools-$(uname -r) linux-cloud-tools-$(uname -r)  linux-tools-generic linux-cloud-tools-generic cmake	
elif [[ $OS == "CentOS Linux" ]]; then
    echo "This is CentOS"
    sudo yum install perf cmake libpcap-devel libconfig-devel boost-devel
elif [[ $OS == "Fedora" ]]; then
	echo "This Fedora "
    sudo dnf -y install perf boost-devel libconfig-devel libpcap-devel cmake
else
	os_found=false
fi

if [[ $os_found == true ]]; then
    wget http://bitbucket.org/eigen/eigen/get/3.3.4.tar.gz
	tar -xvf 3.3.4.tar.gz && cd eigen-eigen-5a0156e40feb && mkdir -p build && cd build && cmake .. && make && sudo make install 
    cd ../.. 
    rm -rf eigen-eigen-5a0156e40feb
else
   echo "Oops, Cannot identify your OS version!"
fi
