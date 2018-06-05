#!/bin/bash

C_NONE="\033[0;00m"
C_GREEN="\033[1;32m"
C_RED_BK="\033[1;41m"

# Detect OS distribution
# Try source all the release files
for file in /etc/*-release; do
    source $file 
done

if [[ "$NAME" != ""  ]]; then
    OS="$NAME"
    VERSION="$VERSION_ID"
elif [[ -f /etc/debian_version ]]; then
    # Older Debian/Ubuntu/etc.
    OS="Debian"
    VERSION="$(cat /etc/debian_version)"
else
    OS="$(lsb_release -si)"
    VERSION="$(lsb_release -sr)"
fi

function inform_sudo()
{
    [[ ! -z "$1" ]] && echo "$1"
    # Exit without printing messages if password is still in the cache.
    sudo -n true 2> /dev/null
    [[ $? == 0 ]] && return 0;
    sudo >&2 echo -e "\033[1;33mRunning with root privilege now...\033[0;00m";
    [[ $? != 0 ]] && >&2 echo -e "\033[1;31mAbort\033[0m" && exit 1;
}

function install_python_packages()
{
    # Install Python packages
    echo -e "${C_GREEN}Installing python packages...${C_NONE}"
     
    if [[ $(which yum) ]]  ; then
        yum install epel-release
        yum install https://centos7.iuscommunity.org/ius-release.rpm
        yum install python36u
        yum install python36u-pip
    elif [[ $(which apt) ]]  ; then	
        sudo add-apt-repository ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install python3.6
    else
	    url_python36="Python-3.6.0.tar.xz"
	    if [[ ! -f "Python-3.6.0.tar.xz" ]]; then
	        wget https://www.python.org/ftp/python/3.6.0/$url_python36
	        tar xJf $url_python36
	        cd Python-3.6.0
	        ./configure ./configure --with-ssl
	        cd -
	    fi 
	    cd Python-3.6.0
	    make -j
	    sudo make install
	    # Install for Python3
	    python3.6 -m pip install --upgrade upgrade pip
	    python3.6 -m pip install numpy pandas networkx cxxfilt 
	    cd - 
	    rm -r Python-3.6.0*
    fi  
    #python3 -m pip uninstall numpy pandas networkx ccxfilt
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy pandas networkx cxxfilt 
    [[ $? != 0 ]] && echo -e "${C_RED_BK}Failed... :(${C_NONE}" && exit 1
}

function install_packages()
{
    echo -e "${C_GREEN}Installing other packages...${C_NONE}"

    #inform_sudo "Running sudo for installing packages"
    if [[ $(which apt) ]] ; then
        apt-get update
        apt-get update --fix-missing
        apt-get install \
            libboost-dev libpcap-dev libconfig-dev libconfig++-dev linux-tools-common \
            linux-tools-$(uname -r) linux-cloud-tools-$(uname -r) linux-tools-generic linux-cloud-tools-generic \
            cmake tcpdump sysstat
        [[ $? != 0 ]] && echo -e "${C_RED_BK}Failed... :(${C_NONE}" && exit 1
    elif [[ $(which yum) ]]  ; then
        yum install epel-release 
        yum install \
            perf tcpdump\
            centos-release-scl devtoolset-4-gcc* sysstat
        [[ $? != 0 ]] && echo -e "${C_RED_BK}Failed... :(${C_NONE}" && exit 1
    elif [[ $(which dnf) ]]  ; then
        dnf -y install \
            perf cmake tcpdump boost-devel libconfig-devel libpcap-devel cmake sysstat
        [[ $? != 0 ]] && echo -e "${C_RED_BK}Failed... :(${C_NONE}" && exit 1
    elif [[ $(which pacman) ]]  ; then
        pacman -S \
            linux-tools cmake boost cmake tcpdump sysstat
    else
        echo -e "${C_RED_BK}This script does not support your OS distribution, '$OS'. Please install the required packages by yourself. :(${C_NONE}"
    fi
}

# main
echo -e "${C_GREEN}OS Distribution:${C_NONE} '$OS'"
echo -e "${C_GREEN}Version:${C_NONE} '$VERSION'"
printf "\n\n"

install_packages
install_python_packages

echo -e "${C_GREEN}Complete!!${C_NONE}"
