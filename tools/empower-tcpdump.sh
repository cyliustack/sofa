#!/bin/bash
# Copyright (c) Jul. 2017, Cheng-Yueh Liu (cyliustack@gmail.com)

C_NONE="\033[0m"
C_CYAN="\033[36m"
C_RED="\033[31m"
C_GREEN="\033[32m"
C_ORANGE="\033[33m"
C_BLUE="\033[34m"
C_PURPLE="\033[35m"
C_CYAN="\033[36m"
C_LIGHT_GRAY="\033[37m"

print_misc() {
    echo -e "${C_PURPLE} $1 ${C_NONE}"
}

print_info() {
    echo -e "${C_BLUE} $1 ${C_NONE}"
}

print_error() {
    echo -e "${C_RED} $1 ${C_NONE}"
}

print_warning() {
    echo -e "${C_ORANGE} $1 ${C_NONE}"
}

function print_help()
{
    print_misc "empower-tcpdump.sh username" 
}

function main()
{
    if [[ "$*" == "" ]]; then
	print_help
	exit -1
    fi

    #args=$*
    print_info "Empower User with Tcpdump"
    
    until [ $# -eq 0 ]
    do
        case "$1" in
            "help" )
                print_help
                exit 0
                ;;
	        "--help" )
                print_help
                exit 0
                ;;
            "-h" )
                print_help
                exit 0
                ;;
            *)
                username=$1
                sudo groupadd pcap
                sudo usermod -a -G pcap ${username}
                sudo chgrp pcap /usr/sbin/tcpdump
                sudo chmod 750 /usr/sbin/tcpdump
                sudo setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump               
                exit 0
                ;;
        esac
        shift
    done
}
      
##### Main() ##### 
main $*

