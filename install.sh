#!/bin/bash

SCRIPT_PATH="$(readlink -f "$(dirname "$0")")"

C_NONE="\033[0;00m"
C_RED="\033[1;31m"
C_GREEN="\033[1;32m"

print_help(){
    echo "This script will install the sofa scripts to your system under the specified <PREFIX> directory."
    echo ""
    echo "Usage: $0 PREFIX"
    echo "Examples: $0 ~/bin/sofa"
    echo "          sudo $0 /opt/bin/sofa"
}

function clear_install_dir()
{
    echo "Creating directories..."
    mkdir -p ${PREFIX}
    mkdir -p ${PREFIX}/bin
    mkdir -p ${PREFIX}/sofaboard
    mkdir -p ${PREFIX}/plugins
    mkdir -p ${PREFIX}/tools
    rm -f ${PREFIX}/bin/*.pyc
}

function install_sofa()
{
    echo "Installing..."
    cp -rf ${SCRIPT_PATH}/bin       ${PREFIX}
    cp -rf ${SCRIPT_PATH}/plugins   ${PREFIX}
    cp -rf ${SCRIPT_PATH}/sofaboard ${PREFIX}

    # Create a new file for configurations
################## heredoc style
    cat > ${PREFIX}/tools/activate.sh <<EOF
export PATH=${PREFIX}/bin:\$PATH
export PATH=\$PATH:/usr/local/cuda/bin
EOF
##################
}

for i in "$@"
do
case $i in
    -h|--help)
        print_help
        exit 0
    ;;
    *)
        # unknown option
    ;;
esac
done

if [[ "$1" == "" ]]; then
    echo -e "${C_RED}Please specify the install directory!${C_NONE}"
    print_help
    exit 1
fi

# Get the first argument and also remove the last / if present
PREFIX="${1%/}"
# Add sofa to PREFIX if not present
[[ $(basename "$PREFIX") != "sofa" ]] && PREFIX="${PREFIX}/sofa"
PREFIX=$(readlink -f "$PREFIX")

echo -e "${C_GREEN}Installation directory is ${PREFIX}${C_NONE}"

# Print every executed command for debugging
set -x -e
clear_install_dir
install_sofa
# Disable printing every executed command for debugging
set +x +e

echo -e "${C_GREEN}Please try 'source ${PREFIX}/tools/activate.sh' to enjoy sofa!${C_NONE}\n"
echo -e "${C_GREEN}Add 'source ${PREFIX}/tools/activate.sh' to your ~/.bashrc if you want to enable sofa on every shells.${C_NONE}\n\n"
