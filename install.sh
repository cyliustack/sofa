#!/bin/bash

SCRIPT_PATH="$(readlink -f "$(dirname "$0")")"

C_NONE="\033[0;00m"
C_RED="\033[1;31m"
C_GREEN="\033[1;32m"

INSTALL_DIRS=(bin
              sofaboard
              plugins
              tools
              )
BIN_FILES=(bin/sofa
           bin/sofa_record.py
           bin/sofa_preprocess.py
           bin/sofa_analyze.py
           bin/sofa_deepprof.py
           bin/STree.py
           bin/sofa_viz.py
           bin/sofa_config.py
           bin/sofa_print.py
           sofa-pcm/pcm-core.x
           sofa-pcm/pcm-numa.x
           sofa-pcm/pcm-pcie.x
           sofa-pcm/pcm-memory.x
           )
PLUGIN_FILES=(plugins/.placeholder
              )
SOFABOARD_FILES=(sofaboard/index.html
                 sofaboard/cpu-report.html
                 sofaboard/gpu-report.html
                 sofaboard/comm-report.html
                 sofaboard/overhead.html
                 sofaboard/timeline.js
                 )

print_help(){
    echo "This script will install the sofa scripts to your system under the specified <PREFIX> directory."
    echo ""
    echo "Usage: $0 PREFIX"
    echo "Examples: $0 ~/third-parties/sofa"
    echo "          sudo $0 /opt/sofa"
}

function clear_install_dir()
{
    # Use uninstall script to safely remove old files
    if [[ -f ${PREFIX}/tools/uninstall.sh ]]; then
        echo "Uninstalling SOFA..."
        bash ${PREFIX}/tools/uninstall.sh
    fi
    echo "Creating directories..."
    mkdir -p ${PREFIX}
    mkdir -p ${PREFIX}/bin
    mkdir -p ${PREFIX}/sofaboard
    mkdir -p ${PREFIX}/plugins
    mkdir -p ${PREFIX}/tools
}

function install_sofa()
{
    echo "Installing..."
    cp sofa-pcm/pcm-pcie.x          ${PREFIX}/bin
    cp sofa-pcm/pcm-numa.x          ${PREFIX}/bin
    cp sofa-pcm/pcm-core.x          ${PREFIX}/bin
    cp sofa-pcm/pcm-memory.x    ${PREFIX}/bin
    cp -rf ${SCRIPT_PATH}/bin       ${PREFIX}
    cp -rf ${SCRIPT_PATH}/bin       ${PREFIX}
    cp -rf ${SCRIPT_PATH}/plugins   ${PREFIX}
    cp -rf ${SCRIPT_PATH}/sofaboard ${PREFIX}

    # Create a new file for SOFA environment
################## heredoc style
    cat > ${PREFIX}/tools/activate.sh <<EOF
export PATH=${PREFIX}/bin:\$PATH
export PATH=\$PATH:/usr/local/cuda/bin
export PATH=\$PATH:/usr/local/intelpcm/bin
EOF
##################

    # Create an uninstall script
################## heredoc style
    cat > ${PREFIX}/tools/uninstall.sh <<EOF
# Change the directory
cd ${PREFIX}
[[ \$(pwd) != ${PREFIX} ]] && echo "Fail to change directory. Stop uninstalling..." && exit 1
# Remove installed files
rm -f ${BIN_FILES[@]}
rm -f ${PLUGIN_FILES[@]}
rm -f ${SOFABOARD_FILES[@]}
# Remove all python caches
rm -rf __pycache__ plugins/__pycache__ bin/__pycache__
# Remove generated files
rm -f tools/activate.sh
rm -f tools/uninstall.sh
# Remove directory only if it is empty!
rmdir --ignore-fail-on-non-empty ${INSTALL_DIRS[@]}
rmdir --ignore-fail-on-non-empty ${PREFIX}
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

# Detect empty string
if [[ "$1" == "" ]]; then
    echo -e "${C_RED}Please specify the install directory!${C_NONE}"
    print_help
    exit 1
fi

# Detect white space in the path
if [[ "$1" != "${1%[[:space:]]*}" ]]; then
    echo -e "${C_RED}The installation path cannot contain space character.${C_NONE}"
    exit 1
fi

# Get the first argument and also remove the tailing "/" if present
PREFIX="${1%/}"
# Add sofa to PREFIX if not present
[[ $(basename "$PREFIX") != "sofa" ]] && PREFIX="${PREFIX}/sofa" 
# Resolve the abs path
PREFIX="$(readlink -f "$(dirname  "$PREFIX")")/$(basename "$PREFIX")"
#PREFIX="$(readlink -f "$PREFIX")"
echo -e "${C_GREEN}Installation directory is ${PREFIX}${C_NONE}"
if [[ $(pwd) == $PREFIX ]]; then 
    echo -e "${C_RED}Warning! Installation directory is the same as the source code directory.${C_NONE}"
    echo -e "${C_RED}Please try another installation direcotry.${C_NONE}"
    exit 1
fi

# Print every executed command for debugging
set -x -e
clear_install_dir
install_sofa
# Disable printing every executed command for debugging
set +x +e

echo -e "${C_GREEN}Please try 'source ${PREFIX}/tools/activate.sh' to enjoy sofa!${C_NONE}\n"
echo -e "${C_GREEN}Add 'source ${PREFIX}/tools/activate.sh' to your ~/.bashrc if you want to enable sofa on every shells.${C_NONE}\n\n"
