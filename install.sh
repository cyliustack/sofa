#!/bin/bash	

print_help(){
    echo "Usage: ./install.sh /path/to/directory/of/sofa"
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

PREFIX=$1

if [[ ${PREFIX} == "" ]]; then
    print_help
    read -p "Use default path /opt/sofa ?(Y/n) " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]
    then        
        PREFIX=/opt/sofa 
        echo ""   
    else
        echo ""
        echo "Try './install.sh {-h|--help}' for more information."
        exit -1 
    fi 
    
fi

echo " Installation directory is ${PREFIX}"
set -x
mkdir -p ${PREFIX}
mkdir -p ${PREFIX}/bin
mkdir -p ${PREFIX}/sofaboard
mkdir -p ${PREFIX}/plugin
mkdir -p ${PREFIX}/tools
rm ${PREFIX}/bin/*.pyc
cp -f sofa                      ${PREFIX}/bin
cp -f sofa-preprocess.py        ${PREFIX}/bin
cp -f sofa-analyze.py           ${PREFIX}/bin
cp -f sofa_print.py             ${PREFIX}/bin
cp -f sofa_config.py            ${PREFIX}/bin
cp -f sofaboard/index.html          ${PREFIX}/sofaboard
cp -f sofaboard/timeline.js         ${PREFIX}/sofaboard
cp -f sofaboard/cpu-report.html     ${PREFIX}/sofaboard
cp -f sofaboard/gpu-report.html     ${PREFIX}/sofaboard
cp -f sofaboard/overhead.html          ${PREFIX}/sofaboard
cp -f sofaboard/comm-report.html          ${PREFIX}/sofaboard
echo "export PATH=\$PATH:${PREFIX}/bin" > tools/activate.sh
echo "export PATH=\$PATH:/usr/local/cuda/bin" >> tools/activate.sh
cp -f tools/activate.sh   ${PREFIX}/tools
set +x
echo "Please try 'source ${PREFIX}/tools/activate.sh' to enjoy sofa!"
