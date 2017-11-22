#!/bin/bash	
for i in "$@"
do
case $i in
    -e=*|--prefix=*)
    PREFIX="${i#*=}"
    shift # past argument=value
    ;;
    *)
          # unknown option
    ;;
esac
done

if [[ ${PREFIX} == "" ]]; then
    echo "Usage: ./install.sh --prefix=/directory/to/install"
    read -p "Use default path /opt/sofa ?(Y/n) " -n 1 -r
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        PREFIX=/opt/sofa 
        echo ""   
    else
        echo ""
        echo "Goodbye." 
        exit -1 
    fi 
    
fi

echo " Installation directory is ${PREFIX}"

set -x
rm -rf ${PREFIX}
rm -f /usr/local/bin/sofastat.py
rm -f /usr/local/bin/sofa
mkdir -p ${PREFIX}/bin
mkdir -p ${PREFIX}/sofaboard
mkdir -p ${PREFIX}/plugin
cp -i sofa                      ${PREFIX}/bin
cp -i sofastat.py               ${PREFIX}/bin
cp -f sofaboard/index.html      ${PREFIX}/sofaboard
cp -f sofaboard/gpu-report.html ${PREFIX}/sofaboard
ln -is ${PREFIX}/bin/sofa           /usr/local/bin/sofa
ln -is ${PREFIX}/bin/sofastat.py    /usr/local/bin/sofastat.py

