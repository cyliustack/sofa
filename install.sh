#!/bin/bash	

print_help(){
    echo "Usage: ./install.sh --prefix=/path/to/directory/of/sofa"
}


for i in "$@"
do
case $i in
    -e=*|--prefix=*)
        PREFIX="${i#*=}"
        shift # past argument=value
    ;;
    -h|--help)
        print_help
        exit 0
    ;;
    *)  
        # unknown option
    ;;
esac
done


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
rm -rf ${PREFIX}
mkdir -p ${PREFIX}/bin
mkdir -p ${PREFIX}/sofaboard
mkdir -p ${PREFIX}/plugin
cp -f sofa                      ${PREFIX}/bin
cp -f sofastat.py               ${PREFIX}/bin
cp -f sofaboard/index.html      ${PREFIX}/sofaboard
cp -f sofaboard/gpu-report.html ${PREFIX}/sofaboard
echo "export PATH=\$PATH:${PREFIX}/bin" > tools/activate.sh
set +x
echo "Please try 'source tools/activate.sh' to enjoy sofa!"
