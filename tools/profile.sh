#!/bin/sh
function print_help()
{
    echo "$0 [OPTIONS, ...] command [command args...]"
    echo "    OPTIONS: None"
}

function encapsulate()
{
    whitespace="[[:space:]]"
    for i in "$@"
    do
        if [[ $i =~ $whitespace ]]
        then
            i=\"$i\"
        fi
        # echo "$i"
        args+="$i "
    done

    echo ${args}
}

options=""
while [[ 1 ]]; do
    case "$1" in
        "--monitor" )
            options+="--monitor "
            ;;
        "--phase" )
            options+="--phase "
            ;;
        "--trace" )
            options+="--trace "
            ;;
        "--jit" )
            options+="--jit "
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
            break
            ;;
    esac
    shift 1 # Remove the fisrt argument
done

args=$*
sofa_root=$(dirname $0)/..
rm perf.data
perf record -a ${args}
perf script > perf.script
${sofa_root}/sofa perf.script
#python3  $SOFA_ROOT/sofaviz.py
