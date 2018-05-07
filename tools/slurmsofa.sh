#!/bin/bash
SOFA(){
        source /opt/sofa/tools/activate.sh
        user_command=''
        for var in $@ ;
        do
                user_command=$user_command" "$var
                echo $var
        done
        echo $user_command
        sofa record "${user_command}"
}

SOFA $@

