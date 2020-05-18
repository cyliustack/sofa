#!/usr/bin/env python

from subprocess import call
import os
import sys 
import argparse

def inform_sudo():
    # Exit without printing messages if password is still in the cache.
    ret = call('sudo -n true 2> /dev/null', shell=True)
    if ret == 0:
        return 0
    
    ret = call('sudo >&2 echo -e "\033[1;33mRunning with root privilege now...\033[0m"', shell=True) 
    if ret != 0:
        call('>&2 echo -e "\033[1;31mAbort\033[0m" && exit 1;', shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Running sudo for setting a system utility (e.g., /usr/sbin/tcpdump) with "sofa" group.')
    parser.add_argument( 'user', metavar='user_name')
    parser.add_argument( 'utility', metavar='system_utility')
    parser.add_argument( '--recover', action='store_true', help='recover capability of specified program')
    parser.add_argument( '--remove-group-sofa', action='store_true', help='remove sofa-group')
    args = parser.parse_args()
 

    if not os.path.isfile(args.utility):
        print(args.utility + ' is not a valid file path.')
        sys.exit(-1)
        
    with_sudo = ''
    if(os.path.exists('/usr/bin/sudo')):
        with_sudo = 'sudo '
        inform_sudo()

    if args.recover:     
        print('Recovering mode of user/group and capability of "' + args.utility + '"...')
        ret = call(with_sudo + 'chgrp root ' + ' ' + args.utility, shell=True)
        if args.remove_group_sofa:
            print('The group "sofa" is going to be removed. Are you sure (y/n)?', end=' ')
            if input() == 'y':
                call(with_sudo + 'groupdel sofa', shell=True)
        sys.exit(0)
    else:
        call(with_sudo + 'groupadd sofa', shell=True)
        ret = call(with_sudo + 'chgrp sofa ' + args.utility, shell=True)
        if ret != 0:
            print('Failed...')
            sys.exit(1)
        call(with_sudo + 'chmod g+rx ' + args.utility, shell=True)

    #Set capabilities for the specified utility
    if args.utility.find('tcpdump') != -1:
        ret = call(with_sudo + 'setcap cap_net_raw,cap_net_admin=eip ' + args.utility, shell=True)
    
    if ret != 0:
        print('Failed...')
        sys.exit(1)

    print('For ' + args.user + ':')
    ret = call('id -u %s > /dev/null 2>/dev/null' % args.user, shell=True)
    if ret != 0:
        print('User name - ' + args.user + ' does not exist.')
        sys.exit(1)
    else:
        inform_sudo()
        call(with_sudo + 'usermod -a -G sofa %s' % args.user, shell=True)
    print('Every thing is ready.')

    #call(with_sudo + 'newgrp sofa', shell=True) 
    print('\n\nPlease logout and then login to make group setting effective if necessary.')
