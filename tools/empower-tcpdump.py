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
    parser = argparse.ArgumentParser(description='Running sudo for setting "/usr/sbin/tcpdump" with "pcap" group.')
    parser.add_argument( 'users', nargs='+', metavar='TheUserName')
    args = parser.parse_args()
    
    with_sudo = ''
    if(os.path.exists('/usr/bin/sudo')):
        with_sudo = 'sudo '
        inform_sudo()
         
    call(with_sudo + 'groupadd pcap', shell=True)
    ret = call(with_sudo + 'chgrp pcap /usr/sbin/tcpdump', shell=True)
    if ret != 0:
        print('Failed...')
        sys.exit(1)
    call(with_sudo + 'chmod 750 /usr/sbin/tcpdump', shell=True)
    ret = call(with_sudo + 'setcap cap_net_raw,cap_net_admin=eip /usr/sbin/tcpdump', shell=True)
    if ret != 0:
        print('Failed...')
        sys.exit(1)

    for user in args.users:
        print('For ' + user + ':')
        ret = call('id -u %s > /dev/null 2>/dev/null' % user, shell=True)
        if ret != 0:
            print('User name - ' + user + ' does not exist.')
            sys.exit(1)
        else:
            inform_sudo()
            call(with_sudo + 'usermod -a -G pcap %s' % user, shell=True)
        print('Every thing is ready.')

    #call(with_sudo + 'newgrp pcap', shell=True) 
    print('\n\nPlease logout and then login to make group setting effective if necessary.')
