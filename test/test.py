#!/usr/bin/env python

import argparse
from subprocess import call, check_output, DEVNULL
import os, sys
from datetime import datetime
import getpass
	

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Script')
    parser.add_argument('dockerfiles', nargs='+', metavar='Dockerfile.xxx.xxx', help='Specify the dockerfile to build and run.')
    args = parser.parse_args()
    
    distro  = ''
    version = ''

    test_script_path = os.path.abspath( __file__ )
    os.chdir(os.path.dirname(test_script_path))
    date_time = datetime.now().strftime("%m-%d-%Y-%H-%M")
    logfile_name = 'test-' + date_time  + '.log'
    with open(logfile_name, 'w') as logfile:
        logfile.write('====== Test SOFA on Different OS distributions ======\n')
        logfile.write('This test is done by ' + getpass.getuser() + ' at ' + date_time + '\n')

    for dockerfile in args.dockerfiles:
        if not os.path.isfile(dockerfile):
            print(dockerfile + ' is not found.')
            sys.exit(-1)
        if len(dockerfile.split('.')) != 3:
            print('Incorrect dockerfile name: ', dockerfile)
        else:
            distro = dockerfile.split('.')[1] 
            version = dockerfile.split('.')[2]
        image_name = distro + ':' + version 
        print('image name  : ', image_name)
        print('distribution: ', distro)
        print('version     : ', version)
        print('========================')
        call('docker build --network host -f %s --tag %s .' % (dockerfile, image_name), shell=True )
        call('docker rm sofa_test', shell=True, stdout=DEVNULL, stderr=DEVNULL)
        result = check_output('docker run --name sofa_test --rm -it --network host -v $(pwd)/..:/sofaroot --privileged %s /sofaroot/tools/prepare.sh' % (image_name), shell=True).decode("utf-8")
        print(result)
        with open(logfile_name, 'a') as logfile:
            if result.find('Complete!!') != -1:
                message = 'Testing on ' + image_name + ' PASSED!\n'
            else:
                message = 'Testing on ' + image_name + ' FAILED!\n'

            print(message)
            logfile.write(message)

