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

    this_path = os.path.abspath( __file__ )
    this_dir = os.path.dirname(this_path)
    os.chdir(this_dir)
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

        # make files ready in directory of test
        call('mkdir -p sofaroot', shell=True )
        call('rm -r sofaroot/*', shell=True )
        call('cp -r ' + this_dir + '/../tools/ sofaroot/', shell=True )
        call('cp -r ' + this_dir + '/../bin/ sofaroot/', shell=True )
        call('cp -r ' + this_dir + '/../sofaboard/ sofaroot/', shell=True )
        call('cp ' + this_dir + '/../install.sh sofaroot/', shell=True )

        # build base image for test
        call('docker build --network host -f %s --tag %s .' % (dockerfile, image_name), shell=True )
        
        # run container in background and then use "docker exec" for multiple tests
        call('docker rm -f sofa_test', shell=True, stdout=DEVNULL, stderr=DEVNULL)
        call('docker run --name sofa_test --rm -itd --network host --privileged ' + image_name + ' bash', shell=True)
        call('docker exec sofa_test bash /sofaroot/install.sh /opt/sofa', shell=True)
        call('docker exec sofa_test python3 /opt/sofa/bin/sofa record "sleep 5"', shell=True)
        result = check_output('docker exec sofa_test python3 /opt/sofa/bin/sofa report --verbose', shell=True).decode("utf-8")
        print(result)
        with open(logfile_name, 'a') as logfile:
            if result.find('Complete!!') != -1:
                message = 'Testing on ' + image_name + ' PASSED!\n'
            else:
                message = 'Testing on ' + image_name + ' FAILED!\n'

            print(message)
            logfile.write(message)

