#!/usr/bin/env python
# requirements: apt install git wget python gcc flex bison make 
import platform
import subprocess

if subprocess.call('which sudo', shell=True) == 0:
    with_sudo = 'sudo ' 
else:
    with_sudo = '' 

major = int(platform.release().split('.')[0])
minor = int(platform.release().split('.')[1])
revision = int(platform.release().split('.')[2].split('-')[0])
url_kernel = 'https://mirrors.edge.kernel.org/pub/linux/kernel/v%d.x/linux-%d.%d.tar.gz' % (major, major, minor)
tarfile = 'linux-%d.%d.tar.gz' % (major, minor)
source_dir = 'linux-%d.%d' % (major, minor)
print('URL: ', url_kernel)
print('TarFile: ', tarfile)
subprocess.call('rm -r %s' % (source_dir), shell=True)
subprocess.call('rm %s' % (tarfile), shell=True)
subprocess.call('wget %s' % (url_kernel) , shell=True)
subprocess.call('tar xf %s && make -j -C %s/tools/perf' % (tarfile, source_dir) , shell=True)
subprocess.call(with_sudo + 'cp %s/tools/perf/perf /usr/bin/' % (source_dir) , shell=True)
subprocess.call('rm -r %s' % (source_dir), shell=True)
subprocess.call('rm %s' % (tarfile), shell=True)
subprocess.call('ls -lah /usr/bin/perf', shell=True)
#get kernelversion
#wget http://www.kernel.org/pub/linux/kernel/v2.6/testing/linux-2.6.33-rc3.tar.bz2

