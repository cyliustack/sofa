import argparse
import csv
import datetime
import glob
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from functools import partial
from pwd import getpwuid

import numpy as np

from sofa_print import *


def service_get_cpuinfo(logdir):
    next_call = time.time()
    while True:
        #print(datetime.datetime.now())
        next_call = next_call + 0.1;
        get_cpuinfo(logdir)
        time.sleep(next_call - time.time())

def get_cpuinfo(logdir):
    with open('/proc/cpuinfo','r') as f:
        lines = f.readlines()
        mhz = 1000
        for line in lines:
            if line.find('cpu MHz') != -1:
                mhz = float(line.split()[3])
                break
        with open('%s/cpuinfo.txt' % logdir, 'a') as logfile:
            unix_time = time.time()
            logfile.write(str('%.9lf %lf'%(unix_time,mhz)+'\n'))

def kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa):
    if p_pcm_pcie != None:
        p_pcm_pcie.terminate()
        os.system('yes|pkill pcm-pcie.x')
        print_info("tried killing pcm-pcie.x")
    if p_pcm_memory != None:
        p_pcm_memory.terminate()
        os.system('yes|pkill pcm-memory.x')
        print_info("tried killing pcm-memory.x")
    if p_pcm_numa != None:
        p_pcm_numa.terminate()
        os.system('yes|pkill pcm-numa.x')
        print_info("tried killing pcm-numa.x")


def sofa_clean(logdir, cfg):
    print_info('Clean previous logged files')
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.html > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.js > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.script > /dev/null 2> /dev/null' % logdir, shell=True)


def sofa_record(command, logdir, cfg):

    p_tcpdump = None
    p_mpstat  = None
    p_vmstat  = None
    p_cpuinfo  = None
    p_nvprof  = None
    p_nvsmi   = None
    p_nvtopo  = None
    p_pcm_pcie = None
    p_pcm_memory = None
    p_pcm_numa = None

    print_info('SOFA_COMMAND: %s' % command)
    sample_freq = 99
    if int(open("/proc/sys/kernel/kptr_restrict").read()) != 0:
        print_error(
            "/proc/kallsyms permission is restricted, please try the command below:")
        print_error("sudo sysctl -w kernel.kptr_restrict=0")
        quit()

    if int(open("/proc/sys/kernel/perf_event_paranoid").read()) != -1:
        print_error('PerfEvent is not avaiable, please try the command below:')
        print_error('sudo sysctl -w kernel.perf_event_paranoid=-1')
        quit()

    if cfg.enable_pcm:
        print_info('Test Capability of PCM programs ...')
        #ret = str(subprocess.check_output(['getcap `which pcm-pcie.x`'], shell=True))
        #if ret.find('cap_sys_rawio+ep') == -1:
        #    print_error('To read/write MSR in userspace is not avaiable, please try the commands below:')
        #    print_error('sudo modprobe msr')
        #    print_error('sudo setcap cap_sys_rawio=ep `which pcm-pcie.x`')
        #    quit()
        ret = str(subprocess.check_output(['getcap `which pcm-memory.x`'], shell=True))
        if ret.find('cap_sys_rawio+ep') == -1:
            print_error('To read/write MSR in userspace is not avaiable, please try the commands below:')
            print_error('sudo modprobe msr')
            print_error('sudo setcap cap_sys_rawio=ep `which pcm-memory.x`')
            quit()
        #ret = str(subprocess.check_output(['getcap `which pcm-numa.x`'], shell=True))
        #if ret.find('cap_sys_rawio+ep') == -1:
        #    print_error('To read/write MSR in userspace is not avaiable, please try the commands below:')
        #    print_error('sudo modprobe msr')
        #    print_error('sudo setcap cap_sys_rawio=ep `which pcm-numa.x`')
        #    quit()

    if subprocess.call(['mkdir', '-p', logdir]) != 0:
        print_error('Cannot create the directory' + logdir + ',which is needed for sofa logged files.' )
        quit()

        print_info('Read NMI watchlog status ...')
        nmi_output = ""
        try:
            with open(logdir+"nmi_status.txt", 'w') as f:
                p_pcm_pcie = subprocess.Popen(['yes | timeout 3 pcm-pcie.x'], shell=True, stdout=f)
                if p_pcm_pcie != None:
                    p_pcm_pcie.kill()
                    print_info("tried killing pcm-pcie.x")
                os.system('pkill pcm-pcie.x')
            with open(logdir+"nmi_status.txt", 'r') as f:
                lines = f.readlines()
                if len(lines) > 0:
                    if lines[0].find('Error: NMI watchdog is enabled.') != -1:
                        print_error('NMI watchdog is enabled., please try the command below:')
                        print_error('sudo sysctl -w kernel.nmi_watchdog=0')
#            output = subprocess.check_output('yes | timeout 3 pcm-pcie.x 2>&1', shell=True)
        except subprocess.CalledProcessError as e:
            print_warning("There was error while reading NMI status.")


    print_info('Clean previous logged files')
    subprocess.call('rm %s/perf.data > /dev/null 2> /dev/null' % logdir, shell=True )
    subprocess.call('rm %s/sofa.pcap > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace*.nvvp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.txt > /dev/null 2> /dev/null' % logdir, shell=True)


    try:
        print_info("Prolog of Recording...")

        if int(os.system('command -v nvprof')) == 0:
            p_nvprof = subprocess.Popen(['nvprof', '--profile-all-processes', '-o', logdir+'/gputrace%p.nvvp'])
            print_info('Launching nvprof')
            time.sleep(3)
            print_info('nvprof is launched')
        else:
            print_warning('Profile without NVPROF')

        if cfg.enable_pcm:
            with open(os.devnull, 'w') as FNULL:
                delay_pcie = 0.02
                #p_pcm_pcie = subprocess.Popen(['yes|pcm-pcie.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_pcie.csv -B '], shell=True)
                p_pcm_memory = subprocess.Popen(['yes|pcm-memory.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_memory.csv '], shell=True)
                #p_pcm_numa = subprocess.Popen(['yes|pcm-numa.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_numa.csv '], shell=True)

        print_info("Recording...")
        if cfg.profile_all_cpus == True:
            perf_options = '-a'
        else:
            perf_options = ''

        subprocess.call('cp /proc/kallsyms %s/' % (logdir), shell=True )
        subprocess.call('chmod +w %s/kallsyms' % (logdir), shell=True )

        # To improve perf timestamp accuracy
        print_info("Script path of SOFA: "+cfg.script_path)
        subprocess.call('%s/sofa_perf_timebase > %s/perf_timebase.txt' % (cfg.script_path,logdir), shell=True)
        subprocess.call('rm %s/*.nvvp' % (logdir), shell=True)
        subprocess.call('nvprof --profile-child-processes -o %s/cuhello%%p.nvvp -- perf record -o %s/cuhello.perf.data %s/cuhello' % (logdir,logdir,cfg.script_path), shell=True)

        # sofa_time is time base for mpstat, vmstat, nvidia-smi
        with open('%s/sofa_time.txt' % logdir, 'w') as logfile:
            unix_time = time.time()
            logfile.write(str('%.9lf'%unix_time)+'\n')

        with open('%s/mpstat.txt' % logdir, 'w') as logfile:
            p_mpstat = subprocess.Popen(
                    ['mpstat', '-P', 'ALL', '1'], stdout=logfile)

        with open('%s/vmstat.txt' % logdir, 'w') as logfile:
            p_vmstat = subprocess.Popen(['vmstat', '-w', '1'], stdout=logfile)
        #TODO:
        with open('%s/cpuinfo.txt' % logdir, 'w') as logfile:
            logfile.write('')
            timerThread = threading.Thread(target=service_get_cpuinfo, args=[logdir])
            timerThread.daemon = True
            timerThread.start()

        with open(os.devnull, 'w') as FNULL:
           p_tcpdump =  subprocess.Popen(["tcpdump",
                              '-i',
                              'any',
                              '-v',
                              'tcp',
                              '-w',
                              '%s/sofa.pcap' % logdir],
                             stderr=FNULL)

        if int(os.system('command -v nvidia-smi')) == 0:
            with open('%s/nvsmi.txt' % logdir, 'w') as logfile:
                p_nvsmi = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'u'], stdout=logfile)
            with open('%s/nvlink_topo.txt' % logdir, 'w') as logfile:
                p_nvtopo = subprocess.Popen(['nvidia-smi', 'topo', '-m'], stdout=logfile)



        if int(os.system('command -v perf')) == 0:
            ret = str(subprocess.check_output(['perf stat -e cycles ls 2>&1 '], shell=True))
            if ret.find('not supported') >=0:
                profile_command = 'perf record -o %s/perf.data -F %s %s -- %s' % (logdir, sample_freq, perf_options, command)
                cfg.perf_events = ""
            else:
                profile_command = 'perf record -o %s/perf.data -e %s -F %s %s -- %s' % (logdir, cfg.perf_events, sample_freq, perf_options, command)
            print_info(profile_command)
            subprocess.call(profile_command, shell=True)
            with open(logdir+'perf_events_used.txt','w') as f:
                f.write(cfg.perf_events)

        print_info("Epilog of Recording...")
        if p_tcpdump != None:
            p_tcpdump.terminate()
            print_info("tried terminating tcpdump")
        if p_vmstat != None:
            p_vmstat.terminate()
            print_info("tried terminating vmstat")
        if p_cpuinfo != None:
            p_cpuinfo.terminate()
            print_info("tried terminating cpuinfo")
        if p_mpstat != None:
            p_mpstat.terminate()
            print_info("tried terminating mpstat")
        if p_nvtopo != None:
            p_nvtopo.terminate()
            print_info("tried terminating nvidia-smi topo")
        if p_nvsmi != None:
            p_nvsmi.terminate()
            print_info("tried terminating nvidia-smi dmon")
        if p_nvprof != None:
            p_nvprof.terminate()
            print_info("tried terminating nvprof")
        if cfg.enable_pcm:
            kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa)
    except BaseException:
        print("Unexpected error:", sys.exc_info()[0])
        if p_tcpdump != None:
            p_tcpdump.kill()
            print_info("tried killing tcpdump")
        if p_vmstat != None:
            p_vmstat.kill()
            print_info("tried killing vmstat")
        if p_cpuinfo != None:
            p_cpuinfo.kill()
            print_info("tried killing cpuinfo")
        if p_mpstat != None:
            p_mpstat.kill()
            print_info("tried killing mpstat")
        if p_nvtopo != None:
            p_nvtopo.kill()
            print_info("tried killing nvidia-smi topo")
        if p_nvsmi != None:
            p_nvsmi.kill()
            print_info("tried killing nvidia-smi dmon")
        if p_nvprof != None:
            p_nvprof.kill()
            print_info("tried killing nvprof")
        if cfg.enable_pcm:
            kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa)
        raise
    print_info("End of Recording")
