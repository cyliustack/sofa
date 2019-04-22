import argparse
import csv
import datetime
import glob
import json
import multiprocessing as mp
import os
import subprocess
from subprocess import DEVNULL
from subprocess import PIPE
import sys
import threading
import time
from functools import partial
from pwd import getpwuid
import pandas as pd
import numpy as np
import re

from sofa_print import *


def service_get_cpuinfo(logdir):
    next_call = time.time()
    while True:
        #print(datetime.datetime.now())
        next_call = next_call + 0.1;
        get_cpuinfo(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

def service_get_mpstat(logdir):
    next_call = time.time()
    while True:
        next_call = next_call + 0.1;
        get_mpstat(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

def service_get_diskstat(logdir):
    next_call = time.time()
    while True:
        next_call = next_call + 0.1;
        get_diskstat(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

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

def get_mpstat(logdir):
    with open('/proc/stat','r') as f:
        lines = f.readlines()
        stat_list = []
        unix_time = time.time()
        cpu_id = -1
        for line in lines:
            if line.find('cpu') != -1: 
                #cpu, user，nice, system, idle, iowait, irq, softirq
                #example: cat /proc/stat 
                #   cpu  36572 0 10886 2648245 1047 0 155 0 0 0
                #   cpu0 3343 0 990 332364 177 0 80 0 0 0
                m = line.split()
                stat_list.append([unix_time,cpu_id]+m[1:8])
                cpu_id = cpu_id + 1 
            else:
                break
        stat = np.array(stat_list) 
        df_stat = pd.DataFrame(stat)
        df_stat.to_csv("%s/mpstat.txt" % logdir, mode='a', header=False, index=False, index_label=False)

def get_diskstat(logdir):
    with open('/proc/diskstats','r') as f:
        lines = f.readlines()
        stat_list = []
        unix_time = time.time()
        for line in lines:
            m = line[:-1]
            m = line.split()
            stat_list.append([unix_time]+[m[2]]+[m[5]]+[m[9]])
        df_stat = pd.DataFrame(stat_list)
        df_stat.to_csv("%s/diskstat.txt" % logdir, mode='a', header=False, index=False, index_label=False)

def kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa):
    if p_pcm_pcie != None:
        p_pcm_pcie.terminate()
        os.system('yes|pkill pcm-pcie.x')
        print_info(cfg,"tried killing pcm-pcie.x")
    if p_pcm_memory != None:
        p_pcm_memory.terminate()
        os.system('yes|pkill pcm-memory.x')
        print_info(cfg,"tried killing pcm-memory.x")
    if p_pcm_numa != None:
        p_pcm_numa.terminate()
        os.system('yes|pkill pcm-numa.x')
        print_info(cfg,"tried killing pcm-numa.x")


def sofa_clean(cfg):
    logdir = cfg.logdir
    print_info(cfg,'Clean previous logged files')
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.html > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.js > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.script > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.tmp > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)


def sofa_record(command, cfg):

    p_perf = None
    p_tcpdump = None
    p_mpstat  = None
    p_diskstat = None
    p_vmstat  = None
    p_cpuinfo  = None
    p_nvprof  = None
    p_nvsmi   = None
    p_nvtopo  = None
    p_pcm_pcie = None
    p_pcm_memory = None
    p_pcm_numa = None 
    logdir = cfg.logdir
    p_strace = None
    print_info(cfg,'SOFA_COMMAND: %s' % command)
    sample_freq = 99
    command_prefix = ''

    if int(open("/proc/sys/kernel/yama/ptrace_scope").read()) != 0:
        print_error(
            "Could not attach to process, please try the command below:")
        print_error("sudo sysctl -w kernel.yama.ptrace_scope=0")
        sys.exit(1)

    if int(open("/proc/sys/kernel/kptr_restrict").read()) != 0:
        print_error(
            "/proc/kallsyms permission is restricted, please try the command below:")
        print_error("sudo sysctl -w kernel.kptr_restrict=0")
        sys.exit(1)

    if int(open("/proc/sys/kernel/perf_event_paranoid").read()) != -1:
        print_error('PerfEvent is not avaiable, please try the command below:')
        print_error('sudo sysctl -w kernel.perf_event_paranoid=-1')
        sys.exit(1)

    if cfg.enable_pcm:
        print_info(cfg,'Test Capability of PCM programs ...')
        ret = str(subprocess.check_output(['getcap `which pcm-memory.x`'], shell=True))
        if ret.find('cap_sys_rawio+ep') == -1:
            print_error('To read/write MSR in userspace is not avaiable, please try the commands below:')
            print_error('sudo modprobe msr')
            print_error('sudo setcap cap_sys_rawio=ep `which pcm-memory.x`')
            sys.exit(1)

    if subprocess.call(['mkdir', '-p', logdir]) != 0:
        print_error('Cannot create the directory' + logdir + ',which is needed for sofa logged files.' )
        sys.exit(1)

        print_info(cfg,'Read NMI watchlog status ...')
        nmi_output = ""
        try:
            with open(logdir+"nmi_status.txt", 'w') as f:
                p_pcm_pcie = subprocess.Popen(['yes | timeout 3 pcm-pcie.x'], shell=True, stdout=f)
                if p_pcm_pcie != None:
                    p_pcm_pcie.kill()
                    print_info(cfg,"tried killing pcm-pcie.x")
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


    print_info(cfg,'Clean previous logged files')
    # Not equal to sofa_clean(...) !!
    subprocess.call('rm %s/perf.data > /dev/null 2> /dev/null' % logdir, shell=True )
    subprocess.call('rm %s/cuhello.perf.data > /dev/null 2> /dev/null' % logdir, shell=True )
    subprocess.call('rm %s/sofa.pcap > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace*.nvvp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/cuhello*.nvvp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.txt > /dev/null 2> /dev/null' % logdir, shell=True)


    try:
        print_progress("Prologue of Recording...")
        if int(os.system('command -v nvprof 1> /dev/null')) == 0:
            p_nvprof = subprocess.Popen(['nvprof', '--profile-all-processes', '-o', logdir+'/gputrace%p.nvvp'], stderr=DEVNULL, stdout=DEVNULL)
            print_info(cfg,'Launching nvprof')
            time.sleep(3)
            print_info(cfg,'nvprof is launched')
        else:
            print_warning('Profile without NVPROF')

        if cfg.enable_pcm:
            with open(os.devnull, 'w') as FNULL:
                delay_pcie = 0.02
                #p_pcm_pcie = subprocess.Popen(['yes|pcm-pcie.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_pcie.csv -B '], shell=True)
                p_pcm_memory = subprocess.Popen(['yes|pcm-memory.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_memory.csv '], shell=True)
                #p_pcm_numa = subprocess.Popen(['yes|pcm-numa.x ' + str(delay_pcie) + ' -csv=sofalog/pcm_numa.csv '], shell=True)

        print_progress("Recording...")
        if cfg.profile_all_cpus == True:
            perf_options = '-a'
        else:
            perf_options = ''

        subprocess.call('cp /proc/kallsyms %s/' % (logdir), shell=True )
        subprocess.call('chmod +w %s/kallsyms' % (logdir), shell=True )

        print_info(cfg,"Script path of SOFA: "+cfg.script_path)
        with open(logdir+'/perf_timebase.txt', 'w') as logfile:
            subprocess.call('%s/sofa_perf_timebase' % (cfg.script_path), shell=True, stderr=logfile, stdout=logfile)
        subprocess.call('nvprof --profile-child-processes -o %s/cuhello%%p.nvvp -- perf record -q -o %s/cuhello.perf.data %s/cuhello' % (logdir,logdir,cfg.script_path), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # sofa_time is time base for vmstat, nvidia-smi
        with open('%s/sofa_time.txt' % logdir, 'w') as logfile:
            unix_time = time.time()
            logfile.write(str('%.9lf'%unix_time)+'\n')

        with open('%s/vmstat.txt' % logdir, 'w') as logfile:
            p_vmstat = subprocess.Popen(['vmstat', '-w', '1'], stdout=logfile)

        with open('%s/cpuinfo.txt' % logdir, 'w') as logfile:
            logfile.write('')
            timerThread = threading.Thread(target=service_get_cpuinfo, args=[logdir])
            timerThread.daemon = True
            timerThread.start()
        
        with open('%s/mpstat.txt' % logdir, 'w') as logfile:
            logfile.write('time,cpu,user,nice,system,idle,iowait,irq,softirq\n')
            timerThread = threading.Thread(target=service_get_mpstat, args=[logdir])
            timerThread.daemon = True
            timerThread.start()

        with open('%s/diskstat.txt' % logdir, 'w') as logfile:
            logfile.write('')
            timerThread = threading.Thread(target=service_get_diskstat, args=[logdir])
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

        if int(os.system('command -v nvidia-smi 1>/dev/null')) == 0:
            with open('%s/nvsmi.txt' % logdir, 'w') as logfile:
                p_nvsmi = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'u'], stdout=logfile)
            with open('%s/nvlink_topo.txt' % logdir, 'w') as logfile:
                p_nvtopo = subprocess.Popen(['nvidia-smi', 'topo', '-m'], stdout=logfile)

        # Primary Profiled Program
            
        if cfg.pid > 0 :
            target_pid = cfg.pid 
        else:
            target_pid = -1
        
        t_command_begin = time.time()
        print_hint('PID of the target program: %d' % target_pid)
        print_hint('Command: %s' % command)


        if cfg.enable_strace:
            command_prefix = ' '.join(['strace', '-q', '-T', '-t', '-tt', '-f', '-o', '%s/strace.txt'%logdir]) + ' '

        if int(os.system('command -v perf 1>/dev/null')) == 0:
            ret = str(subprocess.check_output(['perf stat -e cycles ls 2>&1 '], shell=True))
            if ret.find('not supported') >=0:
                profile_command = 'perf record -o %s/perf.data -F %s %s %s' % (logdir, sample_freq, perf_options, command_prefix+command)
                cfg.perf_events = ""
            else:
                profile_command = 'perf record -o %s/perf.data -e %s -F %s %s %s' % (logdir, cfg.perf_events, sample_freq, perf_options, command_prefix+command) 
            with open(logdir+'perf_events_used.txt','w') as f:
                f.write(cfg.perf_events)

            print_hint(profile_command)
            p_perf = subprocess.Popen(profile_command, shell=True, stderr=DEVNULL, stdout=DEVNULL)
        
        try:
            p_perf.wait()
            t_command_end = time.time()
        except TimeoutExpired:
            print_error('perf: Timeout of profiling process')
            sys.exit(1)

        with open('%s/misc.txt' % logdir, 'w') as f_misc:
            vcores = 0
            cores = 0
            with open('/proc/cpuinfo','r') as f:
                lines = f.readlines()
                vcores = 0
                cores = 0
                for line in lines:
                    if line.find('cpu cores') != -1:
                        cores = int(line.split()[3])
                        vcores = vcores + 1
            f_misc.write('elapsed_time %.6lf\n' % (t_command_end - t_command_begin))
            f_misc.write('cores %d\n' % (cores))
            f_misc.write('vcores %d\n' % (vcores))
            f_misc.write('pid %d\n' % (target_pid))

        print_progress("Epilogue of Recording...")
        if p_tcpdump != None:
            p_tcpdump.terminate()
            print_info(cfg,"tried terminating tcpdump")
        if p_vmstat != None:
            p_vmstat.terminate()
            print_info(cfg,"tried terminating vmstat")
        if p_cpuinfo != None:
            p_cpuinfo.terminate()
            print_info(cfg,"tried terminating cpuinfo")
        if p_mpstat != None:
            p_mpstat.terminate()
            print_info(cfg,"tried terminating mpstat")
        if p_diskstat != None:
            p_diskstat.terminate()
            print_info(cfg,"tried terminating diskstat")
        if p_nvtopo != None:
            p_nvtopo.terminate()
            print_info(cfg,"tried terminating nvidia-smi topo")
        if p_nvsmi != None:
            if p_nvsmi.poll() is None:
                p_nvsmi.terminate()
                print_info(cfg,"tried terminating nvidia-smi dmon")
            else:
                open('%s/nvsmi.txt' % logdir, 'a').write('\nFailed\n')
        if p_nvprof != None:
            p_nvprof.terminate()
            print_info(cfg,"tried terminating nvprof")
        if cfg.enable_pcm:
            kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa)
        if p_strace != None:
            p_strace.terminate()
            print_info(cfg,"tried terminating strace")
    except BaseException:
        print("Unexpected error:", sys.exc_info()[0])
        if p_tcpdump != None:
            p_tcpdump.kill()
            print_info(cfg,"tried killing tcpdump")
        if p_vmstat != None:
            p_vmstat.kill()
            print_info(cfg,"tried killing vmstat")
        if p_cpuinfo != None:
            p_cpuinfo.kill()
            print_info(cfg,"tried killing cpuinfo")
        if p_mpstat != None:
            p_mpstat.kill()
            print_info(cfg,"tried killing mpstat")
        if p_diskstat != None:
            p_diskstat.kill()
            print_info(cfg,"tried killing diskstat")
        if p_nvtopo != None:
            p_nvtopo.kill()
            print_info(cfg,"tried killing nvidia-smi topo")
        if p_nvsmi != None:
            p_nvsmi.kill()
            print_info(cfg,"tried killing nvidia-smi dmon")
        if p_nvprof != None:
            p_nvprof.kill()
            print_info(cfg,"tried killing nvprof")
        if cfg.enable_pcm:
            kill_pcm_modules(p_pcm_pcie, p_pcm_memory, p_pcm_numa)
        if p_strace != None:
            p_strace.kill()
            print_info(cfg,"tried killing strace")

        raise
    print_progress("End of Recording")
