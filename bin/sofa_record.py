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
from os import listdir
from os.path import isfile, join
import platform 

from sofa_print import *

def service_get_cpuinfo(logdir, cfg):
    next_call = time.time()
    while True:
        #print(datetime.datetime.now())
        next_call = next_call + (1 / float(cfg.sys_mon_rate));
        get_cpuinfo(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

def service_get_mpstat(logdir, cfg):
    next_call = time.time()
    while True:
        next_call = next_call + (1 / float(cfg.sys_mon_rate))
        get_mpstat(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

def service_get_diskstat(logdir, cfg):
    next_call = time.time()
    while True:
        next_call = next_call + (1 / float(cfg.sys_mon_rate))
        get_diskstat(logdir)
        time_remained = next_call - time.time()
        if time_remained > 0: 
            time.sleep(time_remained)

def service_get_netstat(logdir, interface, cfg):
    next_call = time.time()
    while True:
        next_call = next_call + (1 / float(cfg.sys_mon_rate))
        get_netstat(logdir, interface)
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
    # /proc/diskstats
    #
    # 0 1 2   3     4    5       6      7       8       9        10      11 12     13
    # 3 0 hda 43205 4113 4800428 280967 1051597 1682874 21876608 1950120 0  858685 2231096
    #
    # Field 3 -- # of reads issued
    # Field 5 -- # of sectors read
    # Field 6 -- # of milliseconds spent reading
                 # This is the total number of milliseconds spent by all reads (as
                 # measured from __make_request() to end_that_request_last()).
    # Field 7 -- # of writes completed
    # Field 9 -- # of sectors written
    # Field 10-- # of milliseconds spent writing
                 # This is the total number of milliseconds spent by all writes (as
                 # measured from __make_request() to end_that_request_last()).
    
    with open('/proc/diskstats','r') as f:
        lines = f.readlines()
        stat_list = []
        unix_time = time.time()
        for line in lines:
            m = line[:-1]
            m = line.split()
            stat_list.append([unix_time]+[m[2]]+[m[5]]+[m[9]]+[m[3]]+[m[7]]+[m[6]]+[m[10]])
        df_stat = pd.DataFrame(stat_list)
        df_stat.to_csv("%s/diskstat.txt" % logdir, mode='a', header=False, index=False, index_label=False)

def get_netstat(logdir, interface):
    if interface == '':
        return
    with open('/sys/class/net/%s/statistics/tx_bytes' %interface, 'r') as f:
        net_time = time.time()
        tx_line = f.readline().splitlines()
        [tx] = tx_line
    with open('/sys/class/net/%s/statistics/rx_bytes' %interface, 'r') as f:
        rx_line = f.readline().splitlines()
        [rx] = rx_line
    tt = [net_time, tx, rx]
    content = pd.DataFrame([tt], columns=['timestamp', 'tx_bytes', 'rx_bytes'])
    content.to_csv("%s/netstat.txt" % logdir, mode='a', header=False, index=False, index_label=False)

        
def sofa_clean(cfg):
    logdir = cfg.logdir
    print_info(cfg,'Clean previous logged files')
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.html > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.js > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.script > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.tmp > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)
    subprocess.call('rm %s/network_report.pdf > /dev/null 2> /dev/null' % logdir, shell=True, stderr=DEVNULL, stdout=DEVNULL)


def sofa_record(command, cfg):
    print_main_progress('SOFA recording...')
    p_perf = None
    p_tcpdump = None
    p_mpstat  = None
    p_diskstat = None
    p_netstat = None
    p_vmstat  = None
    p_blktrace  = None
    p_cpuinfo  = None
    p_nvprof  = None
    p_nvsmi   = None
    p_nvsmi_query = None
    p_nvtopo  = None
    logdir = cfg.logdir
    p_strace = None
    p_pystack = None
    print_info(cfg,'SOFA_COMMAND: %s' % command)
    sample_freq = 99
    command_prefix = ''

    sudo = ''
    if int(os.system('command -v sudo  1> /dev/null')) == 0:
        sudo = 'sudo '
 
    if subprocess.call(['mkdir', '-p', logdir]) != 0:
        print_error('Cannot create the directory' + logdir + ',which is needed for sofa logged files.' )
        sys.exit(1)
   
    subprocess.call('g++ ' + cfg.script_path + '/sofa_perf_timebase.cc -o ' + logdir + '/sofa_perf_timebase', shell=True)
    if not os.path.isfile(logdir + '/sofa_perf_timebase'):
        print_error(logdir + '/sofa_perf_timebase is not found')
        sys.exit(-1)

    subprocess.call('nvcc ' + cfg.script_path + '/cuhello.cu -o ' + logdir + '/cuhello', shell=True, stderr=DEVNULL)
    if not os.path.isfile(logdir + '/cuhelo'):
        print_warning(cfg, 'No nvcc found; nvcc is required to improve perf timestamp accuracy.') 
    
    if os.path.isfile("/proc/sys/kernel/kptr_restrict"):
        if int(open("/proc/sys/kernel/kptr_restrict").read()) != 0:
            print_error(
                "/proc/kallsyms permission is restricted, please try the command below:")
            print_error(sudo + "sysctl -w kernel.kptr_restrict=0")
            sys.exit(1)

    if os.path.isfile("/proc/sys/kernel/perf_event_paranoid"):
        if int(open("/proc/sys/kernel/perf_event_paranoid").read()) != -1:
            print_error('PerfEvent is not avaiable, please try the command below:')
            print_error(sudo + 'sysctl -w kernel.perf_event_paranoid=-1')
            sys.exit(1)

    print_info(cfg,'Clean previous logged files')
    # Not equal to sofa_clean(...) !!
    subprocess.call('rm ' + os.getcwd() + '/perf.data 1> /dev/null 2> /dev/null', shell=True )
    subprocess.call('rm %s/perf.data > /dev/null 2> /dev/null' % logdir, shell=True )
    subprocess.call('rm %s/cuhello.perf.data > /dev/null 2> /dev/null' % logdir, shell=True )
    subprocess.call('rm %s/sofa.pcap > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace*.nvvp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/cuhello*.nvvp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.csv > /dev/null 2> /dev/null' % logdir, shell=True)
    subprocess.call('rm %s/*.txt > /dev/null 2> /dev/null' % logdir, shell=True)
    if os.path.isfile('%s/container_root' % logdir):
        subprocess.call( sudo + 'umount %s/container_root' % logdir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        print_main_progress("Prologue of Recording...")
        if int(os.system('command -v nvprof 1> /dev/null')) == 0:
            p_nvprof = subprocess.Popen(['nvprof', '--profile-all-processes', '-o', logdir+'/gputrace%p.nvvp'], stderr=DEVNULL, stdout=DEVNULL)
            print_info(cfg,'Launching nvprof')
            time.sleep(3)
            print_info(cfg,'nvprof is launched')
        else:
            print_warning(cfg,'Profile without NVPROF')

        print_main_progress("Recording...")
        if cfg.profile_all_cpus == True:
            perf_options = '-a'
        else:
            perf_options = ''

        if os.path.isfile("/proc/kallsyms"):
            subprocess.call('cp /proc/kallsyms %s/' % (logdir), shell=True )
            subprocess.call('chmod +w %s/kallsyms' % (logdir), shell=True )

        print_info(cfg,"Script path of SOFA: " + cfg.script_path) 
        with open(logdir+'/perf_timebase.txt', 'w') as logfile:
            subprocess.call('%s/sofa_perf_timebase' % (logdir), shell=True, stderr=logfile, stdout=logfile)
        subprocess.call('nvprof --profile-child-processes -o %s/cuhello%%p.nvvp -- perf record -q -o %s/cuhello.perf.data %s/cuhello' % (logdir,logdir,cfg.script_path), shell=True, stderr=DEVNULL, stdout=DEVNULL)
        if int(os.system('perf 2>&1 1>/dev/null')) == 0:
            subprocess.call('nvprof --profile-child-processes -o %s/cuhello%%p.nvvp -- perf record -q -o %s/cuhello.perf.data %s/cuhello' % (logdir,logdir,cfg.script_path), shell=True, stderr=DEVNULL, stdout=DEVNULL)
        else:
            subprocess.call('nvprof --profile-child-processes -o %s/cuhello%%p.nvvp -- /usr/bin/time -v %s/cuhello' % (logdir,cfg.script_path), shell=True, stderr=DEVNULL, stdout=DEVNULL)

        # sofa_time is time base for vmstat, nvidia-smi
        with open('%s/sofa_time.txt' % logdir, 'w') as logfile:
            unix_time = time.time()
            logfile.write(str('%.9lf'%unix_time)+'\n')

        if subprocess.call('which vmstat', shell=True) == 0: 
            with open('%s/vmstat.txt' % logdir, 'w') as logfile:
                p_vmstat = subprocess.Popen(['vmstat', '-w', '1'], stdout=logfile)

        if cfg.blktrace_device is not None:
            p_blktrace = subprocess.Popen(sudo + 'blktrace --dev=%s -o trace -D %s' % (cfg.blktrace_device, logdir), stderr=DEVNULL, stdout=DEVNULL, shell=True)
            subprocess.call('echo "blktrace enabled"', shell=True)

        if os.path.isfile('/proc/cpuinfo'): 
            with open('%s/cpuinfo.txt' % logdir, 'w') as logfile:
                logfile.write('')
                timerThread = threading.Thread(target=service_get_cpuinfo, args=[logdir, cfg])
                timerThread.daemon = True
                timerThread.start()
        
        if subprocess.call('which mpstat', shell=True) == 0: 
            with open('%s/mpstat.txt' % logdir, 'w') as logfile:
                logfile.write('time,cpu,user,nice,system,idle,iowait,irq,softirq\n')
                timerThread = threading.Thread(target=service_get_mpstat, args=[logdir, cfg])
                timerThread.daemon = True
                timerThread.start()

        if os.path.isfile('/proc/diskstats'):
            with open('%s/diskstat.txt' % logdir, 'w') as logfile:
                logfile.write('')
                timerThread = threading.Thread(target=service_get_diskstat, args=[logdir, cfg])
                timerThread.daemon = True
                timerThread.start()

        if subprocess.call('which ip', shell=True) == 0:
            with open('%s/netstat.txt' % logdir, 'w') as logfile:
                logfile.write('')
                interface = subprocess.check_output("ip addr | awk '/state UP/{print $2}'", shell=True)
                interface = str(interface, 'utf-8')
                if cfg.netstat_interface is not None:
                    interface = cfg.netstat_interface
                else:
                    interface = interface.split(':')[0]
                timerThread = threading.Thread(target=service_get_netstat, args=[logdir, interface, cfg])
                timerThread.daemon = True
                timerThread.start()
        
        if cfg.enable_tcpdump:    
            with open(os.devnull, 'w') as FNULL:
               p_tcpdump =  subprocess.Popen(["tcpdump",
                                  '-i',
                                  'any',
                                  '-w',
                                  '%s/sofa.pcap' % logdir],
                                 stderr=FNULL)

        if int(os.system('command -v nvidia-smi 1>/dev/null')) == 0:
            with open('%s/nvsmi.txt' % logdir, 'w') as logfile:
                p_nvsmi = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'u'], stdout=logfile)
            with open('%s/nvsmi_query.txt' % logdir, 'w') as logfile:
                sample_time = 1 / float(cfg.sys_mon_rate)
                if sample_time >= 1:
                    p_nvsmi_query = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,gpu_name,index,utilization.gpu,utilization.memory',
                                            '-l', str(int(sample_time)), '--format=csv'], stdout=logfile)
                else:                                                                                                                                          
                    p_nvsmi_query = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,gpu_name,index,utilization.gpu,utilization.memory',
                                            '-lms', str(int(sample_time * 1000)), '--format=csv'], stdout=logfile)
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


        if cfg.enable_py_stacks:
            if command.find('python') == -1:
                print_warning(cfg,"Not a python program to recorded, skip recording callstacks")
            elif cfg.enable_strace:
                print_warning(cfg,"Only one of --enable_py_stacks or --enable_strace option holds, ignore --enable_py_stack options")
            else:
                # command_prefix = ' '.join(['py-spy','-n', '-s', '{}/pystacks.txt'.format(logdir), '-d', str(sys.maxsize), '--']) + ' '
                command_prefix  = ' '.join(['pyflame', '--flamechart', '-o', '{}pystacks.txt'.format(logdir), '-t']) + ' '
        

        if cfg.enable_strace:
            command_prefix = ' '.join(['strace', '-q', '-T', '-t', '-tt', '-f', '-o', '%s/strace.txt'%logdir]) + ' '

        if platform.platform().find('Darwin') != -1:
            print_warning(cfg,"Use /usr/bin/time to measure program performance instead of perf.")
            profile_command = '/usr/bin/time -l %s' % (command_prefix + command)
            cfg.perf_events = ""
        else:
            if cfg.no_perf_events or int(os.system('command -v perf 1> /dev/null')) != 0:
                print_warning(cfg,"Use /usr/bin/time to measure program performance instead of perf.")
                profile_command = '/usr/bin/time -v %s' % (command_prefix + command)
                cfg.perf_events = ""
            else:
                ret = str(subprocess.check_output(['perf stat -e cycles ls 2>&1 '], shell=True))
                if ret.find('not supported') >=0:
                    profile_command = 'perf record -o %s/perf.data -F %s %s %s' % (logdir, sample_freq, perf_options, command_prefix+command)
                    cfg.perf_events = ""
                else:
                    profile_command = 'perf record -o %s/perf.data -e %s -F %s %s %s' % (logdir, cfg.perf_events, sample_freq, perf_options, command_prefix+command) 
        
        with open(logdir+'perf_events_used.txt','w') as f:
            f.write(cfg.perf_events)

        subprocess.call('rm ' + os.getcwd() + '/perf.data 1> /dev/null 2> /dev/null', shell=True )

        # Launch SOFA recording
        if command.find('docker') != -1:
            command_create = command.replace('docker run', 'docker create --cidfile=%s/cidfile.txt' % cfg.logdir)
            cid = subprocess.check_output(command_create, shell=True).decode('utf-8')
            inspect_text = subprocess.check_output('docker inspect  '+cid, shell=True).decode('utf-8')
            inspect_info = json.loads(inspect_text)
            ccmd = ' '.join(inspect_info[0]['Config']['Cmd'])
            if subprocess.run('docker rm -f %s'%(cid), shell=True).returncode != 0:
                print('oops with remove container ', cid)
                sys.exit(-1)
            
            if os.path.isfile(cfg.logdir+'/cidfile.txt'):
                subprocess.run('rm '+cfg.logdir+'/cidfile.txt', shell=True)
           
            # Step1: launch sleep process to keep the container alive 
            command_sleep = command.replace(ccmd, 'sleep 600')
            command_sleep = command_sleep.replace('docker run', 'docker run -u 1000:1000 --cidfile=%s/cidfile.txt -v %s:/sofalog ' % (cfg.logdir, cfg.logdir))
            print_hint(command_sleep)
            p_container_sleep = subprocess.Popen(command_sleep.split())
            time.sleep(1)
            
            if os.path.isfile(cfg.logdir+'/cidfile.txt'):
                with open(cfg.logdir+'/cidfile.txt', 'r') as cidfile: 
                    cid = cidfile.readlines()[0]
                  
                    # Step2: launch containerized application 
                    if cfg.nvprof_inside:
                        ccmd = '/usr/local/cuda/bin/nvprof --profile-child-processes -o /sofalog/gputrace001001%p.nvvp ' + ccmd 
                    app_command = 'docker exec %s %s' % (cid, ccmd)
                    print_hint(app_command)
                    p_container_app = subprocess.Popen(app_command.split())
            
                    # Step3: launch perf profiling process 
                    profile_command = 'perf record -o %s/perf.data -a -e cpu-clock --cgroup=docker/%s sleep 30' % (cfg.logdir, cid) 
                    print_hint('Profiling Command : ' + profile_command)
                    p_perf = subprocess.Popen(profile_command, shell=True)
            else:
                print('Cannot find cidfile.')
                sys.exit(-1)
        else:    
            print_hint('Profiling Command : ' + profile_command)
            if platform.platform().find('Darwin') != -1:
                print_hint('On Darwin, press Ctrl+C to continue if being blocked for a while.')
            with open(cfg.logdir + '/sofa.err', 'w') as errfile:
                p_perf = subprocess.Popen(profile_command, shell=True, stderr=errfile)
        
        try:
            if os.path.isfile(cfg.logdir+'/cidfile.txt'):
                p_container_app.wait() 
            p_perf.wait()
            t_command_end = time.time()
        except TimeoutExpired:
            print_error('perf: Timeout of profiling process')
            sys.exit(1)

        with open('%s/misc.txt' % logdir, 'w') as f_misc:
            vcores = 1
            cores = 1
            if os.path.isfile('/proc/cpuinfo'): 
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

        print_main_progress("Epilogue of Recording...")
        if p_tcpdump != None:
            p_tcpdump.terminate()
            print_info(cfg,"tried terminating tcpdump")
        if p_vmstat != None:
            p_vmstat.terminate()
            print_info(cfg,"tried terminating vmstat")
        if p_blktrace != None:
            #TODO: seek for a elegant killing solution 
            subprocess.call(sudo + 'pkill blktrace', shell=True)
            if cfg.blktrace_device is not None:
                os.system(sudo + 'blkparse -i %s/trace.blktrace.* -o %s/blktrace.txt -d %s/blktrace.out > /dev/null' % (logdir, logdir, logdir))
                os.system('rm -rf %s/trace.blktrace.*' % logdir)
            print_info(cfg,"tried terminating blktrace")
        if p_cpuinfo != None:
            p_cpuinfo.terminate()
            print_info(cfg,"tried terminating cpuinfo")
        if p_mpstat != None:
            p_mpstat.terminate()
            print_info(cfg,"tried terminating mpstat")
        if p_diskstat != None:
            p_diskstat.terminate()
            print_info(cfg,"tried terminating diskstat")
        if p_netstat != None:
            p_netstat.terminate()
            print_info(cfg,"tried terminating netstat")
        if p_nvtopo != None:
            p_nvtopo.terminate()
            print_info(cfg,"tried terminating nvidia-smi topo")
        if p_nvsmi != None:
            if p_nvsmi.poll() is None:
                p_nvsmi.terminate()
                print_info(cfg,"tried terminating nvidia-smi dmon")
            else:
                open('%s/nvsmi.txt' % logdir, 'a').write('\nFailed\n')
        if p_nvsmi_query != None:
            if p_nvsmi_query.poll() is None:
                p_nvsmi_query.terminate()
                print_info(cfg,"tried terminating nvidia-smi query")
            else:
                open('%s/nvsmi_query.txt' % logdir, 'a').write('\nFailed\n')
        if p_nvprof != None:
            p_nvprof.terminate()
            print_info(cfg,"tried terminating nvprof")
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
        if p_blktrace != None:
            #TODO: seek for a elegant killing solution 
            subprocess.call(sudo + 'pkill blktrace', shell=True)
            if cfg.blktrace_device is not None:
                os.system(sudo + 'blkparse -i %s/trace.blktrace.* -o %s/blktrace.txt -d %s/blktrace.out > /dev/null' % (logdir, logdir, logdir))
                os.system('rm -rf %s/trace.blktrace.*' % logdir)
            print_info(cfg,"tried terminating blktrace")
        if p_cpuinfo != None:
            p_cpuinfo.kill()
            print_info(cfg,"tried killing cpuinfo")
        if p_mpstat != None:
            p_mpstat.kill()
            print_info(cfg,"tried killing mpstat")
        if p_diskstat != None:
            p_diskstat.kill()
            print_info(cfg,"tried killing diskstat")
        if p_netstat != None:
            p_netstat.kill()
            print_info(cfg, "tried killing netstat")
        if p_nvtopo != None:
            p_nvtopo.kill()
            print_info(cfg,"tried killing nvidia-smi topo")
        if p_nvsmi != None:
            p_nvsmi.kill()
            print_info(cfg,"tried killing nvidia-smi dmon")
        if p_nvsmi_query != None:
            p_nvsmi_query.kill()
            print_info(cfg,"tried killing nvidia-smi query")
        if p_nvprof != None:
            p_nvprof.kill()
            print_info(cfg,"tried killing nvprof")
        if p_strace != None:
            p_strace.kill()
            print_info(cfg,"tried killing strace")

        raise
    print_main_progress("End of Recording")
