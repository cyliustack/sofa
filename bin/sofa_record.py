#!/usr/bin/python
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp
import glob
import os
from functools import partial
from sofa_print import *
import subprocess
from time import sleep


def sofa_record(command, logdir, cfg):

    print_info('SOFA_COMMAND: %s' % command)
    sample_freq = 99
    if int(subprocess.check_output(
            ["cat", "/proc/sys/kernel/kptr_restrict"])) != 0:
        print_error(
            "/proc/kallsyms permission is restricted, please try the command below:")
        print_error("sudo sysctl -w kernel.kptr_restrict=0")
        quit()

    if cfg.profile_all_cpus:
        if int(subprocess.check_output(
                ["cat", "/proc/sys/kernel/perf_event_paranoid"])) != -1:
            print_error('PerfEvent is not avaiable, please try the command below:')
            print_error('sudo sysctl -w kernel.perf_event_paranoid=-1')
            quit()

    subprocess.call(['mkdir', '-p', logdir])
    os.system('rm %s/perf.data > /dev/null 2> /dev/null' % logdir)
    os.system('rm %s/sofa.pcap > /dev/null 2> /dev/null' % logdir)
    os.system('rm %s/gputrace*.nvvp > /dev/null 2> /dev/null' % logdir)
    os.system('rm %s/gputrace.tmp > /dev/null 2> /dev/null' % logdir)
    os.system('rm %s/*.csv > /dev/null 2> /dev/null' % logdir)
    os.system('rm %s/*.txt > /dev/null 2> /dev/null' % logdir)
    try:
        print_info("Prolog of Recording...")
        with open(os.devnull, 'w') as FNULL:
            subprocess.Popen(["tcpdump",
                              '-i',
                              'any',
                              '-v',
                              'tcp',
                              '-w',
                              '%s/sofa.pcap' % logdir],
                             stderr=FNULL)
        with open('%s/mpstat.txt' % logdir, 'w') as logfile:
            subprocess.Popen(
                ['mpstat', '-P', 'ALL', '1', '600'], stdout=logfile)
        with open('%s/vmstat.txt' % logdir, 'w') as logfile:
            subprocess.Popen(['vmstat', '-w', '1', '600'], stdout=logfile)
        if int(os.system('command -v nvprof')) == 0:
            with open('%s/nvsmi.txt' % logdir, 'w') as logfile:
                subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'u'], stdout=logfile)
            with open('%s/nvlink_topo.txt' % logdir, 'w') as logfile:
                subprocess.Popen(['nvidia-smi', 'topo', '-m'], stdout=logfile)  
        with open('%s/sofa_time.txt' % logdir, 'w') as logfile:
            subprocess.Popen(['date', '+%s'], stdout=logfile)

        print_info("Recording...")

        if cfg.profile_all_cpus == True:
            perf_options = '-a'
        else:
            perf_options = ''

            
        os.system('cp /proc/kallsyms %s/' % (logdir) )
        if int(os.system('command -v nvprof')) == 0:
            profile_command = 'nvprof --profile-child-processes -o %s/gputrace%%p.nvvp perf record -e cycles,bus-cycles -o %s/perf.data -F %s %s -- %s ' % (logdir, logdir, sample_freq, perf_options, command)
        else:
            print_warning('Profile without NVPROF')
            profile_command = 'perf record -o %s/perf.data -e cycles,bus-cycles -F %s %s -- %s' % (logdir, sample_freq, perf_options, command)
        print_info( profile_command)
        os.system( profile_command)
        print_info("Epilog of Recording...")
        os.system('pkill tcpdump')
        os.system('pkill mpstat')
        os.system('pkill vmstat')
        os.system('pkill nvidia-smi')
    except BaseException:
        print "Unexpected error:", sys.exc_info()[0]
        while os.system('pkill tcpdump')  != 0 or \
                os.system('pkill mpstat') != 0 or \
                os.system('pkill vmstat') != 0 or \
                os.system('pkill nvidia-smi') != 0:
            print_warning(
                "Try to kill tcpdump, mpstat, vmstat and nvidia-smi. If not, Ctrl+C to stop the action.")
            sleep(0.5)
        print_info("tcpdump, mpstat, vmstat and nvidia-smi are killed.")
        raise
    print_info("End of Recording")
