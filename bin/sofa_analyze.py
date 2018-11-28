#!/usr/bin/env python3.6
import os
import sys
import pandas as pd
import numpy as np
import csv
import json
import random
from operator import itemgetter, attrgetter
import argparse
import multiprocessing as mp
from functools import partial
from sofa_print import *
from sofa_config import *
from sofa_common import *
from sofa_aisi import *
import networkx as nx
import re 
import requests

def payload_sum(df):
    print((len(df)))


class Event:

    def __init__(self, name, ttype, timestamp, duration):
        self.name = name
        self.ttype = ttype  # 0 for begin, 1 for end
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return repr((self.name, self.ttype, self.timestamp, self.duration))


def gpu_profile(logdir, cfg, df_gpu):
    total_kernel_time = 0.0
    total_gpu_time = 0.0

    print_title("Task Time (MEMCPY included) for each Device (s)")
    grouped_df = df_gpu.groupby("deviceId")["duration"]
    total_tasktime = 0
    for key, item in grouped_df:
        print(("[%d]: %lf" % (int(float(key)), grouped_df.get_group(key).sum())))
        total_tasktime = total_tasktime + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    per_gpu_time = total_tasktime / n_devices
    print(("Averaged GPU time of devices: %.2lf" % per_gpu_time))

    print_title("Data Traffic (bidirection) for each Device (MB)")
    grouped_df = df_gpu.groupby("deviceId")["payload"]
    for key, item in grouped_df:
        print(("[%d]: %lf" % (key, grouped_df.get_group(key).sum() / 1000000.0)))

    grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        if key == 0:
            total_kernel_time = grouped_df.get_group(key).sum()

    print_title("All-reduce Time (s)")
    all_reduce_time = 0
    grouped_df = df_gpu.groupby("name")["duration"]
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("AllReduce") != -1:
            all_reduce_time = all_reduce_time + grouped_df.get_group(key).sum()

    comm_profile(logdir, cfg, df_gpu)
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))

    print_title("Summary of Kernels")
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))
    print(("MeasuredAllReduceTime : %lf (s)" % all_reduce_time))
    get_top_k_events(df_gpu, 10)


def net_profile(logdir, cfg, df):
    print_title("Network Profiling: Communication Time (s)")
    grouped_df = df.groupby("name")["duration"]
    total_net_time = 0
    n_packets = 0
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("network:tcp:") != -1:
            total_net_time = total_net_time + grouped_df.get_group(key).sum()
            n_packets = n_packets + 1
    print(("total network time = %.3lf" % total_net_time))
    print(("total amount of network packets  = %d" % n_packets))


def cpu_profile(logdir, cfg, df):
    print_title("CPU Profiling: Task Time (IO included) for each Core (s)")
    grouped_df = df.groupby("deviceId")["duration"]
    total_exec_time = 0
    for key, item in grouped_df:
        if cfg.verbose:
            print(("[%d]: %lf" % (key, grouped_df.get_group(key).sum())))
        total_exec_time = total_exec_time + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    avg_exec_time = total_exec_time / n_devices
    print(("total execution time = %.3lf" % total_exec_time))
    print(("average execution time across devices = %.3lf" % avg_exec_time))

def vmstat_profile(logdir, cfg, df):
    print_title("VMSTAT Profiling:")
    df_name = df['name']
    
    vmstat_fieldnames = []
    fields = df_name.iloc[0].split('|')
    for field in fields:
       vmstat_fieldnames.append(field.split('=')[0]) 
    
    records = []
    for name in df_name:
        fields = name.split('|')
        row = []
        for field in fields:
            row.append(int(field.split('=')[1]))
        records.append(row)

    #1542188143.006416,-1,0.000010,-1,-1,-1,-1,-1,-1,-1,-1,r=1|b=0|sw=0|fr=12566580|bu=2140|ca=17433464|si=0|so=0|bi=0|bo=3|in=3|cs=2|usr=0|sys=0|idl=100|wa=0|st=0,-1
    vmstat_traces = pd.DataFrame(records)
    vmstat_traces.columns = vmstat_fieldnames

    print('sum of vmstat bi: ',vmstat_traces['bi'].sum())
    print('sum of vmstat bo: ',vmstat_traces['bo'].sum())
    print('max of vmstat wa (%%): %d' % vmstat_traces['wa'].max())
    print('mean of vmstat wa (%%): %.2lf' % vmstat_traces['wa'].mean())


#def mpstat_profile(logdir, cfg, df):
#    print_title("VMSTAT Profiling:")
#    df.rename(columns={'event': 'cpuid'}, inplace=True)
#    df.rename(columns={'copyKind': 'class'}, inplace=True)
#    df.rename(columns={'duration': 'usage'}, inplace=True)
#    z = {0: 'USR', 1: 'SYS', 2: 'IOW'}
#    df['class'] = df['class'].map(z)
#
#    gdf = df.groupby("cpuid")["usage"]
#    print("Number of Cores: %d" % (len(gdf) - 1))
#    gdf = df.groupby("class")["usage"]
#    print("Class\tMax.\tAvg.\tStd.")
#    for key, item in gdf:
#        print("%s\t%3d\t%3d\t%3d" % (key,
#                                     int(gdf.get_group(key).max()),
#                                     int(gdf.get_group(key).mean()),
#                                     int(gdf.get_group(key).std())))
#    print("For more info. about each core, please enable verbose mode.")
#
#    gdf = df.groupby("cpuid")["usage"]
#    if cfg.verbose:
#        print("===== Max. of Usages for Each Core =====")
#        table = df.pivot_table(
#            index='cpuid',
#            columns='class',
#            values='usage',
#            aggfunc=np.max)
#        print(table[1:].astype(int))
#
#        print("===== Avg. of Usages for Each Core =====")
#        table = df.pivot_table(
#            index='cpuid',
#            columns='class',
#            values='usage',
#            aggfunc=np.mean)
#        print(table[1:].astype(int))
#
#        print("===== Std. of Usages for Each Core =====")
#        table = df.pivot_table(
#            index='cpuid',
#            columns='class',
#            values='usage',
#            aggfunc=np.std)
#        print(table[1:].astype(int))


class ProfiledDomainDNN:
    domain_name = "DNN"
    prefix = "[ProfiledDomain%s]\t" % domain_name

    def __init__(self):
        self.name = "general"
        self.batch_size = 64
        self.iterations = 11
        self.throughput = 1
        self.avg_cpu_time = 1

    def get_batch_size(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                pos = line.find("--batch_size")
                if pos >= 0:
                    self.batch_size = int(line[pos:].split()[0].split('=')[1])
                    print((self.prefix + "batch_size: %d" % self.batch_size))
                    break

    def get_iterations(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                pos = line.find("--num_batches")
                if pos >= 0:
                    self.iterations = int(
                        line[pos:].split()[0].split('=')[1]) + 11
                    print((self.prefix + "iterations: %d" % self.iterations))
                    break

    def get_throughput(self, filepath):
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                if line.find("total images/sec:") != -1:
                    self.throughput = float(line.split()[2])
                    print((self.prefix + "Throughput: %.2lf" % self.throughput))
                    break


def sofa_analyze(logdir, cfg):
    filein = []
    df_gpu = []
    df_cpu = []
    df_vmstat = []

    filein_gpu = logdir + "gputrace.csv"
    filein_cpu = logdir + "cputrace.csv"
    filein_vmstat = logdir + "vmstat_trace.csv"
    
    if os.path.isfile('%s/nvlink_topo.txt' % logdir):
        
        with open(logdir + 'nvlink_topo.txt') as f:
            lines = f.readlines()
            if len(lines) > 0:
                title = lines[0]
                num_gpus = 1 
                for word in title.split():
                    if re.match(r'GPU', word) != None :
                       num_gpus = num_gpus + 1 
                print_info('# of GPUs: ' + str(num_gpus) )
                edges = []
                if len(lines) >= num_gpus+1:
                    for i in range(num_gpus):
                        connections = lines[1+i].split()
                        for j in range(len(connections)):
                            if connections[j] == 'NV1' or connections[j] == 'NV2':
                                edges.append((i,j-1))
                                #print('%d connects to %d' % (i, j-1))
                    
                    ring_found = False
                    G = nx.DiGraph(edges)           
                    # Try to find ring with its length of num_gpus
                    for cycle in nx.simple_cycles(G):
                        if len(cycle) == num_gpus:
                            print(("One of the recommended ring having length of %d" % len(cycle) ))
                            ring_found = True
                            os.system("mkdir -p sofalog/sofa_hints/")
                            xring_order = ','.join(map(str, cycle))
                            with open("sofalog/sofa_hints/xring_order.txt", "w") as f:
                                f.write('export CUDA_VISIBLE_DEVICES=' + xring_order)
                            break
                    
                    # Try to find ring with its length of num_gpus/2 
                    if not ring_found:
                        for cycle in nx.simple_cycles(G):
                            if len(cycle) == num_gpus/2:
                                print(("One of the recommended ring having length of %d" % len(cycle) ))
                                ring_found = True
                                os.system("mkdir -p sofalog/sofa_hints/")
                                xring_order = ','.join(map(str, cycle))
                                with open("sofalog/sofa_hints/xring_order.txt", "w") as f:
                                    f.write('export CUDA_VISIBLE_DEVICES=' + xring_order)
                                break   
    try:
        df_cpu = pd.read_csv(filein_cpu)
        df_vmstat = pd.read_csv(filein_vmstat)
        cpu_profile(logdir, cfg, df_cpu)
        net_profile(logdir, cfg, df_cpu)
        vmstat_profile(logdir, cfg, df_vmstat)
    except IOError:
        print_warning("cputrace.csv is not found")
        #quit()

    try:
        df_gpu = pd.read_csv(filein_gpu)
        #df_gpu.loc[:, 'timestamp'] -= df_gpu.loc[0, 'timestamp']
        gpu_profile(logdir, cfg, df_gpu)
        if cfg.enable_aisi:
            sofa_aisi(logdir, cfg, df_cpu, df_gpu)        
    except IOError:
        print_warning("gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

    if cfg.potato_server:
        print_title('POTATO Feedback')
        r = requests.get(cfg.potato_server+'/image/best')
        image_tag = r.json()
        print(image_tag)
    #print_warning('Something wrong with POTATO client')

    print('\n\n')
