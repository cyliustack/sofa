import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import re
import sys
from functools import partial
from operator import attrgetter, itemgetter
import networkx as nx
import numpy as np
import pandas as pd
import time
from sofa_aisi import *
from sofa_common import *
from sofa_config import *
from sofa_hsg import *
from sofa_print import *


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


def gpu_profile(logdir, cfg, df_gpu, features):
    print_title("GPU Profiling")
    print('Per-GPU time (s):')
    groups = df_gpu.groupby("deviceId")["duration"]
    total_gpu_time = 0
    for key, item in groups:
        gpuid = int(float(key))
        per_gpu_time = groups.get_group(key).sum()
        print("[%d]: %lf" % (gpuid, per_gpu_time))
        total_gpu_time = total_gpu_time + per_gpu_time 
    n_gpus = len(groups)
    mean_gpu_time = total_gpu_time / n_gpus
    print(("Total GPU time of all GPUs (s) = %.3lf" % total_gpu_time))
    print(("Mean per-GPU time of all GPUs (s) = %.3lf" % mean_gpu_time))
   
    total_kernel_time = 0
    grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        if key == 0:
            total_kernel_time = grouped_df.get_group(key).sum()

    all_reduce_time = 0
    grouped_df = df_gpu.groupby("name")["duration"]
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("AllReduce") != -1:
            all_reduce_time = all_reduce_time + grouped_df.get_group(key).sum()

    features = comm_profile(logdir, cfg, df_gpu, features)

    get_top_k_events(df_gpu, 10)
    df = pd.DataFrame({ 'name':['total_gpu_time', 'mean_gpu_time', 'total_kernel_time', 'all_reduce_time'], 
                        'value':[total_gpu_time, mean_gpu_time, total_kernel_time, all_reduce_time] }, 
                        columns=['name','value'])
    features = pd.concat([features, df])
    return features

def net_profile(logdir, cfg, df):
    print_title("Network Profiling:")
    grouped_df = df.groupby("name")["duration"]
    total_net_time = 0
    n_packets = 0
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("network:tcp:") != -1:
            total_net_time = total_net_time + grouped_df.get_group(key).sum()
            n_packets = n_packets + 1
    print(("total network time (s) = %.3lf" % total_net_time))
    print(("total amount of network packets  = %d" % n_packets))


def cpu_profile(logdir, cfg, df):
    print_title("CPU Profiling:")
    print('elapsed_time (s) = %.6lf' % cfg.elapsed_time) 
    grouped_df = df.groupby("deviceId")["duration"]
    total_exec_time = 0
    for key, item in grouped_df:
        if cfg.verbose:
            print(("[%d]: %lf" % (key, grouped_df.get_group(key).sum())))
        total_exec_time = total_exec_time + grouped_df.get_group(key).sum()
    
    print("total execution time (s) = %.3lf" % total_exec_time)

    cpu_detail_profile_df = df[['timestamp','duration','name']]
    cpu_detail_profile_df = cpu_detail_profile_df.sort_values(by=['duration'], ascending=False)
    cpu_detail_profile_df['ratio(%)'] = cpu_detail_profile_df['duration']/total_exec_time * 100
    cpu_detail_profile_df = cpu_detail_profile_df[['timestamp','ratio(%)','duration','name']]
    print(cpu_detail_profile_df[:20].to_string(index=False))


def potato_client(logdir, cfg, df_cpu, df_gpu, df_vmstat, iter_summary):
    print_title("POTATO Client")


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

def mpstat_topdown(cfg, df_mpstat, features):

    return features

def mpstat_profile(logdir, cfg, df, features):
    print_title("MPSTAT Profiling:")
    n_cores = int(df['deviceId'].max() + 1)
    df_summary = pd.DataFrame( np.zeros((n_cores,5)), columns=['USR','SYS','IDL','IOW','IRQ'])
    for i in range(len(df)):
        dt = df.iloc[i]['duration']
        core = int(df.iloc[i]['deviceId'])
        fields = df.loc[i,'name'].split('|')
        r_usr = float(fields[5])
        r_sys = float(fields[6])
        r_idl = float(fields[7])
        r_iow = float(fields[8])
        r_irq = float(fields[9])
        if r_idl == 100:
            dt_all = 0.1
        else:
            dt_all = dt/((100-r_idl)/100.0)
        t_usr = dt_all * r_usr/100.0
        t_sys = dt_all * r_sys/100.0
        t_idl = dt_all * r_idl/100.0
        t_iow = dt_all * r_iow/100.0
        t_irq = dt_all * r_irq/100.0
        df_summary.iloc[core]['USR'] = df_summary.iloc[core]['USR'] + t_usr 
        df_summary.iloc[core]['SYS'] = df_summary.iloc[core]['SYS'] + t_sys 
        df_summary.iloc[core]['IDL'] = df_summary.iloc[core]['IDL'] + t_idl 
        df_summary.iloc[core]['IOW'] = df_summary.iloc[core]['IOW'] + t_iow 
        df_summary.iloc[core]['IRQ'] = df_summary.iloc[core]['IRQ'] + t_irq 
    
    print('CPU Utilization (%):')
    print('core\tUSR\tSYS\tIDL\tIOW\tIRQ')
    for i in range(len(df_summary)):
        t_sum = df_summary.iloc[i].sum() 
        print('%3d\t%3d\t%3d\t%3d\t%3d\t%3d'%(i,int(100.0*df_summary.iloc[i]['USR']/t_sum),
                                                int(100.0*df_summary.iloc[i]['SYS']/t_sum),
                                                int(100.0*df_summary.iloc[i]['IDL']/t_sum),
                                                int(100.0*df_summary.iloc[i]['IOW']/t_sum),
                                                int(100.0*df_summary.iloc[i]['IRQ']/t_sum) ))
    print('CPU Time (s):')
    print('core\tUSR\tSYS\tIDL\tIOW\tIRQ')
    for i in range(len(df_summary)):
        t_sum = df_summary.iloc[i].sum() 
        print('%3d\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf'%(i,
                                                df_summary.iloc[i]['USR'],
                                                df_summary.iloc[i]['SYS'],
                                                df_summary.iloc[i]['IDL'],
                                                df_summary.iloc[i]['IOW'],
                                                df_summary.iloc[i]['IRQ'] ))

    total_cpu_time = df_summary[['USR','SYS','IOW','IRQ']].sum().sum()
    print('Active CPU Time (s): %.3lf' % total_cpu_time) 
    active_cpu_ratio = int(100*total_cpu_time / (n_cores*cfg.elapsed_time))
    print('Active CPU ratio (%%): %3d' % active_cpu_ratio)
    df_feature = pd.DataFrame({ 'name':['active_cpu_ratio'], 
                        'value':[active_cpu_ratio] }, 
                        columns=['name','value'])
    features = pd.concat([features, df_feature])   
    return features

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


def sofa_analyze(cfg):
    filein = []
    df_cpu = pd.DataFrame([], columns=cfg.columns)
    df_gpu = pd.DataFrame([], columns=cfg.columns)
    df_net = pd.DataFrame([], columns=cfg.columns)
    df_mpstat = pd.DataFrame([], columns=cfg.columns)
    df_vmstat = pd.DataFrame([], columns=cfg.columns)
    iter_summary = None
    logdir = cfg.logdir

    with open(logdir+'/misc.txt') as f:
        lines = f.readlines()
        elapsed_time = float(lines[0].split()[1])
        vcores = int(lines[2].split()[1])
        cfg.elapsed_time = float(lines[0].split()[1])
    
    filein_gpu = logdir + "gputrace.csv"
    filein_cpu = logdir + "cputrace.csv"
    filein_net = logdir + "nettrace.csv"
    filein_vmstat = logdir + "vmstat.csv"
    filein_mpstat = logdir + "mpstat.csv"
    filein_strace = logdir + "strace.csv"

    if os.path.isfile('%s/nvlink_topo.txt' % logdir):

        with open(logdir + 'nvlink_topo.txt') as f:
            lines = f.readlines()
            if len(lines) > 0:
                title = lines[0]
                num_gpus = 1
                for word in title.split():
                    if re.match(r'GPU', word) != None :
                       num_gpus = num_gpus + 1
                print_info(cfg,'# of GPUs: ' + str(num_gpus) )
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
    # Construct Performance Features
    features = pd.DataFrame({'name':['elapsed_time'], 'value':[cfg.elapsed_time]}, columns=['name','value'])
    
    
    try:
        df_cpu = pd.read_csv(filein_cpu)
        cpu_profile(logdir, cfg, df_cpu)
    except IOError as e:
        df_cpu = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_cpu)

    try:
        df_strace = pd.read_csv(filein_strace)
    except IOError as e:
        df_strace = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_strace)

    try:
        df_net = pd.read_csv(filein_net)
        net_profile(logdir, cfg, df_net)
    except IOError as e:
        df_net = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_net)

    try:
        df_vmstat = pd.read_csv(filein_vmstat)
        vmstat_profile(logdir, cfg, df_vmstat)
    except IOError as e:
        df_vmstat = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_vmstat)

    try:
        df_mpstat = pd.read_csv(filein_mpstat)
        features = mpstat_profile(logdir, cfg, df_mpstat, features)
    except IOError as e:
        df_mpstat = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_mpstat)

    try:
        df_gpu = pd.read_csv(filein_gpu)
        features = gpu_profile(logdir, cfg, df_gpu, features)
    except IOError:
        df_gpu = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found. If there is no need to profile GPU, just ignore it." % filein_gpu)

    if cfg.enable_aisi:
        selected_pattern, iter_summary = sofa_aisi(logdir, cfg, df_cpu, df_gpu, df_strace, df_mpstat)
                    
    print_title('Final Performance Features')
    print('%s%s%s' % ('ID'.ljust(10),'Feature'.ljust(30),'Value'.ljust(20)) )
    for i in range(len(features)):
        name = features.iloc[i]['name']
        value = features.iloc[i]['value']
        print('%s%s%s' % (str(i).ljust(10), name.ljust(30), ('%.3lf'%value).ljust(20)))

    if cfg.potato_server:
        potato_client(logdir, cfg, df_cpu, df_gpu, df_vmstat, iter_summary)
        print_title('POTATO Feedback')
        r = {'image':'im001', 'score':1.2, 'action':'None'}
        print('Tag of optimal image recommended from POTATO: '+ highlight(r['image']))
        print('Estimated speedup: %.2lfx' % r['score'] )
        print('Optimization action: '+r['action'])
        print('Please re-launch KubeFlow Jupyter-notebook with the new tag.')
    #print_warning('Something wrong with POTATO client')

    print('\n\n')
