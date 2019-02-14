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
from matplotlib import pyplot as plt   
import networkx as nx
import numpy as np
import pandas as pd
import requests

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

def potato_submit(cfg, data_in):
    headers = {'Content-type': 'application/json'}
    url = cfg.potato_server+'/metric/' + data_in['name']
    data_json = json.dumps(data_in)
    response = requests.delete(url, data=data_json, headers=headers)
    response = requests.post(url, data=data_json, headers=headers)
    response = requests.get(url, data=data_json, headers=headers)
    print('SUBMIT %s: %.3lf ' % ( data_in['name'], float(response.json()['value'])))

def potato_client(logdir, cfg, df_cpu, df_gpu, df_vmstat, iter_summary):
    print_title("POTATO Client")
    grouped_df = df_cpu.groupby("deviceId")["duration"]
    cpu_time = 0.0
    for key, item in grouped_df:
        if cfg.verbose:
            print(("[%d]: %lf" % (key, grouped_df.get_group(key).sum())))
        cpu_time = cpu_time + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    avg_cpu_time = cpu_time / n_devices
    data = {'name' : 'avg_cpu_time', 'unit':'s', 'value': avg_cpu_time }
    potato_submit(cfg, data)

    #return {'elapsed_time': elapsed_time, 'fw_time': fw_time, 'bw_time': bw_time, 'kernel_time': kernel_time, 'payload': payload, 'copy_time':copy_time, 'gpu_time':gpu_time, 'gemm_time':gemm_time, 'streams':streams}
    if len(iter_summary) > 0:
       print('mean: ', iter_summary['elapsed_time'].mean())
       step_time = scipy.stats.gmean(iter_summary['elapsed_time'])
       if cfg.num_iterations > 1:
           step_time = scipy.stats.gmean(iter_summary['elapsed_time'].iloc[1:])
       fw_time = iter_summary['fw_time'].mean()
       bw_time = iter_summary['bw_time'].mean()
       copy_time = iter_summary['copy_time'].mean()
       gpu_time = iter_summary['gpu_time'].mean()
       gemm_time = iter_summary['gemm_time'].mean()
       kernel_time = gpu_time - copy_time
       payload = iter_summary['payload'].mean()
       streams = iter_summary['streams'].mean()

       print_title('Upload performance data to POTATO server')
       data = {'name' : 'step_time', 'unit':'s', 'value': step_time }
       potato_submit(cfg, data)
       data = {'name' : 'copy_time', 'unit':'s', 'value': copy_time }
       potato_submit(cfg, data)
       #TODO: remove it
       data = {'name' : 'payload_h2d', 'unit':'B', 'value': payload/2 }
       data = {'name' : 'payload_d2h', 'unit':'B', 'value': payload/2 }
       potato_submit(cfg, data)

       #data = {'name' : 'kernel_time', 'unit':'s', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'h2d_time', 'unit':'s', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'd2h_time', 'unit':'s', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'p2p_time', 'unit':'s', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'h2d_payload', 'unit':'B', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'd2h_payload', 'unit':'B', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'p2p_payload', 'unit':'B', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'h2d_bw', 'unit':'GBps', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'd2h_bw', 'unit':'GBps', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'd2d_bw', 'unit':'GBps', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data = {'name' : 'p2p_bw', 'unit':'GBps', 'value': total_exec_time }
       #potato_submit(cfg, data)
       #data_json = json.dumps(data)
    #print('cpu_time: %.3lf (%s)' % (r.json()['value'], r.json()['unit']))

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
    fig = plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    df_summary.plot.bar(stacked=True);
    #plt.ylabel('USR')
    plt.subplot(2, 1, 2)
    df_summary.plot.area()
    #plt.ylabel('SYS')
    #df.query('category == 1')['duration'].hist(bins=10)
    #plt.tight_layout()
    #plt.xlabel('Percentage of CPU Utilization')
    #plt.ylabel('IOW')
    #df.query('category == 3')['duration'].hist(bins=10)
    plt.tight_layout()
    fig.savefig(logdir + 'mpstat.png')
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
    df_gpu = []
    df_cpu = []
    df_vmstat = []
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
        df_vmstat = pd.read_csv(filein_vmstat)
        df_mpstat = pd.read_csv(filein_mpstat)
        df_net = pd.read_csv(filein_net)
        df_strace = pd.read_csv(filein_strace)
        cpu_profile(logdir, cfg, df_cpu)
        net_profile(logdir, cfg, df_net)
        vmstat_profile(logdir, cfg, df_vmstat)
        features = mpstat_profile(logdir, cfg, df_mpstat, features)
    except IOError:
        print_warning("cputrace.csv is not found")
        #quit()

    try:
        df_gpu = pd.read_csv(filein_gpu)
        #df_gpu.loc[:, 'timestamp'] -= df_gpu.loc[0, 'timestamp']
        features = gpu_profile(logdir, cfg, df_gpu, features)
        if cfg.enable_aisi:
            iter_summary = sofa_aisi(logdir, cfg, df_cpu, df_gpu, df_strace, df_mpstat)
    except IOError:
        print_warning("gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

    print_title('Final Performance Features')
    print(features)

    if cfg.potato_server:
        potato_client(logdir, cfg, df_cpu, df_gpu, df_vmstat, iter_summary)
        print_title('POTATO Feedback')
        r = requests.get(cfg.potato_server+'/image/best')
        print('Tag of optimal image recommended from POTATO: '+ highlight(r.json()['tag']))
        print('Estimated speedup: %.2lfx' % r.json()['score'] )
        print('Optimization action: '+r.json()['description'])
        print('Please re-launch KubeFlow Jupyter-notebook with the new tag.')
    #print_warning('Something wrong with POTATO client')

    print('\n\n')
