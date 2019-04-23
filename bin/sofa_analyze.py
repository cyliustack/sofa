import argparse
import matplotlib
matplotlib.use('agg')
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
import grpc
import potato_pb2
import potato_pb2_grpc
import socket

# input: pfv(performance feature vector), Pandas.DataFrame
# output: hint, docker_image  
def get_hint(features):

    if len(features) > 0:
        pfv = potato_pb2.PerformanceFeatureVector() 
        for i in range(len(features)):
            name = features.iloc[i]['name']
            value = features.iloc[i]['value']
            print('%s%s%s' % (str(i).ljust(10), name.ljust(30), ('%.3lf'%value).ljust(20)))
            pfv.name.append(name)
            pfv.value.append(value)
			
        print('Wait for response from POTATO server...')
        myhostname = socket.gethostname()
        channel = grpc.insecure_channel('localhost:50051')
        stub = potato_pb2_grpc.HintStub(channel)
        request = potato_pb2.HintRequest( hostname = myhostname,
                                          pfv = pfv) 
        response = stub.Hint(request)
        hint = response.hint 
        docker_image = response.docker_image
    else:
        hint = 'There is no pfv to get hints.' 
        docker_image = 'NA' 

    print(hint)
    print(docker_image) 
    return hint, docker_image 

def dynamic_top_down(logdir, cfg, df_mpstat, df_cpu, df_gpu, features):
    print_title("Dynamic Top-Down Analysis")

    total_elapsed_time = {'usr':0, 'sys':0, 'gpu':0, 'iow':0} 
    elapsed_time_ratio = {'usr':0, 'sys':0, 'gpu':0, 'iow':0} 
   
    if len(df_mpstat) == 0 or len(df_cpu) == 0:
        print_warning('no mpstat and perf traces!')
        return features

    t_begin = df_mpstat.iloc[0]['timestamp']
    t_end = df_mpstat.iloc[-1]['timestamp']
    
    t = t_begin
    while t < t_end:
        window_begin = t - 0.1 
        window_end = t
        t = t + 0.1
        
        if df_cpu.iloc[0].timestamp > window_end:
            continue
        cond1 = (df_cpu['timestamp'] > window_begin)
        cond2 = (df_cpu['timestamp'] <= window_end)
        df_cpu_interval = df_cpu[ cond1 & cond2 ]
        
        cond1 = (df_gpu['timestamp'] > window_begin)
        cond2 = (df_gpu['timestamp'] <= window_end)
        df_gpu_interval = df_gpu[ cond1 & cond2 ]
        
        cond1 = (df_mpstat['timestamp'] > window_begin)
        cond2 = (df_mpstat['timestamp'] <= window_end)
        df_mpstat_interval = df_mpstat[ cond1 & cond2 ]
        mp_usr = []
        mp_sys = []
        mp_iow = []
        for i in range(len(df_mpstat_interval)):
            ratios = df_mpstat_interval.iloc[i]['name'].split(':')[1].split('|') 
            #print(ratios)
            mp_usr.append(0.1*int(ratios[1])/100.0)
            mp_sys.append(0.1*int(ratios[2])/100.0)
            mp_iow.append(0.1*int(ratios[4])/100.0)
        mp_usr = np.asarray(mp_usr)
        mp_sys = np.asarray(mp_sys)
        mp_iow = np.asarray(mp_iow)

        elapsed_time = {'usr':0, 'sys':0, 'gpu':0, 'iow':0} 

        if len(df_mpstat_interval) > 0:
            elapsed_time['usr'] = mp_usr.max()
            elapsed_time['sys'] = mp_sys.max()
            elapsed_time['gpu'] = df_gpu_interval['duration'].sum()
            elapsed_time['iow'] = mp_iow.max()
            dominator = max(elapsed_time, key=elapsed_time.get)
            if elapsed_time['gpu'] > 0 :
                dominator = 'gpu'
            total_elapsed_time[dominator] = total_elapsed_time[dominator] + 0.1

    total_all_elapsed_time = sum(total_elapsed_time.values())
    if total_all_elapsed_time > 0 :
        elapsed_time_ratio['usr'] = 100 * total_elapsed_time['usr'] / total_all_elapsed_time 
        elapsed_time_ratio['sys'] = 100 * total_elapsed_time['sys'] / total_all_elapsed_time 
        elapsed_time_ratio['gpu'] = 100 * total_elapsed_time['gpu'] / total_all_elapsed_time 
        elapsed_time_ratio['iow'] = 100 * total_elapsed_time['iow'] / total_all_elapsed_time 
        print('Elapsed Time = %.1lf ' % total_all_elapsed_time)
        print('USR = %.1lf %%' % elapsed_time_ratio['usr'])
        print('SYS = %.1lf %%' % elapsed_time_ratio['sys'])
        print('GPU = %.1lf %%' % elapsed_time_ratio['gpu'])
        print('IOW = %.1lf %%' % elapsed_time_ratio['iow'])
        df = pd.DataFrame({'name':['elapsed_usr_time_ratio', 'elapsed_sys_time_ratio', 'elapsed_gpu_time_ratio', 'elapsed_iow_time_ratio'], 
                        'value':[elapsed_time_ratio['usr'],elapsed_time_ratio['sys'],elapsed_time_ratio['gpu'],elapsed_time_ratio['iow'] ] }, 
                        columns=['name','value'])
        features = pd.concat([features, df])

    return features

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

def nvsmi_profile(logdir, cfg, df_nvsmi, features):
    print_title("SM & MEM Profiling")
    
    if len(df_nvsmi) > 0 :
        sm_start = df_nvsmi.iloc[0].timestamp 
        sm_end = df_nvsmi.iloc[-1].timestamp
        SM_time = sm_end - sm_start
        result = df_nvsmi.groupby(['deviceId','event'])['duration'].mean() 
        result = result.astype(int)
        print(result)
        
        gpu_sm_util = df_nvsmi.groupby(['event'])['duration'].mean()[0]
        gpu_mem_util = df_nvsmi.groupby(['event'])['duration'].mean()[1]
        print('Average SM Utilization (%): ', int(gpu_sm_util))
        print('Average MEM Utilization (%): ', int(gpu_mem_util))
        print('Active GPU Time (s): %.3lf' % (SM_time * gpu_sm_util/100.0))
        df = pd.DataFrame({'name':['gpu_sm_util', 'gpu_mem_util'], 
                        'value':[gpu_sm_util, gpu_mem_util] }, 
                        columns=['name','value'])
        features = pd.concat([features, df])

    return features

def gpu_profile(logdir, cfg, df_gpu, features):
    print_title("GPU Profiling")
    print('Per-GPU time (s):')
    groups = df_gpu.groupby("deviceId")["duration"]
    gpu_time = 0
    for key, item in groups:
        gpuid = int(float(key))
        per_gpu_time = groups.get_group(key).sum()
        print("[%d]: %lf" % (gpuid, per_gpu_time))
        gpu_time = gpu_time + per_gpu_time 
    n_gpus = len(groups)
    print(("Total GPU time of all GPUs (s) = %.3lf" % gpu_time))
   
    kernel_time = 0
    grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        if key == 0:
            kernel_time = grouped_df.get_group(key).sum()

    nccl_time = 0
    grouped_df = df_gpu.groupby("name")["duration"]
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("nccl") != -1:
            nccl_time = nccl_time + grouped_df.get_group(key).sum()

    features = comm_profile(logdir, cfg, df_gpu, features)

    get_top_k_events(df_gpu, 10)
    df = pd.DataFrame({'name':['gpu_time', 'num_gpus', 'kernel_time', 'nccl_time'], 
                        'value':[gpu_time, n_gpus, kernel_time, nccl_time] }, 
                        columns=['name','value'])
    features = pd.concat([features, df])
    return features

def net_profile(logdir, cfg, df, features):
    print_title("Network Profiling:")
    grouped_df = df.groupby("name")["duration"]
    net_time = 0
    n_packets = 0
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("network:tcp:") != -1:
            net_time = net_time + grouped_df.get_group(key).sum()
            n_packets = n_packets + 1
    print(("total network time (s) = %.3lf" % net_time))
    print(("total amount of network packets  = %d" % n_packets))
    df = pd.DataFrame({'name':['net_time'], 
                        'value':[net_time] }, 
                        columns=['name','value'])
    features = pd.concat([features, df])
    return features

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

def vmstat_profile(logdir, cfg, df, features):
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

    vm_bi = vmstat_traces['bi'].mean()
    vm_bo = vmstat_traces['bo'].mean()
    vm_cs = vmstat_traces['cs'].mean()
    vm_in = vmstat_traces['in'].mean()
    print('sum of vmstat bi: ', vm_cs)
    print('sum of vmstat bo: ', vm_in)
    print('sum of vmstat cs: ', vm_bi)
    print('sum of vmstat in: ', vm_bo)

    df_feature = pd.DataFrame({ 'name':['vm_bi', 'vm_bo', 'vm_cs', 'vm_in' ], 
                        'value':[vm_bi, vm_bo, vm_cs, vm_in] }, 
                        columns=['name','value'])
    features = pd.concat([features, df_feature])   

    return features

def mpstat_profile(logdir, cfg, df, features):
    print_title("MPSTAT Profiling:")
    num_cores = int(df['deviceId'].max() + 1)
    df_summary = pd.DataFrame( np.zeros((num_cores,5)), columns=['USR','SYS','IDL','IOW','IRQ'])
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

    total_cpu_time = df_summary[['USR','SYS','IRQ']].sum().sum()
    print('Active CPU Time (s): %.3lf' % total_cpu_time) 
    cpu_util = int(100*total_cpu_time / (num_cores*cfg.elapsed_time))
    print('Active CPU ratio (%%): %3d' % cpu_util)
    df_feature = pd.DataFrame({ 'name':['num_cores', 'cpu_util'], 
                        'value':[num_cores, cpu_util] }, 
                        columns=['name','value'])
    features = pd.concat([features, df_feature])   
    return features


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
    filein_nvsmi = logdir + "nvsmi_trace.csv"

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
        features = net_profile(logdir, cfg, df_net, features)
    except IOError as e:
        df_net = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_net)

    try:
        df_vmstat = pd.read_csv(filein_vmstat)
        features = vmstat_profile(logdir, cfg, df_vmstat, features)
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
        df_nvsmi = pd.read_csv(filein_nvsmi) 
        features = nvsmi_profile(logdir, cfg, df_nvsmi, features)
    except IOError:
        print_warning("nvsmi_trace.csv is not found")

    try:
        df_gpu = pd.read_csv(filein_gpu)
        features = gpu_profile(logdir, cfg, df_gpu, features)
    except IOError:
        df_gpu = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found. If there is no need to profile GPU, just ignore it." % filein_gpu)

    try:
        features = dynamic_top_down(logdir, cfg, df_mpstat, df_cpu, df_gpu, features)
    except IOError as e:
        print_warning("Some files are not found, which are needed for dynamic_top_down analysis")



    if cfg.enable_aisi:
        selected_pattern, iter_summary = sofa_aisi(logdir, cfg, df_cpu, df_gpu, df_strace, df_mpstat)
                    
    print_title('Final Performance Features')
    print('%s%s%s' % ('ID'.ljust(10),'Feature'.ljust(30),'Value'.ljust(20)) )
    
    for i in range(len(features)):
        name = features.iloc[i]['name']
        value = features.iloc[i]['value']
        print('%s%s%s' % (str(i).ljust(10), name.ljust(30), ('%.3lf'%value).ljust(20)))

    if cfg.potato_server:
        print_title('POTATO Feedback')
        hint, docker_image = get_hint(features)
        print('Optimization hints: ' + hint)
        print('Tag of optimal image recommended from POTATO: ' + highlight(docker_image))
        print('Please re-launch KubeFlow Jupyter-notebook with the new tag.')
    
    print('\n\n')
