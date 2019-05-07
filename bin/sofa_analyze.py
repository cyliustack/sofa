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
from matplotlib import pyplot as plt
import grpc
import potato_pb2
import potato_pb2_grpc
import socket

# input: pfv(performance feature vector), Pandas.DataFrame
# output: hint, docker_image  
def get_hint(potato_server, features):

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
        channel = grpc.insecure_channel(potato_server)
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
        t = t + 0.1
        if cfg.roi_end > 0 and (t < cfg.roi_begin or t > cfg.roi_end):
            continue
        
        window_begin = t - 0.1 
        window_end = t
        
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
            if elapsed_time['gpu'] > 0.1 :
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
    if not cfg.cluster_ip:
        print_title("SM & MEM Profiling")
    
    if len(df_nvsmi) > 0 :
        sm_start = df_nvsmi.iloc[0].timestamp 
        sm_end = df_nvsmi.iloc[-1].timestamp
        SM_time = sm_end - sm_start
        result = df_nvsmi.groupby(['deviceId','event'])['duration'].mean() 
        result = result.astype(int)
        
        gpu_sm_util = df_nvsmi.groupby(['event'])['duration'].mean()[0]
        gpu_mem_util = df_nvsmi.groupby(['event'])['duration'].mean()[1]
        
        if not cfg.cluster_ip:
            print(result)
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
    if not cfg.cluster_ip:
        print_title("Network Profiling:")
    grouped_df = df.groupby("name")["duration"]
    net_time = 0
    n_packets = 0
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("network:tcp:") != -1:
            net_time = net_time + grouped_df.get_group(key).sum()
            n_packets = n_packets + 1
    #print(("total network time (s) = %.3lf" % net_time))
    #print(("total amount of network packets  = %d" % n_packets))
    # total network packet
    packet_num_matrix = df.groupby(['pkt_src','pkt_dst','payload']).size().unstack(level=1, fill_value=0)
    
    # total network traffic
    packet_sum_matrix = df.groupby(['pkt_src','pkt_dst'])["payload"].sum().unstack(level=1, fill_value=0)
    
    # ================ change pandas table columns and index name ====
    rename_index = packet_sum_matrix.index.tolist()
    rename_index2 = packet_num_matrix.index.tolist()
    rename_columns = packet_sum_matrix.columns.tolist()
    rename_columns2 = packet_num_matrix.columns.tolist() 

    def zero(s):
        if s[0:2] == '00':
            s = s[2]
        elif (s[0] == '0') and (s[1] != '0'): 
            s = s[1:3]
        return(s)

    def check_str(rename_list):
        rename_list_new = []
        for j in rename_list:
            j = str(int(j))
            a = j[-9:-6]
            b = j[-6:-3]
            c = j[-3:]  
            j = j[:-9] + '.' + zero(a) + '.' + zero(b) + '.' + zero(c)
            rename_list_new.append(j)
        return(rename_list_new)
 
    def check_str2(rename_list):
        rename_columns_2 = []
        for i in rename_list:
            i = str(int(i[0]))
            a = i[-9:-6] 
            b = i[-6:-3]
            c = i[-3:]
            i = i[:-9] + '.' + zero(a) + '.' + zero(b) + '.' + zero(c)
            rename_columns_2.append(i)
        return(rename_columns_2)
    
    def convertbytes(B):
        B = float(B)
        KB = float(1024)
        MB = float(KB ** 2) # 1,048,576
        GB = float(KB ** 3) # 1,073,741,824
        TB = float(KB ** 4) # 1,099,511,627,776

        if B < KB:
            return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
        elif KB <= B < MB:
            return '{0:.2f} KB'.format(B/KB)
        elif MB <= B < GB:
            return '{0:.2f} MB'.format(B/MB)
        elif GB <= B < TB:
            return '{0:.2f} GB'.format(B/GB)
        elif TB <= B:
            return '{0:.2f} TB'.format(B/TB)

    rename_index_new = check_str(rename_index)
    rename_index_new = dict(zip(rename_index, rename_index_new))
    
    rename_index2_new = check_str2(rename_index2)
    rename_index2_final = list(set(rename_index2_new))
    rename_index2_final.sort(key=rename_index2_new.index)
    
    rename_columns_new = check_str(rename_columns)
    rename_columns_new = dict(zip(rename_columns, rename_columns_new))
    
    rename_columns2_new = check_str(rename_columns2)
    rename_columns2_new = dict(zip(rename_columns2, rename_columns2_new))          
 
    # rename here
    packet_sum_matrix = packet_sum_matrix.rename(columns=rename_columns_new)
    packet_num_matrix = packet_num_matrix.rename(columns=rename_columns2_new)
    packet_sum_matrix = packet_sum_matrix.rename(index=rename_index_new)
    packet_num_matrix.index.set_levels(rename_index2_final , level = 0, inplace = True)
    
    print("total amount of network traffic : ", convertbytes(df['payload'].sum()), '\n', packet_sum_matrix.to_string(), "\n")
    if cfg.verbose:
        print("total amount of network packets = %d\n" % packet_num_matrix.sum().sum() ,packet_num_matrix.to_string(), "\n")
    
    network_value = []
    src = []
    dst = []
    final = []

    for index in packet_sum_matrix.index:
        for column in packet_sum_matrix.columns:
            src.append(index)
            dst.append(column)
            network_value.append(packet_sum_matrix[column][index])
    record = list(zip(src, dst, network_value))
    record.sort(key=lambda tup:tup[2], reverse=True)
    
    for src, dst, value in record:
        if value == 0:
            pass
        else:
            item = [src, dst, convertbytes(value), round(value / df['payload'].sum(), 2)]
            final.append(item)
    summary = pd.DataFrame(final, columns=['Source', 'Destination', 'Amount', 'Percentage of a Node'])
    summary.to_csv(logdir + 'netrank.csv',
                mode='w',
                header=True,
                index=False)

    df = pd.DataFrame({'name':['net_time'],                                                 
                       'value':[net_time] },  
                       columns=['name','value'])
    features = pd.concat([features, df])
    return features

def convertbytes(B):
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB/s'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB/s'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB/s'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB/s'.format(B/TB)

def netbandwidth_profile(logdir, cfg, df, features):
    if not cfg.cluster_ip:
        print_title("Network Bandwidth Profiling:")
        print('Bandwidth Quartile :')
    tx = df['event'] == float(0)
    rx = df['event'] == float(1)
  
    if not cfg.cluster_ip:
        bw_tx_q1 = df[tx]['bandwidth'].quantile(0.25)
        bw_tx_q2 = df[tx]['bandwidth'].quantile(0.5)
        bw_tx_q3 = df[tx]['bandwidth'].quantile(0.75)
        bw_tx_mean = int(df[tx]['bandwidth'].mean())
        bw_rx_q1 = df[rx]['bandwidth'].quantile(0.25)
        bw_rx_q2 = df[rx]['bandwidth'].quantile(0.5)
        bw_rx_q3 = df[rx]['bandwidth'].quantile(0.75)
        bw_rx_mean = int(df[rx]['bandwidth'].mean())

        print('Q1 tx : %s, rx : %s' % ( convertbytes(bw_tx_q1), convertbytes(bw_rx_q1)))
        print('Q2 tx : %s, rx : %s' % ( convertbytes(bw_tx_q2), convertbytes(bw_rx_q2)))
        print('Q3 tx : %s, rx : %s' % ( convertbytes(bw_tx_q3), convertbytes(bw_rx_q3)))
        print('Avg tx : %s, rx : %s'% ( convertbytes(bw_tx_mean), convertbytes(bw_rx_mean)))
                                 
    #network chart part
    all_time = df[tx]['timestamp'].tolist()
    all_tx = df[tx]['bandwidth'].tolist()
    all_rx = df[rx]['bandwidth'].tolist()
    fig = plt.figure(dpi=128, figsize=(16, 14))
    plt.plot(all_time, all_tx, c='red', alpha=0.5, label='tx')
    plt.plot(all_time, all_rx, c='blue', alpha=0.5, label='rx')
    plt.legend(loc='upper right')
    plt.title("Network Report", fontsize=18)
    plt.xlabel('Timestamp (s)', fontsize=16)
    plt.ylabel("Bandwidth (bytes)", fontsize=16)
    fig.savefig("%s/network_report.pdf" % logdir, bbox_inches='tight')
    if not cfg.cluster_ip:
        print('Network Bandwidth Chart is saved at %s/network_report.pdf' %logdir)

        df_feature = pd.DataFrame({ 'name':['bw_tx_q2', 'bw_tx_q3', 'bw_rx_q2', 'bw_rx_q3'], 
                        'value':[bw_tx_q2, bw_tx_q3, bw_rx_q2, bw_rx_q3] }, 
                        columns=['name','value'])
        features = pd.concat([features, df_feature])   
 
    return features

def blktrace_latency_profile(logdir, cfg, df, features):
    print_title("Storage Profiling:")
    print('Blktracae Latency Quartile (s):')
    blktrace_latency = df['event'] == 'C'
    blktrace_latency_q1 = df[blktrace_latency]['duration'].quantile(0.25)
    blktrace_latency_q2 = df[blktrace_latency]['duration'].quantile(0.5)
    blktrace_latency_q3 = df[blktrace_latency]['duration'].quantile(0.75)
    blktrace_latency_mean = df[blktrace_latency]['duration'].mean()

    print('Q1 blktrace latency : %f' % blktrace_latency_q1)
    print('Q2 blktrace latency : %f' % blktrace_latency_q2)
    print('Q3 blktrace latency : %f' % blktrace_latency_q3)
    print('Avg blktrace latency : %f'% blktrace_latency_mean)
    df_feature = pd.DataFrame({ 'name':['blktrace_latency_q1','blktrace_latency_q2','blktrace_latency_q3'], 
                        'value': [blktrace_latency_q1, blktrace_latency_q2, blktrace_latency_q3] }, 
                        columns=['name','value'])
    features = pd.concat([features, df_feature])   
 
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
    print('average bi/s: %d' % int(vm_cs))
    print('average bo/s: %d' % int(vm_in))
    print('average cs/s: %d' % int(vm_bi))
    print('average in/s: %d' % int(vm_bo))

    df_feature = pd.DataFrame({ 'name':['vm_bi', 'vm_bo', 'vm_cs', 'vm_in' ], 
                        'value':[vm_bi, vm_bo, vm_cs, vm_in] }, 
                        columns=['name','value'])
    features = pd.concat([features, df_feature])   

    return features

def mpstat_profile(logdir, cfg, df, features):
    if not cfg.cluster_ip:
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
    if not cfg.cluster_ip:
        print('CPU Utilization (%):')
        print('core\tUSR\tSYS\tIDL\tIOW\tIRQ')
    for i in range(len(df_summary)):
        t_sum = df_summary.iloc[i].sum() 
        if not cfg.cluster_ip:
            print('%3d\t%3d\t%3d\t%3d\t%3d\t%3d'%(i,int(100.0*df_summary.iloc[i]['USR']/t_sum),
                                                    int(100.0*df_summary.iloc[i]['SYS']/t_sum),
                                                    int(100.0*df_summary.iloc[i]['IDL']/t_sum),
                                                    int(100.0*df_summary.iloc[i]['IOW']/t_sum),
                                                    int(100.0*df_summary.iloc[i]['IRQ']/t_sum) ))
    if not cfg.cluster_ip:
        print('CPU Time (s):')
        print('core\tUSR\tSYS\tIDL\tIOW\tIRQ')
    for i in range(len(df_summary)):
        t_sum = df_summary.iloc[i].sum()
        if not cfg.cluster_ip:
            print('%3d\t%.2lf\t%.2lf\t%.2lf\t%.2lf\t%.2lf'%(i,
                                                    df_summary.iloc[i]['USR'],
                                                    df_summary.iloc[i]['SYS'],
                                                    df_summary.iloc[i]['IDL'],
                                                    df_summary.iloc[i]['IOW'],
                                                    df_summary.iloc[i]['IRQ'] ))

    total_cpu_time = df_summary[['USR','SYS','IRQ']].sum().sum()
    cpu_util = int(100*total_cpu_time / (num_cores*cfg.elapsed_time))
    if not cfg.cluster_ip:
        print('Active CPU Time (s): %.3lf' % total_cpu_time)
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
    df_bandwidth = pd.DataFrame([], columns=cfg.columns)
    df_blktrace = pd.DataFrame([], columns=cfg.columns)
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
    filein_bandwidth = logdir + "netstat.csv"
    filein_blktrace = logdir + "blktrace.csv"

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
        df_bandwidth = pd.read_csv(filein_bandwidth)
        features = netbandwidth_profile(logdir, cfg, df_bandwidth, features)
    except IOError as e:
        df_bandwidth = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_bandwidth)

    try:
        df_blktrace = pd.read_csv(filein_blktrace)
        features = blktrace_latency_profile(logdir, cfg, df_blktrace, features)
    except IOError as e:
        df_blktrace = pd.DataFrame([], columns=cfg.columns)
        print_warning("%s is not found" % filein_blktrace)

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
        hint, docker_image = get_hint(cfg.potato_server, features)
        print('Optimization hints: ' + hint)
        print('Tag of optimal image recommended from POTATO: ' + highlight(docker_image))
        print('Please re-launch KubeFlow Jupyter-notebook with the new tag.')
    
    print('\n\n')

def cluster_analyze(cfg):
    print_title("Cluster Network Profiling :")
    cluster = cfg.cluster_ip.split(',')
    summary_net = pd.DataFrame([], columns=['Source', 'Destination', 'Amount', 'Percentage of a Node'])
    summary_compute = pd.DataFrame([], columns=['gpu_sm_util','gpu_mem_util','cpu_util'])
    summary_band = pd.DataFrame([], columns=['Q1', 'Q2', 'Q3', 'Avg'])  
    all = []
    for i, ip in enumerate(cluster):
        features = pd.DataFrame({'name':['elapsed_time'],
                                 'value':[cfg.elapsed_time]},
                                 columns=['name','value'])
        
        node = 'node ' + str(i)
        print('node ' + str(i) + ' is ' + ip)
        logdir = './sofalog-' + ip +'/'
        filein_net = logdir + "nettrace.csv"
        filein_mpstat = logdir + "mpstat.csv"
        filein_nvsmi = logdir + "nvsmi_trace.csv"
        filein_bandwidth = logdir + "netstat.csv"
        with open(logdir+'/misc.txt') as f:
            lines = f.readlines()
            elapsed_time = float(lines[0].split()[1])
            vcores = int(lines[2].split()[1])
            cfg.elapsed_time = float(lines[0].split()[1])
        try:
            df_net = pd.read_csv(filein_net)
            features = net_profile(logdir, cfg, df_net, features)
        except IOError as e:
            df_net = pd.DataFrame([], columns=cfg.columns)
            print_warning("%s is not found" % filein_net)
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
            df_bandwidth = pd.read_csv(filein_bandwidth)
            features = netbandwidth_profile(logdir, cfg, df_bandwidth, features)
        except IOError as e:
            df_bandwidth = pd.DataFrame([], columns=cfg.columns)
            print_warning("%s is not found" % filein_bandwidth)

        sm = int(features[features['name'] == 'gpu_sm_util']['value'])
        mem = int(features[features['name'] == 'gpu_mem_util']['value'])
        cpu = int(features[features['name'] == 'cpu_util']['value'])
        sm_mem_cpu = [sm, mem, cpu]
   
        compute_tmp = pd.DataFrame([sm_mem_cpu], columns = ['gpu_sm_util', 'gpu_mem_util', 'cpu_util'])
        summary_compute = pd.concat([summary_compute, pd.concat([compute_tmp], keys=[node])]) 
        net_tmp = pd.read_csv(logdir + "netrank.csv")
        summary_net = pd.concat([summary_net, pd.concat([net_tmp], keys=[node])])
       
        # for bandwidth report
        tx = df_bandwidth['event'] == float(0)
        rx = df_bandwidth['event'] == float(1)
        tx_tmp = [convertbytes(df_bandwidth[tx]['bandwidth'].quantile(0.25)),                         
                    convertbytes(df_bandwidth[tx]['bandwidth'].quantile(0.5)),
                    convertbytes(df_bandwidth[tx]['bandwidth'].quantile(0.75)),
                    convertbytes(df_bandwidth[tx]['bandwidth'].mean())]
        rx_tmp = [convertbytes(df_bandwidth[rx]['bandwidth'].quantile(0.25)),                         
                    convertbytes(df_bandwidth[rx]['bandwidth'].quantile(0.5)),
                    convertbytes(df_bandwidth[rx]['bandwidth'].quantile(0.75)),
                    convertbytes(df_bandwidth[rx]['bandwidth'].mean())]
        band_tmp = pd.DataFrame([tx_tmp], columns = ['Q1', 'Q2', 'Q3', 'Avg'], index = ['tx'])
        rx_pd = pd.DataFrame([rx_tmp], columns = ['Q1', 'Q2', 'Q3', 'Avg'], index = ['rx'])
        band_tmp = pd.concat([band_tmp, rx_pd]) 
        summary_band = pd.concat([summary_band, pd.concat([band_tmp], keys=[node])]) 
    print('Ranked Network Traffic : \n', summary_net, '\n')
    print('Cluster Bandwidth Quartile: \n', summary_band)
    print_title('Cluster Computation Profiling:')
    print(summary_compute)
