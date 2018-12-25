import pandas as pd
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp
import glob
import os
from functools import partial
import subprocess
import re
from sofa_config import *
from sofa_print import *
from random import *
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import cxxfilt 
from operator import itemgetter
import warnings

def list_downsample(list_in, plot_ratio):
    new_list = []
    for i in range(len(list_in)):
        if i % plot_ratio == 0:
            # print("%d"%(i))
            new_list.append(list_in[i])
    return new_list


def list_to_csv_and_traces(logdir, _list, csvfile, _mode):
    traces = pd.DataFrame(_list[1:])    
    traces.columns = sofa_fieldnames
    _header = True if _mode == 'w' else False
    traces.to_csv(logdir +
                  csvfile,
                  mode=_mode,
                  header=_header,
                  index=False,
                  float_format='%.6f')
    return traces

# 0/0     [004] 96050.733788:          1 bus-cycles:  ffffffff8106315a native_write_msr_safe
# 0/0     [004] 96050.733788:          7     cycles:  ffffffff8106315a native_write_msr_safe
# 359342/359342 2493492.850125:          1 bus-cycles:  ffffffff8106450a native_write_msr_safe
# 359342/359342 2493492.850128:          1 cycles:  ffffffff8106450a
# native_write_msr_safe

def cpu_trace_read_hsg(sample, t_offset, cfg, cpu_mhz_xp, cpu_mhz_fp):
    fields = sample.split()
    event = event_raw = 0
    counts = 0

    if re.match('\[\d+\]', fields[1]) is not None:
        time = float(fields[2].split(':')[0])
        func_name = '[%s]'%fields[4].replace('-','_') + fields[6] + fields[7] 
        counts = float(fields[3])
        event_raw = 1.0 * int("0x01" + fields[5], 16)
        # add new column to cpu_traces        
        feature_types = fields[3].split(':')[0]
        mem_addr = fields[5]
    else:
        time = float(fields[1].split(':')[0])
        func_name = '[%s]'%fields[3].replace('-','_')  + fields[5] + fields[6] 
        counts = float(fields[2])
        event_raw = 1.0 * int("0x01" + fields[4], 16)        
        # add new column to cpu_traces        
        feature_types = fields[3].split(':')[0]        
        mem_addr = fields[4]

    t_begin = time + t_offset
    t_end = time + t_offset
 
    if len(cpu_mhz_xp) > 1:
        duration = counts/(np.interp(t_begin, cpu_mhz_xp, cpu_mhz_fp)*1e6)
    else:
        duration = counts/(3000.0*1e6)
    
    event  = np.log10(event_raw)

    if cfg.perf_events.find('cycles') == -1:
        duration = np.log2(event_raw/1e14)

    trace = [t_begin,                          # 0
             event,  # % 1000000               # 1
             duration,                         # 2
             -1,                               # 3
             -1,                               # 4 
             0,                                # 5
             0,                                # 6
             -1,                               # 7
             -1,                               # 8
             int(fields[0].split('/')[0]),     # 9
             int(fields[0].split('/')[1]),     # 10
             func_name,                        # 11
             0,                                # 12
             feature_types,                    # 13
             mem_addr]                         # 14
    return trace

def cpu_trace_read(sample, t_offset, cfg, cpu_mhz_xp, cpu_mhz_fp):
    fields = sample.split()
    event = event_raw = 0
    counts = 0
    if re.match('\[\d+\]', fields[1]) is not None:
        time = float(fields[2].split(':')[0])
        func_name = '[%s]'%fields[4].replace('-','_') + fields[6] + fields[7] 
        counts = float(fields[3])
        event_raw = 1.0 * int("0x01" + fields[5], 16)
    else:
        time = float(fields[1].split(':')[0])
        func_name = '[%s]'%fields[3].replace('-','_')  + fields[5] + fields[6] 
        counts = float(fields[2])
        event_raw = 1.0 * int("0x01" + fields[4], 16)

    t_begin = time + t_offset
    t_end = time + t_offset

    if len(cpu_mhz_xp) > 1:
        duration = counts/(np.interp(t_begin, cpu_mhz_xp, cpu_mhz_fp)*1e6)
    else:
        duration = counts/(3000.0*1e6)
    
    event  = np.log10(event_raw)

    if cfg.perf_events.find('cycles') == -1:
        duration = np.log2(event_raw/1e14)

    trace = [t_begin,                          # 0
             event,  # % 1000000               # 1
             duration,                         # 2
             -1,                               # 3
             -1,                               # 4 
             0,                                # 5
             0,                                # 6
             -1,                               # 7
             -1,                               # 8
             int(fields[0].split('/')[0]),     # 9
             int(fields[0].split('/')[1]),     # 10
             func_name,                        # 11
             0]                                # 12
    return trace


def net_trace_read(packet, t_offset):
    time = float(packet.split()[0])
    t_begin = time + t_offset
    t_end = time + t_offset
    if packet.split()[1] != 'IP':
        return []
    payload = int(packet.split()[6])
    duration = float(payload / 125.0e6)
    bandwidth = 125.0e6
    pkt_src = 0
    pkt_dst = 0
    for i in range(4):
        pkt_src = pkt_src + \
            int(packet.split()[2].split('.')[i]) * np.power(1000, 3 - i)
        pkt_dst = pkt_dst + \
            int(packet.split()[4].split('.')[i]) * np.power(1000, 3 - i)
    trace = [ t_begin,
              payload * 100 + 17,
              duration,
              -1,
              -1,
              payload,
              bandwidth,
              pkt_src,
              pkt_dst,
              -1,
              -1,
              "network:tcp:%d_to_%d_with_%d" % (pkt_src, pkt_dst, payload),
              0
              ]
    return trace

def cuda_api_trace_read(
        record,
        indices,
        n_cudaproc,
        ts_rescale,
        dt_rescale,
        payload_unit,
        t_offset):
    values = record.replace('"', '').split(',')
    api_name = '[CUDA_API]' + values[indices.index('Name')]

    # print("kernel name = %s" % kernel_name)
    time = float(values[indices.index('Start')]) / ts_rescale + t_offset
    duration = float(values[indices.index('Duration')]) / dt_rescale
    t_begin = time
    t_end = time + duration
    payload = 0
    bandwidth = 0
    pid = n_cudaproc
    deviceId = -1 
    tid = stream_id = -1
    pkt_src = pkt_dst = copyKind = 0

    # print("%d:%d [%s] ck:%d, %lf,%lf: %d -> %d: payload:%d, bandwidth:%lf,
    # duration:%lf "%(deviceId, streamId, kernel_name, copyKind,
    # t_begin,t_end, pkt_src, pkt_dst, payload, bandwidth, duration))
    trace = [t_begin,
             payload * 100 + 17,
             duration,
             deviceId,
             copyKind,
             payload,
             bandwidth,
             pkt_src,
             pkt_dst,
             pid,
             tid,
             api_name,
             0]
    return trace





def gpu_trace_read(
        record,
        indices,
        n_cudaproc,
        ts_rescale,
        dt_rescale,
        payload_unit,
        t_offset):
    values = record.replace('"', '').split(',')
    kernel_name = values[indices.index('Name')]

    # print("kernel name = %s" % kernel_name)
    time = float(values[indices.index('Start')]) / ts_rescale + t_offset
    duration = float(values[indices.index('Duration')]) / dt_rescale
    t_begin = time
    t_end = time + duration
    try:
        payload = int(float(values[indices.index('Size')]) * payload_unit)
    except BaseException:
        payload = 0

    try:
        bandwidth = float(values[indices.index('Throughput')])
    except BaseException:
        bandwidth = 0

    pid = n_cudaproc

    deviceId = -1 
    try:
        deviceId = int(float(values[indices.index('Context')]))
    except BaseException:
        deviceId = -1

    tid = stream_id = -1
    try:
        tid = streamId = int(float(values[indices.index('Stream')]))
    except BaseException:
        tid = streamId = -1

    pkt_src = pkt_dst = copyKind = 0
    if kernel_name.find('HtoD') != -1:
        copyKind = 1
        pkt_src = 0
        pkt_dst = deviceId
        kernel_name = "CUDA_COPY_H2D_%dB" % (payload)
    elif kernel_name.find('DtoH') != -1:
        copyKind = 2
        pkt_src = deviceId
        pkt_dst = 0
        kernel_name = "CUDA_COPY_D2H_%dB" % (payload)
    elif kernel_name.find('DtoD') != -1:
        copyKind = 8
        pkt_src = deviceId
        pkt_dst = deviceId
        kernel_name = "CUDA_COPY_D2D_%dB" % (payload)
    elif kernel_name.find('PtoP') != -1:
        copyKind = 10
        try:
            pkt_src = int(values[indices.index('Src Ctx')])
        except BaseException:
            pkt_src = 0

        try:
            pkt_dst = int(values[indices.index('Dst Ctx')])
        except BaseException:
            pkt_dst = 0

        kernel_name = "[CUDA_COPY_P2P]from_gpu%d_to_gpu%d_%dB" % (pkt_src, pkt_dst, payload)
    else:
        copyKind = 0

    if deviceId != -1:
        kernel_name = '[gpu%d]'%deviceId + kernel_name
    # print("%d:%d [%s] ck:%d, %lf,%lf: %d -> %d: payload:%d, bandwidth:%lf,
    # duration:%lf "%(deviceId, streamId, kernel_name, copyKind,
    # t_begin,t_end, pkt_src, pkt_dst, payload, bandwidth, duration))
    trace = [t_begin,
             payload * 100 + 17,
             duration,
             deviceId,
             copyKind,
             payload,
             bandwidth,
             pkt_src,
             pkt_dst,
             pid,
             tid,
             kernel_name,
             0]
    return trace


def gpu_kernel_trace_read(record, pid, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end = (record[1] - t_base) / 1e9 + t_glb_base
    kernel_name = "%s" % (
        gpu_symbol_table.loc[gpu_symbol_table._id_ == record[2], 'value'])
    kernel_name = kernel_name[:-30]
    trace = [	t_begin,
              record[2],
              float(t_end - t_begin),
              record[4],
              -1,
              0,
              0.0,
              -1,
              -1,
              pid,
              record[3],
              kernel_name,
              0]
    return trace


def gpu_memcpy_trace_read(record, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end = (record[1] - t_base) / 1e9 + t_glb_base

    src = -1
    dst = -1
    if record[3] == 1:
        src = 0
        dst = record[4] + 1
    elif record[3] == 2:
        src = record[4] + 1
        dst = 0
    elif record[3] == 8:
        src = record[4] + 1
        dst = record[4] + 1

    trace = [	t_begin,
              record[2],
              float(t_end - t_begin),
              record[4],
              record[3],
              record[2],
              float(record[2]) / (t_end - t_begin) / 1.0e6,
              src,
              dst,
              record[7],  # streamId
              -1,
              "gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]),
              0]
    return trace


def gpu_memcpy2_trace_read(record, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end = (record[1] - t_base) / 1e9 + t_glb_base
    trace = [	t_begin,
              record[2],
              float(t_end - t_begin),
              record[4],
              record[3],
              record[2],
              float(record[2]) / (t_end - t_begin) / 1.0e6,
              record[8],
              record[9],
              record[7],  # streamId
              -1,
              "gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]),
              0]
    return trace


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_warning(content):
    print((bcolors.WARNING + "[WARNING] " + content + bcolors.ENDC))


def print_info(content):
    print((bcolors.OKGREEN + "[INFO] " + content + bcolors.ENDC))


def print_progress(content):
    print((bcolors.OKBLUE + "[PROGRESS] " + content + bcolors.ENDC))


class SOFATrace:
    data = []
    name = []
    title = []
    color = []
    x_field = []
    y_field = []

# traces_to_json()
#    Please set "turboThreshold:0" in PlotOptions of ScatterChart to unlimit number of points
#    series: [{
#        name: 'Female',
#        color: 'rgba(223, 83, 83, .5)',
#        data: [{x:161.2, y:51.6, name:'b'}, {x:167.5, y:59.0, name:"a"}]
#    },
#    {
#        name: 'Male',
#        color: 'rgba(23, 83, 183, .5)',
#        data: [{x:16.2, y:151.6, name:'bd'}, {x:67.5, y:59.0, name:"ad"}]
#    }
#    ]


def traces_to_json(traces, path, cfg):
    if len(traces) == 0:
        print_warning("Empty traces!")
        return
    with open(path, 'w') as f:
        for trace in traces:
            if len(trace.data) > 0:
                f.write(trace.name + " = ")
                trace.data.rename(
                    columns={
                        trace.x_field: 'x',
                        trace.y_field: 'y'},
                    inplace=True)
                sofa_series = {
                    "name": trace.title,
                    "color": trace.color,
                    "data": json.loads(
                        trace.data.to_json(
                            orient='records'))}
                json.dump(sofa_series, f)
                trace.data.rename(
                    columns={
                        'x': trace.x_field,
                        'y': trace.y_field},
                    inplace=True)
            f.write("\n\n")

        f.write("sofa_traces = [ ")
        for trace in traces:
            if len(trace.data) > 0:
                f.write(trace.name + ",")
        f.write(" ]")

def random_generate_color():
        rand = lambda: randint(0, 255)
        return '#%02X%02X%02X' % (rand(), rand(), rand())

def kmeans_cluster(num_of_cluster, X):
    '''
    num_of_cluster: how many groups of data you prefer
    X: input taining data    
    '''
    random_state = 170
    try:
        num_of_cluster = 5
        y_pred = KMeans(n_clusters=num_of_cluster, random_state=random_state).fit_predict(X)                
    except :
        num_of_cluster = len(X) # minimum number of data
        y_pred = KMeans(n_clusters=num_of_cluster, random_state=random_state).fit_predict(X)  
    
    return y_pred

sofa_fieldnames = [
    "timestamp",  # 0
    "event",  # 1
    "duration",  # 2
    "deviceId",  # 3
    "copyKind",  # 4
    "payload",  # 5
    "bandwidth",  # 6
    "pkt_src",  # 7
    "pkt_dst",  # 8
    "pid",  # 9
    "tid",  # 10
    "name",  # 11
    "category"] # 12


def sofa_preprocess(logdir, cfg):
    t_glb_base = 0
    t_glb_gpu_base = 0

    with open(logdir + 'perf.script', 'w') as logfile:
        subprocess.call(['perf',
                         'script',
                         '--kallsym',
                         '%s/kallsyms' % logdir,
                         '-i',
                         '%s/perf.data' % logdir,
                         '-F',
                         'time,pid,tid,event,ip,sym,dso,symoff,period,brstack,brstacksym'],
                        stdout=logfile)
    
    with open(logdir + 'sofa_time.txt') as f:
        lines = f.readlines() 
        t_glb_base = float(lines[0]) + cfg.cpu_time_offset
        print_info('Time offset applied to timestamp (s):' + str(cfg.cpu_time_offset))
        print_info('SOFA global time base (s):' + str(t_glb_base))
    
    cpu_mhz_xp = [0.0]
    cpu_mhz_fp = [3000.0]
    #np.interp(2.5, xp, fp)
    try:
        with open(logdir + 'cpuinfo.txt') as f:
            lines = f.readlines()
            for line in lines:
                fields = line.split()
                timestamp = float(fields[0])
                mhz = float(fields[1])
                cpu_mhz_xp.append(timestamp)
                cpu_mhz_fp.append(mhz)
    except:
        print_warning('no cpuinfo file is found, default cpu MHz = %lf'%(fp[0]))
    print('cpu_mhz length: ',len(cpu_mhz_xp))
    
    net_traces = []
    cpu_traces = []
    cpu_traces_viz = []
    vm_usr_traces = []
    vm_sys_traces = []
    vm_bi_traces = []
    vm_b0_traces = []
    vm_in_traces = []
    vm_cs_traces = []
    vm_wa_traces = []
    vm_st_traces = []
    nvsmi_sm_traces = []
    nvsmi_mem_traces = []
    pcm_pcie_traces = []
    pcm_core_traces = []
    pcm_memory_traces = []
    gpu_traces = []
    gpu_traces_viz = []
    gpu_api_traces = []
    gpu_api_traces_viz = []
    gpu_kernel_traces = []
    gpu_memcpy_traces = []
    gpu_memcpy2_traces = []
    gpu_memcpy_h2d_traces = []
    gpu_memcpy_d2h_traces = []
    gpu_memcpy_d2d_traces = []
    gpu_glb_kernel_traces = []
    gpu_glb_memcpy_traces = []
    gpu_glb_memcpy2_traces = []
    gpu_glb_memcpy_h2d_traces = []
    gpu_glb_memcpy_d2h_traces = []
    gpu_glb_memcpy_d2d_traces = []

    gpulog_mode = 'w'
    gpulog_header = 'True'
    cpu_count = mp.cpu_count()

    # procs -----------------------memory---------------------- ---swap-- -
    #  r  b         swpd         free         buff        cache   si   so    bi    bo   in   cs  us  sy  id  wa  st
    #  2  0            0    400091552       936896    386150912    0    0     3    18    0    1   5   0  95   0   0
    # ============ Preprocessing VMSTAT Trace ==========================
    with open('%s/vmstat.txt' % logdir) as f:
        lines = f.readlines()
        print_info("Length of vmstat_traces = %d" % len(lines))
        if len(lines) > 0:
            vm_usr_list = []
            vm_sys_list = []
            vm_bi_list = []
            vm_bo_list = []
            vm_in_list = []
            vm_cs_list = []
            vm_wa_list = []
            vm_st_list = []
            vm_usr_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_sys_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_bi_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_bo_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_in_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_cs_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_wa_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            vm_st_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            t = 0
            for i in range(len(lines)):
                if lines[i].find('procs') == - \
                        1 and lines[i].find('swpd') == -1:
                    fields = lines[i].split()
                    if len(fields) < 17:
                        continue
                    vm_r = float(fields[0]) + 1e-5
                    vm_b = float(fields[1]) + 1e-5
                    vm_sw = float(fields[2]) + 1e-5
                    vm_fr = float(fields[3]) + 1e-5
                    vm_bu = float(fields[4]) + 1e-5
                    vm_ca = float(fields[5]) + 1e-5
                    vm_si = float(fields[6]) + 1e-5
                    vm_so = float(fields[7]) + 1e-5
                    vm_bi = float(fields[8]) + 1e-5
                    vm_bo = float(fields[9]) + 1e-5
                    vm_in = float(fields[10]) + 1e-5
                    vm_cs = float(fields[11]) + 1e-5
                    vm_usr = float(fields[12]) + 1e-5
                    vm_sys = float(fields[13]) + 1e-5
                    vm_idl = float(fields[14]) + 1e-5
                    vm_wa = float(fields[15]) + 1e-5
                    vm_st = float(fields[16]) + 1e-5

                    t_begin = t + t_glb_base
                    deviceId = cpuid = -1
                    event = -1
                    copyKind = -1
                    payload = -1
                    bandwidth = -1
                    pkt_src = pkt_dst = -1
                    pid = tid = -1
                    vmstat_info = 'r=' + str(int(vm_r)) + '|'\
                        + 'b=' + str(int(vm_b)) + '|'\
                        + 'sw=' + str(int(vm_sw)) + '|'\
                        + 'fr=' + str(int(vm_fr)) + '|'\
                        + 'bu=' + str(int(vm_bu)) + '|'\
                        + 'ca=' + str(int(vm_ca)) + '|'\
                        + 'si=' + str(int(vm_si)) + '|'\
                        + 'so=' + str(int(vm_so)) + '|'\
                        + 'bi=' + str(int(vm_bi)) + '|'\
                        + 'bo=' + str(int(vm_bo)) + '|'\
                        + 'in=' + str(int(vm_in)) + '|'\
                        + 'cs=' + str(int(vm_cs)) + '|'\
                        + 'usr=' + str(int(vm_usr)) + '|'\
                        + 'sys=' + str(int(vm_sys)) + '|'\
                        + 'idl=' + str(int(vm_idl)) + '|'\
                        + 'wa=' + str(int(vm_wa)) + '|'\
                        + 'st=' + str(int(vm_st))

                    trace = [
                        t_begin,
                        event,
                        vm_bi,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_bi_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_bo,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_bo_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_in,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_in_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_cs,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_cs_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_wa,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_wa_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_st,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_st_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_usr,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_usr_list.append(trace)

                    trace = [
                        t_begin,
                        event,
                        vm_sys,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        vmstat_info,
                        cpuid]
                    vm_sys_list.append(trace)

                    t = t + 1

            vm_bi_traces = list_to_csv_and_traces(
                logdir, vm_bi_list, 'vmstat_trace.csv', 'w')
            vm_bo_traces = list_to_csv_and_traces(
                logdir, vm_bo_list, 'vmstat_trace.csv', 'a')
            vm_in_traces = list_to_csv_and_traces(
                logdir, vm_in_list, 'vmstat_trace.csv', 'a')
            vm_cs_traces = list_to_csv_and_traces(
                logdir, vm_cs_list, 'vmstat_trace.csv', 'a')
            vm_wa_traces = list_to_csv_and_traces(
                logdir, vm_wa_list, 'vmstat_trace.csv', 'a')
            vm_st_traces = list_to_csv_and_traces(
                logdir, vm_st_list, 'vmstat_trace.csv', 'a')
            vm_usr_traces = list_to_csv_and_traces(
                logdir, vm_usr_list, 'vmstat_trace.csv', 'a')
            vm_sys_traces = list_to_csv_and_traces(
                logdir, vm_sys_list, 'vmstat_trace.csv', 'a')

    # gpu    sm   mem   enc   dec
    # Idx     %     %     %     %
    #        0     0     0     0     0
    #        1     0     0     0     0
    #        2     0     0     0     0
    if os.path.isfile('%s/nvsmi.txt' % logdir):
        with open('%s/nvsmi.txt' % logdir) as f:
            lines = f.readlines()
            nvsmi_has_data = True 
            for line in lines:
                if line.find('failed') != -1 or line.find('Failed') != -1: 
                    nvsmi_has_data = False
                    print_warning('No nvsmi data.')
                    break
            if nvsmi_has_data: 
                print_info("Length of nvsmi_traces = %d" % len(lines))
                nvsmi_sm_list = []
                nvsmi_mem_list = []
                nvsmi_sm_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                nvsmi_mem_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                t = 0
                for i in range(len(lines)):
                    if lines[i].find('gpu') == -1 and lines[i].find('Idx') == -1:
                        fields = lines[i].split()
                        if len(fields) < 5:
                            continue
                        nvsmi_id = int(fields[0])
                        nvsmi_sm = float(fields[1]) + 1e-5
                        nvsmi_mem = float(fields[2]) + 1e-5

                        t_begin = t + t_glb_base
                        deviceId = cpuid = nvsmi_id
                        event = -1
                        copyKind = -1
                        payload = -1
                        bandwidth = -1
                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        nvsmi_info = "GPUID_sm_mem=%d_%lf_%lf" % (
                            nvsmi_id, nvsmi_sm, nvsmi_mem)

                        trace = [
                            t_begin,
                            event,
                            nvsmi_sm,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            nvsmi_info,
                            cpuid]
                        if t > 3 :
                            nvsmi_sm_list.append(trace)

                        trace = [
                            t_begin,
                            event,
                            nvsmi_mem,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            nvsmi_info,
                            cpuid]
                        if t > 3 :
                            nvsmi_mem_list.append(trace)
                        if nvsmi_id == 0:
                            t = t + 1
                if len(nvsmi_sm_list)>1:
                    nvsmi_sm_traces = list_to_csv_and_traces(logdir, nvsmi_sm_list, 'nvsmi_trace.csv', 'w')
                    nvsmi_mem_traces = list_to_csv_and_traces(logdir, nvsmi_mem_list, 'nvsmi_trace.csv', 'a')

    # ============ Preprocessing Network Trace ==========================
    filtered_net_groups = []
    if os.path.isfile('%s/sofa.pcap' % logdir):
        with open(logdir + 'net.tmp', 'w') as f:
            subprocess.check_call(
                ["tcpdump", "-q", "-n", "-tt", "-r",
                "%s/sofa.pcap"%logdir ], stdout=f)
        with open(logdir + 'net.tmp') as f:
            packets = lines = f.readlines()
            print_info("Length of net_traces = %d" % len(packets))
            if packets:
                with mp.Pool(processes=cpu_count) as pool:
                    res = pool.map(
                        partial(
                            net_trace_read,
                            t_offset=0),
                        packets)
                res_viz = list_downsample(res, cfg.plot_ratio)
                net_traces = pd.DataFrame(res_viz)
                net_traces.columns = sofa_fieldnames
                net_traces.to_csv(
                    logdir + 'cputrace.csv',
                    mode='a',
                    header=False,
                    index=False,
                    float_format='%.6f')
    
                # ============ Apply for Network filter =====================
                if cfg.net_filters:
                    packet_not_zero = net_traces['payload'] > 0
                    start = (net_traces['pkt_src'] == float(cfg.net_filters[0]))
                    for filter in cfg.net_filters[1:]:
                        end = (net_traces['pkt_dst'] == float(filter))
                        group = net_traces[packet_not_zero & start & end]
                        filtered_net_groups.append({'group': group,
                                                    'color': 'rgba(%s,%s,%s,0.8)' %(randint(0,255),randint(0,255),randint(0,255)),
                                                 'keyword': 'to_%s' %filter})

                    end = (net_traces['pkt_dst'] == float(cfg.net_filters[0]))
                    for filter in cfg.net_filters[1:]:
                        start = (net_traces['pkt_src'] == float(filter))
                        group = net_traces[packet_not_zero & start & end]
                        filtered_net_groups.append({'group': group,
                                                    'color': 'rgba(%s,%s,%s,0.8)' %(randint(0,255),randint(0,255),randint(0,255)),
                                                    'keyword': 'from_%s' %filter})
    else:
        print_warning("no network traces were recorded.")

    # ============ Preprocessing GPU Trace ==========================
    num_cudaproc = 0
    filtered_gpu_groups = []
    indices = []
    for nvvp_filename in glob.glob(logdir + "gputrace*[0-9].nvvp"):
        print_progress("Read " + nvvp_filename + " by nvprof -- begin")
        with open(logdir + "gputrace.tmp", "w") as f:
            subprocess.call(["nvprof", "--csv", "--print-gpu-trace", "-i", nvvp_filename], stderr=f)

        #Automatically retrieve the timestamp of the first CUDA activity(e.g. kernel, memory op, etc..)
        engine = create_engine("sqlite:///"+nvvp_filename)
        t_glb_gpu_bases = []

        try:
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMSET',engine)).iloc[0]['start'])
        except BaseException:
            print_info('NO MEMSET')
        try:
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMCPY',engine)).iloc[0]['start'])
        except BaseException:
            print_info('NO MEMCPY')
        try: 
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL',engine)).iloc[0]['start'])
        except BaseException:
            print_info('NO CONCURRENT KERNEL')
        try: 
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_KERNEL',engine)).iloc[0]['start'])
        except BaseException:
            print_info('NO KERNEL')
        print(t_glb_gpu_bases)
        if len(t_glb_gpu_bases) > 0: 
            t_glb_gpu_base = sorted(t_glb_gpu_bases)[0]*1.0/1e+9
        else:
           print_warning("There is no data in tables of NVVP file.") 

        print_info("Timestamp of the first GPU trace = " + str(t_glb_gpu_base))
       
        print_progress("Read " + nvvp_filename + " by nvprof -- end")
        num_cudaproc = num_cudaproc + 1
        with open(logdir + 'gputrace.tmp') as f:
            records = f.readlines()
            # print(records[1])

            if len(records) > 0 and records[1].split(',')[0] == '"Start"':
                indices = records[1].replace(
                    '"', '').replace(
                    '\n', '').split(',')
                print(indices)
                # ms,ms,,,,,,,,B,B,MB,GB/s,,,,
                payload_unit = 1
                if records[2].split(',')[11] == 'GB':
                    payload_unit = np.power(1024,3)
                elif records[2].split(',')[11] == 'MB':
                    payload_unit = np.power(1024,2)
                elif records[2].split(',')[11] == 'KB':
                    payload_unit = np.power(1024,1)
                elif records[2].split(',')[11] == 'B':
                    payload_unit = 1
                else: 
                    print_info("The payload unit in gputrace.tmp was not recognized!")
                    quit()
                
                ts_rescale = 1.0
                if records[2].split(',')[0] == 'ms':
                    ts_rescale = 1.0e3
                elif records[2].split(',')[0] == 'us':
                    ts_rescale = 1.0e6

                dt_rescale = 1.0
                if records[2].split(',')[1] == 'ms':
                    dt_rescale = 1.0e3
                elif records[2].split(',')[1] == 'us':
                    dt_rescale = 1.0e6

                records = records[3:]
                print_info("Length of gpu_traces = %d" % len(records))
                t_base = float(records[0].split(',')[0])
                with mp.Pool(processes=cpu_count) as pool:
                    res = pool.map(
                        partial(
                            gpu_trace_read,
                            indices=indices,
                            ts_rescale=ts_rescale,
                            dt_rescale=dt_rescale,
                            payload_unit=payload_unit,
                            n_cudaproc=num_cudaproc,
                            t_offset=t_glb_gpu_base -
                            t_base),
                        records)
                gpu_traces = pd.DataFrame(res)
                gpu_traces.columns = sofa_fieldnames
                res_viz = list_downsample(res, cfg.plot_ratio)
                gpu_traces_viz = pd.DataFrame(res_viz)
                gpu_traces_viz.columns = sofa_fieldnames

                gpu_traces.to_csv(
                    logdir + 'gputrace.csv',
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.6f')

                # Apply filters for GPU traces
                df_grouped = gpu_traces.groupby('name')
                for filter in cfg.gpu_filters:
                    group = gpu_traces[gpu_traces['name'].str.contains(
                        filter.keyword)]
                    filtered_gpu_groups.append({'group': group, 'color': filter.color,
                                                'keyword': filter.keyword})
            else:
                print_warning(
                    "gputrace existed, but no kernel traces were recorded.")
                os.system('cat %s/gputrace.tmp' % logdir)
 
    # ============ Preprocessing GPU API Trace ==========================
    if cfg.cuda_api_tracing:
        num_cudaproc = 0
        indices = []
        for nvvp_filename in glob.glob(logdir + "gputrace*[0-9].nvvp"):
            print_progress("Read " + nvvp_filename + " for API traces by nvprof -- begin")
            with open(logdir + "cuda_api_trace.tmp", "w") as f:
                subprocess.call(["nvprof", "--csv", "--print-api-trace", "-i", nvvp_filename], stderr=f)

            #Automatically retrieve the timestamp of the first CUDA activity(e.g. kernel, memory op, etc..)
            engine = create_engine("sqlite:///"+nvvp_filename)
            t_glb_gpu_bases = []
            first_corid = 1
            try:
                t_glb_gpu_bases.append((pd.read_sql_table('CUPTI_ACTIVITY_KIND_RUNTIME',engine)).iloc[0]['start'])
                first_corid = (pd.read_sql_table('CUPTI_ACTIVITY_KIND_RUNTIME',engine)).iloc[0]['correlationId']
            except BaseException:
                print_info('NO RUNTIME')

            if len(t_glb_gpu_bases) > 0: 
                t_glb_gpu_base = sorted(t_glb_gpu_bases)[0]*1.0/1e+9
            else:
               print_warning("There is no data in tables of NVVP file.") 

            print_info("Timestamp of the first CUDA API trace = " + str(t_glb_gpu_base))
           
            print_progress("Read " + nvvp_filename + " by nvprof -- end")
            num_cudaproc = num_cudaproc + 1
            with open(logdir + 'cuda_api_trace.tmp') as f:
                records = f.readlines()
                # print(records[1])
                if len(records) > 0 and records[1].split(',')[0] == '"Start"':
                    indices = records[1].replace(
                        '"', '').replace(
                        '\n', '').split(',')
                    print(indices)
                    
                    ts_rescale = 1.0
                    if records[2].split(',')[0] == 'ms':
                        ts_rescale = 1.0e3
                    elif records[2].split(',')[0] == 'us':
                        ts_rescale = 1.0e6

                    dt_rescale = 1.0
                    if records[2].split(',')[1] == 'ms':
                        dt_rescale = 1.0e3
                    elif records[2].split(',')[1] == 'us':
                        dt_rescale = 1.0e6

                    records = records[3:]
                    print_info("Length of cuda_api_traces = %d" % len(records))
                   
                    #TODO: Apply parallel search to speed up  
                    t_base = float(records[0].split(',')[0])
                    if len(records[0].split(',')) == 4: 
                        for record in records:
                            if int(record.split(',')[3]) == first_corid:
                                t_base = float(record.split(',')[0]) 
                                print_info('First Correlation_ID ' + str(first_corid) + ' is found in cuda_api_trace.tmp')
                                print_info('First API trace timestamp is ' + str(t_base))
                                break 
                    
                    with mp.Pool(processes=cpu_count) as pool:
                        res = pool.map(
                            partial(
                                cuda_api_trace_read,
                                indices=indices,
                                ts_rescale=ts_rescale,
                                dt_rescale=dt_rescale,
                                payload_unit=payload_unit,
                                n_cudaproc=num_cudaproc,
                                t_offset=t_glb_gpu_base -
                                t_base),
                            records)
                    cuda_api_traces = pd.DataFrame(res)
                    cuda_api_traces.columns = sofa_fieldnames
                    res_viz = list_downsample(res, cfg.plot_ratio)
                    cuda_api_traces_viz = pd.DataFrame(res_viz)
                    cuda_api_traces_viz.columns = sofa_fieldnames

                    cuda_api_traces.to_csv(
                        logdir + 'cuda_api_trace.csv',
                        mode='w',
                        header=True,
                        index=False,
                        float_format='%.6f')
    
    # ============ Preprocessing CPU Trace ==========================
    with open(logdir+'perf_events_used.txt','r') as f:
        lines = f.readlines()
        if lines: 
            cfg.perf_events = lines[0]
        else:
            cfg.perf_events = ''
        print_info('perf_events_used: %s' % (cfg.perf_events))
    # Determine time base for perf traces
    perf_timebase_uptime = 0
    perf_timebase_unix = 0
    last_nvvp_ts = 0
    for nvvp_filename in glob.glob(logdir + "cuhello*[0-9].nvvp"):
        print_progress("Read " + nvvp_filename + " by nvprof -- begin")
        engine = create_engine('sqlite:///' + nvvp_filename)
        last_nvvp_tss = []
        try:
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMSET',engine)).iloc[-1]['start'])
        except BaseException:
            print_info('NO MEMSET')
        
        try:
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMCPY',engine)).iloc[-1]['start'])
        except BaseException:
            print_info('NO MEMCPY')
        
        try: 
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL',engine)).iloc[-1]['start'])
        except BaseException:
            print_info('NO CONCURRENT KERNEL')
        
        try: 
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_KERNEL',engine)).iloc[-1]['start'])
        except BaseException:
            print_info('NO KERNEL')
        if len(last_nvvp_tss) > 0: 
            last_nvvp_ts = sorted(last_nvvp_tss,reverse=True)[0]*1.0/1e+9
        else:
           print_warning("There is no data in tables of NVVP file.") 
        
        with open(logdir + 'cuhello.perf.script', 'w') as logfile:
            subprocess.call(['perf',
                     'script',
                     '--kallsym',
                     '%s/kallsyms' % logdir,
                     '-i',
                     '%s/cuhello.perf.data' % logdir,
                     '-F',
                     'time,pid,tid,event,ip,sym,dso,symoff,period,brstack,brstacksym'],
                    stdout=logfile)

        with open(logdir + 'cuhello.perf.script') as f:
            samples = f.readlines()
            print_info("Length of cpu_traces = %d" % len(samples))
            if len(samples) > 0:
                for sample in reversed(samples):
                    fields = sample.split()
                    function_name = "" 
                    if re.match('\[\d+\]', fields[1]) is not None:
                        function_name = '[%s]'%fields[4].replace('-','_') + fields[6] + fields[7] 
                    else:
                        function_name = '[%s]'%fields[3].replace('-','_')  + fields[5] + fields[6] 

                    if function_name.find('libcuda.so') != -1 and len(last_nvvp_tss)>0: 
                        perf_timebase_uptime = float(sample.split()[1].split(':')[0])
                        perf_timebase_unix = last_nvvp_ts 
                        break
        print_progress("Read " + nvvp_filename + " by nvprof -- end")
    
    if perf_timebase_unix == 0:
        with open(logdir + 'perf_timebase.txt') as f:
            lines = f.readlines()
            perf_timebase_uptime = float(lines[-2].split()[2].split(':')[0])
            perf_timebase_unix = float(lines[-1].split()[0])
    
    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        print_info("Length of cpu_traces = %d" % len(samples))
        if len(samples) > 0:
            with mp.Pool(processes=cpu_count) as pool:
                res = pool.map(
                    partial(
                        cpu_trace_read,
                        t_offset = perf_timebase_unix - perf_timebase_uptime,
                        cfg = cfg,
                        cpu_mhz_xp = cpu_mhz_xp,
			cpu_mhz_fp = cpu_mhz_fp),
                    samples)
            cpu_traces = pd.DataFrame(res)                      
            cpu_traces.columns = sofa_fieldnames
            cpu_traces.to_csv(
                logdir + 'cputrace.csv',
                mode='w',
                header=True,
                index=False,
                float_format='%.6f')
            res_viz = list_downsample(res, cfg.plot_ratio)
            cpu_traces_viz = pd.DataFrame(res_viz)                
            cpu_traces_viz.columns = sofa_fieldnames
            char1 = ']'
            char2 = '+'
            # demangle c++ symbol, little dirty work here...
            cpu_traces_viz['name'] = cpu_traces_viz['name'].apply(
                lambda x: cxxfilt.demangle(str( x[x.find(char1)+1 : x.find(char2)].split('@')[0] ))
            )            
        ###  Apply filters for cpu traces
        filtered_groups = []
        if len(cpu_traces) > 0:
            df_grouped = cpu_traces_viz.groupby('name')
            for filter in cfg.cpu_filters:
                group = cpu_traces_viz[cpu_traces_viz['name'].str.contains(
                    filter.keyword)]
                filtered_groups.append({'group': group,
                                        'color': filter.color,
                                        'keyword': filter.keyword})
    ### hierarchical swarm generation
    if cfg.enable_hsg:
        with open(logdir + 'perf.script') as f, warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            samples = f.readlines()
            print_info("Length of cpu_traces = %d" % len(samples))
            if len(samples) > 0:
                with mp.Pool(processes=cpu_count) as pool:
                    res = pool.map(
                        partial(
                            cpu_trace_read_hsg,
                            t_offset = perf_timebase_unix - perf_timebase_uptime,
                            cfg = cfg,
                            cpu_mhz_xp = cpu_mhz_xp,
                            cpu_mhz_fp = cpu_mhz_fp
                            ),
                        samples)                
                cpu_traces = pd.DataFrame(res)      
                sofa_fieldnames_ext = sofa_fieldnames + ['feature_types', 'mem_addr'] # if you want to add customized column, add to the latter
                cpu_traces.columns = sofa_fieldnames_ext          
                cpu_traces.to_csv(
                    logdir + 'cputrace.csv',
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.6f')
                res_viz = list_downsample(res, cfg.plot_ratio)
                swarm_cpu_traces_viz = pd.DataFrame(res_viz)
                swarm_cpu_traces_viz.columns = sofa_fieldnames_ext        
               
                char1 = ']'
                char2 = '+'      
                # demangle c++ symbol, little dirty work here...
                swarm_cpu_traces_viz['name'] = swarm_cpu_traces_viz['name'].apply(
                    lambda x: cxxfilt.demangle(str( x[x.find(char1)+1 : x.find(char2)].split('@')[0] ))
                )

                ### N features ###
                ## give unique id of each data within 10 msec by time quotient
                swarm_cpu_traces_viz['quotient'] = swarm_cpu_traces_viz['timestamp'].apply(lambda x: int( x * 1000 // 10)) # //: quotient

                # count feature_types in each 10 msec groups, and create a dictionary for mapping
                df2s = {}
                for quotient, dataframe in swarm_cpu_traces_viz.groupby(['quotient','event']):
                    # api value_counts(): return pandas series
                    df2s[quotient] = dataframe.feature_types.value_counts()
                df2 = pd.DataFrame.from_dict(df2s, orient='index').fillna(0).astype(np.int64)

                df = swarm_cpu_traces_viz.copy()
                swarm_cpu_traces_viz = pd.merge(df, df2, left_on=['quotient','event'], right_index=True).copy()                
            

            ### swarm seperation by memory location 
            swarm_groups = []        
            feature_list = ['event']
            if cfg.hsg_multifeatures:
                with open(logdir+'perf_events_used.txt','r') as f:
                    lines = f.readlines()
                    feature_list.extend(lines[0].split(','))
                try:
                    feature_list.remove('cycles')
                    feature_list.remove('event')
                except:
                    pass
           
            print_info('HSG features: '+','.join(feature_list))
            
            idx = 0
            showing_idx = 0

            if len(cpu_traces) > 0:            
                # get memory index by cheange float to integer
                swarm_cpu_traces_viz['event_int'] = swarm_cpu_traces_viz.event.apply(lambda x: int(x)) # add new column 'event_int'            
                # swarm seperate
                event_groups = swarm_cpu_traces_viz.groupby('event_int')
                swarm_stats = []
                # add different swarm groups                        
                for mem_index, l1_group in event_groups:                                
                    # kmeans 
                    X = pd.DataFrame(l1_group['event'])                                
                    num_of_cluster = 2
                    y_pred = kmeans_cluster(num_of_cluster, X)

                    # add new column
                    # TODO: Eliminate warning of SettingWithCopyWarning 
                    l1_group['cluster'] = y_pred
                    #for i in range(len(y_pred)):
                    #    group.loc[i, 'cluster'] = y_pred[i]
                    
                    # group by new column
                    clusters = l1_group.groupby('cluster')
                                    
                    for l2_group_idx, l2_group in clusters:                       
                        # group by process id
                        #pid_clusters = cluster.groupby('pid')
                        X = pd.DataFrame(l2_group['event'])                                
                        num_of_cluster = 4
                        y_pred = kmeans_cluster(num_of_cluster, X)

                        # add new column 
                        l2_group['cluster'] = y_pred                 
                        #for i in range(len(y_pred)):
                        #    l2_group.loc[i, 'cluster'] = y_pred[i]       
                        
                        # group by new column
                        last_clusters = l2_group.groupby('cluster')
 
                        for last_cluster_idx, last_cluster in last_clusters:                              
                            # kmeans
                            X = pd.DataFrame(last_cluster[feature_list])
                            num_of_cluster = 4
                            y_pred_pid_cluster = kmeans_cluster(num_of_cluster, X)

                            # add new column
                            last_cluster['cluster_in_pid'] = y_pred_pid_cluster
                            # group by new column
                            cluster_in_pid_clusters = last_cluster.groupby('cluster_in_pid')

                            for mini_cluster_id, cluster_in_pid_cluster in cluster_in_pid_clusters:                                  
                                # duration time
                                total_duration = cluster_in_pid_cluster.duration.sum()                            
                                mean_duration = cluster_in_pid_cluster.duration.mean()      

                                # swarm diff
                                # caption: assign mode of function name              
                                mode = str(cluster_in_pid_cluster['name'].mode()[0]) # api pd.Series.mode() returns a pandas series                                
                                mode = mode.replace('::', '@') # str.replace(old, new[, max])
                                #print('mode of this cluster: {}'.format(str(mode[:35])))

                                swarm_stats.append({'keyword': 'SWARM_' + '["' + str(mode[:35]) + ']' +  ('_' * showing_idx),
                                                    'duration_sum': total_duration,
                                                    'duration_mean': mean_duration,
                                                    'example':cluster_in_pid_cluster.head(1)['name'].to_string().split('  ')[2]}) 

                                swarm_groups.append({'group': cluster_in_pid_cluster.drop(columns = ['event_int', 'cluster', 'cluster_in_pid']), # data of each group
                                                     'color':  random_generate_color(),                                                     
                                                     'keyword': 'SWARM_' + '[' + str(mode[:35]) + ']' +  ('_' * showing_idx),
                                                     'total_duration': total_duration})                                                    
                                idx += 1
                                            
                swarm_groups.sort(key=itemgetter('total_duration'), reverse = True) # reverse = True: descending                
                swarm_stats.sort(key=itemgetter('duration_sum'), reverse = True)
                print_title('HSG Statistics - Top-%d Swarms'%(cfg.num_swarms))

                for i in range(len(swarm_stats)):
                    if i >= cfg.num_swarms:
                        break
                    else:
                        swarm = swarm_stats[i]
                        print('%s: execution_time(sum,mean): %.6lf(s),%.6lf(s)  caption: %s'%(swarm['keyword'],swarm['duration_sum']/4.0,swarm['duration_mean']/4.0,swarm['example']))

    #=== Intel PCM Trace =======#
    ### Skt,PCIeRdCur,RFO,CRd,DRd,ItoM,PRd,WiL,PCIe Rd (B),PCIe Wr (B)
    ### 0,0,852,0,0,48,0,0,54528,57600
    ### 1,0,600,0,0,0,0,0,38400,38400
    if cfg.enable_pcm and os.path.isfile('%s/pcm_pcie.csv' % logdir):
        with open( logdir + '/pcm_pcie.csv' ) as f:
            lines = f.readlines()
            print_info("Length of pcm_pcie_traces = %d" % len(lines))
            if len(lines) > 0:
                pcm_pcie_list = []
                pcm_pcie_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                for line in lines:
                    if line.find('Skt') == -1:
                        fields = line.split(',')
                        #for f in range(len(fields)):
                        #    print("field[%d] %s" % (f, fields[f]))
                       
                        skt = int(fields[1])
                        t_begin = float(fields[0]) 
                        deviceId = skt
                        event = -1
                        copyKind = -1
                        payload = -1
                        pcm_pcie_wt_count = int(float(fields[2])*64/1e3)+1e-6
                        pcm_pcie_rd_count = int(float(fields[6])*64/1e3)+1e-6
                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        pcm_pcie_info = "PCM=pcie | skt=%d | RD=%d (KB)" % (
                            skt, pcm_pcie_rd_count)

                        bandwidth = pcm_pcie_rd_count
                        trace = [
                            t_begin,
                            event,
                            bandwidth,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            pcm_pcie_info,
                            cpuid]
                        pcm_pcie_list.append(trace)
 
                        pcm_pcie_info = "PCM=pcie | skt=%d | WT=%d (KB)" % (
                            skt, pcm_pcie_wt_count)
                        bandwidth = pcm_pcie_wt_count
                        trace = [
                            t_begin,
                            event,
                            bandwidth,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            pcm_pcie_info,
                            cpuid]
                        pcm_pcie_list.append(trace)                 
                pcm_pcie_traces = list_to_csv_and_traces(logdir, pcm_pcie_list, 'pcm_pcie_trace.csv', 'w')

            else:
                print_warning('No pcm-pcie counter values are recorded.')
                print_warning('If necessary, run /usr/local/intelpcm/bin/pcm-pcie.x ONCE to reset MSR so as to enable correct pcm recording')
   
    ### time, skt, iMC_Read, iMC_Write [, partial_write] [, EDC_Read, EDC_Write] , sysRead, sysWrite, sysTotal
    if cfg.enable_pcm and os.path.isfile('%s/pcm_memory.csv' % logdir):
        with open( logdir + '/pcm_memory.csv' ) as f:
            lines = f.readlines()
            print_info("Length of pcm_memory_traces = %d" % len(lines))
            if len(lines) > 0:
                pcm_memory_list = []
                pcm_memory_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                for line in lines:
                    if line.find('Skt') == -1:
                        fields = line.split(',')
                        #for f in range(len(fields)):
                        #    print("field[%d] %s" % (f, fields[f]))
                       
                        skt = int(fields[1])
                        t_begin = float(fields[0]) 
                        deviceId = skt
                        event = -1
                        copyKind = -1
                        payload = -1
                        try: 
                            pcm_memory_wt_count = int(float(fields[3]))
                            pcm_memory_rd_count = int(float(fields[2]))
                        except:
                            pcm_memory_wt_count=0
                            pcm_memory_rd_count=0

                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        pcm_memory_info = "PCM=memory | skt=%d | RD=%d (MB/s)" % (
                            skt, pcm_memory_rd_count)

                        bandwidth = pcm_memory_rd_count
                        trace = [
                            t_begin,
                            event,
                            bandwidth,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            pcm_memory_info,
                            cpuid]
                        pcm_memory_list.append(trace)
 
                        pcm_memory_info = "PCM=memory | skt=%d | WT=%d (MB/s)" % (
                            skt, pcm_memory_wt_count)
                        bandwidth = pcm_memory_wt_count
                        trace = [
                            t_begin,
                            event,
                            bandwidth,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            pcm_memory_info,
                            cpuid]
                        pcm_memory_list.append(trace)                 
                pcm_memory_traces = list_to_csv_and_traces(logdir, pcm_memory_list, 'pcm_memory_trace.csv', 'w')

            else:
                print_warning('No pcm-memory counter values are recorded.')
                print_warning('If necessary, run /usr/local/intelpcm/bin/pcm-memory.x ONCE to reset MSR so as to enable correct pcm recording')
    
    print_progress(
        "Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- begin")

    # TODO: provide option to use absolute or relative timestamp
    # cpu_traces.loc[:,'timestamp'] -= cpu_traces.loc[0,'timestamp']
    # net_traces.loc[:,'timestamp'] -= net_traces.loc[0,'timestamp']
    # gpu_traces.loc[:,'timestamp'] -= gpu_traces.loc[0,'timestamp']

    traces = []
    sofatrace = SOFATrace()
    sofatrace.name = 'cpu_trace'
    sofatrace.title = 'CPU'
    sofatrace.color = 'DarkGray'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = cpu_traces_viz
    traces.append(sofatrace)

    for filtered_group in filtered_groups:
        sofatrace = SOFATrace()
        sofatrace.name = filtered_group['keyword']
        sofatrace.title = '[keyword]' + sofatrace.name
        sofatrace.color = filtered_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_group['group'].copy()
        traces.append(sofatrace)
    

    if cfg.enable_hsg:
        dummy_i = 0
        record_for_auto_caption = True # temperarily: for auto-caption

        # top 10 cumulative time of a swarm
        number_of_swarm = cfg.num_swarms
        ##### temperarily: for auto-caption #####
        if record_for_auto_caption == True:
            number_of_swarm += 5 # record 15 swarms            
            # write empty csv file
            auto_caption_filename_with_path = logdir + 'auto_caption.csv'
            with open(auto_caption_filename_with_path, 'w') as f: # write to csv file
                pass # keep it emtpy        
        ##### temperarily: for auto-caption #####

        for swarm in swarm_groups[:number_of_swarm]: 
            sofatrace = SOFATrace()            
            sofatrace.name = 'swarm' + str(dummy_i) # Avoid errors casued by JavaScript. No special meaning, can be random unique ID.
            sofatrace.title = swarm['keyword'] # add number of swarm
            sofatrace.color = swarm['color']        
            sofatrace.x_field = 'timestamp'
            sofatrace.y_field = 'duration'        
            sofatrace.data = swarm['group'].copy()                               
            traces.append(sofatrace)            

            ##### temperarily: for auto-caption #####
            if record_for_auto_caption == True:                                
                # append to csv file every time using pandas funciton
                swarm['group']['cluster_ID'] = dummy_i # add new column cluster ID to dataframe swarm['group']
                copy = swarm['group'].copy()
                copy.to_csv(auto_caption_filename_with_path, mode='a', header=False, index=False)                                                        
                print( copy.head(2) )
            ##### temperarily: for auto-caption #####
            dummy_i += 1

    sofatrace = SOFATrace()
    sofatrace.name = 'vmstat_bi'
    sofatrace.title = 'VMSTAT_BI'
    sofatrace.color = 'DarkOrange'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = vm_bi_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'vmstat_bo'
    sofatrace.title = 'VMSTAT_BO'
    sofatrace.color = 'DarkOrchid'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = vm_bo_traces
    traces.append(sofatrace)
    
    if cfg.enable_vmstat:
        sofatrace = SOFATrace()
        sofatrace.name = 'vmstat_usr'
        sofatrace.title = 'CPU_USAGE_USR'
        sofatrace.color = 'Magenta'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = vm_usr_traces
        traces.append(sofatrace)
    
        sofatrace = SOFATrace()
        sofatrace.name = 'vmstat_sys'
        sofatrace.title = 'CPU_USAGE_SYS'
        sofatrace.color = 'LightBlue'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = vm_sys_traces

        traces.append(sofatrace)
        sofatrace = SOFATrace()
        sofatrace.name = 'vmstat_in'
        sofatrace.title = 'VMSTAT_IN'
        sofatrace.color = 'DarkMagenta'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = vm_in_traces
        traces.append(sofatrace)

        sofatrace = SOFATrace()
        sofatrace.name = 'vmstat_cs'
        sofatrace.title = 'VMSTAT_CS'
        sofatrace.color = 'DarkOliveGreen'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = vm_cs_traces
        traces.append(sofatrace)



    sofatrace = SOFATrace()
    sofatrace.name = 'nvsmi_mem'
    sofatrace.title = 'GPU_MEM_Util.'
    sofatrace.color = 'lightblue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = nvsmi_mem_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'nvsmi_sm'
    sofatrace.title = 'GPU_SM_Util.'
    sofatrace.color = 'red'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = nvsmi_sm_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'pcm_pcie'
    sofatrace.title = 'PCM_PCIE'
    sofatrace.color = 'purple'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'bandwidth'
    sofatrace.data = pcm_pcie_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'pcm_memory'
    sofatrace.title = 'PCM_MEMORY'
    sofatrace.color = 'pink'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'bandwidth'
    sofatrace.data = pcm_memory_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'net_trace'
    sofatrace.title = 'NET'
    sofatrace.color = 'blue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = net_traces
    traces.append(sofatrace)
    
    if cfg.net_filters:
        for filtered_net_group in filtered_net_groups:
            sofatrace = SOFATrace()
            sofatrace.name = filtered_net_group['keyword']
            sofatrace.title = '[keyword]' + sofatrace.name
            sofatrace.color = filtered_net_group['color']
            sofatrace.x_field = 'timestamp'
            sofatrace.y_field = 'duration'
            sofatrace.data = filtered_net_group['group'].copy()
            traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_kernel_trace'
    sofatrace.title = 'GPU kernel'
    sofatrace.color = 'rgba(0,180,0,0.8)'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_traces_viz
    traces.append(sofatrace)

    for filtered_gpu_group in filtered_gpu_groups:
        sofatrace = SOFATrace()
        sofatrace.name = filtered_gpu_group['keyword']
        sofatrace.title = '[keyword]' + sofatrace.name
        sofatrace.color = filtered_gpu_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_gpu_group['group'].copy()
        traces.append(sofatrace)

    if cfg.cuda_api_tracing:
        sofatrace = SOFATrace()
        sofatrace.name = 'cuda_api_trace'
        sofatrace.title = 'CUDA API'
        sofatrace.color = 'DarkSlateGray'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = cuda_api_traces_viz
        traces.append(sofatrace)

    traces_to_json(traces, logdir + 'report.js', cfg)
    print_progress(
        "Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- end")