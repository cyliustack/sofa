import argparse
import csv
import glob
import itertools
import json
import multiprocessing as mp
import os
import re
import datetime

import subprocess
import sys
import warnings
from functools import partial
from operator import itemgetter

import cxxfilt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sqlalchemy import create_engine

from sofa_config import *
from sofa_hsg import sofa_hsg, sofa_hsg_to_sofatrace
from sofa_models import SOFATrace
from sofa_print import *

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

def list_downsample(list_in, plot_ratio):
    new_list = []
    for i in range(len(list_in)):
        if i % plot_ratio == 0:
            # print("%d"%(i))
            new_list.append(list_in[i])
    return new_list

def trace_init():
    t_begin = 0 
    deviceId = 0
    metric = 0
    event = -1
    copyKind = -1
    payload = -1
    bandwidth = -1
    pkt_src = pkt_dst = -1
    pid = tid = -1
    name = ''
    category = 0

    trace = [
       t_begin,
       event,
       metric,
       deviceId,
       copyKind,
       payload,
       bandwidth,
       pkt_src,
       pkt_dst,
       pid,
       tid,
       name,
       category]
    
    return trace

def list_to_csv_and_traces(logdir, _list, csvfile, _mode):
    traces = []
    if len(_list[1:]) > 0:
    	traces = pd.DataFrame(_list[1:])
    	traces.columns = sofa_fieldnames
    	_header = True if _mode == 'w' else False
    	traces.to_csv(logdir +
    	              csvfile,
    	              mode=_mode,
    	              header=_header,
    	              index=False,
    	              float_format='%.6f')
    else:
        print_warning('Empty list cannot be exported to %s!' % csvfile)
    return traces

# 0/0     [004] 96050.733788:          1 bus-cycles:  ffffffff8106315a native_write_msr_safe
# 0/0     [004] 96050.733788:          7     cycles:  ffffffff8106315a native_write_msr_safe
# 359342/359342 2493492.850125:          1 bus-cycles:  ffffffff8106450a native_write_msr_safe
# 359342/359342 2493492.850128:          1 cycles:  ffffffff8106450a
# native_write_msr_safe

def cpu_trace_read(sample, cfg, t_offset, cpu_mhz_xp, cpu_mhz_fp):
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

    if not cfg.absolute_timestamp:
        time = time - cfg.time_base

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

def net_trace_read(packet, cfg, t_offset):
    time = float(packet.split()[0])
    
    if not cfg.absolute_timestamp:
        time = time - cfg.time_base
    
    t_begin = time + t_offset
    t_end = time + t_offset
    if packet.split()[1] != 'IP':
        return []
    payload = int(packet.split()[6])
    duration = float(payload / 128.0e6)
    bandwidth = 128.0e6
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
        cfg,
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

    if not cfg.absolute_timestamp:
        time = time - cfg.time_base

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
        cfg, 
        indices,
        n_cudaproc,
        ts_rescale,
        dt_rescale,
        payload_unit,
        t_offset):
    values = record.replace('"', '').split(',')
    kernel_name = values[indices.index('Name')]

    time = float(values[indices.index('Start')]) / ts_rescale + t_offset
    
    if not cfg.absolute_timestamp:
        time = time - cfg.time_base
    
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


def sofa_preprocess(cfg):
    cfg.time_base = 0
    t_glb_gpu_base = 0
    logdir = cfg.logdir
    with open(logdir + 'misc.txt', 'r') as f:
        lines = f.readlines()
        if len(lines) == 4:
            cfg.pid = int(lines[3].split()[1])
        else:
            print_warning('Incorrect misc.txt content. Some profiling information may not be available.')

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
        cfg.time_base = float(lines[0]) + cfg.cpu_time_offset
        print_info(cfg,'Time offset applied to timestamp (s):' + str(cfg.cpu_time_offset))
        print_info(cfg,'SOFA global time base (s):' + str(cfg.time_base))

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

    net_traces = []
    cpu_traces = []
    cpu_traces_viz = []
    blk_d_traces = []
    blk_traces = []
    vm_usr_traces = []
    vm_sys_traces = []
    vm_bi_traces = []
    vm_b0_traces = []
    vm_in_traces = []
    vm_cs_traces = []
    vm_wa_traces = []
    vm_st_traces = []
    mpstat_traces = []
    diskstat_traces = []
    tx_traces = []
    rx_traces = []
    strace_traces = []
    pystacks_traces = []
    nvsmi_sm_traces = []
    nvsmi_mem_traces = []
    nvsmi_enc_traces = []
    nvsmi_dec_traces = []
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

    with open('%s/mpstat.txt' % logdir) as f:
        mpstat = np.genfromtxt(logdir+'/mpstat.txt', delimiter=',', invalid_raise=False )
        header = mpstat[0]
        mpstat = mpstat[1:]
        mpstat_list = []
        mpstat_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
        n_cores = int(mpstat[:,1].max() + 1)
        stride = n_cores + 1
        for i in range(len(mpstat)):
            if len(mpstat[i]) < len(header):
                continue
            if i <= stride or mpstat[i,1] == -1:
                continue
            #time, cpu,  user，nice, system, idle, iowait, irq, softirq
            core = mpstat[i,1]
            d_mp = mpstat[i,:] - mpstat[i-stride,:]
            d_mp_total = np.sum(d_mp[2:8])
            if d_mp_total == 0 :
                print_info(cfg, 'No increases in mpstat values')
                continue
            d_mp_usr = d_mp[2] * 100 / float(d_mp_total)
            d_mp_sys = d_mp[4] * 100 / float(d_mp_total)
            d_mp_idl = d_mp[5] * 100 / float(d_mp_total)
            d_mp_iow = d_mp[6] * 100 / float(d_mp_total)
            d_mp_irq = d_mp[7] * 100 / float(d_mp_total)
            cpu_time = (d_mp_total - d_mp[5]) * 0.01
            t_begin = mpstat[i,0]

            if not cfg.absolute_timestamp:
                t_begin = t_begin - cfg.time_base

            deviceId = core
            metric = cpu_time
            event = -1
            copyKind = -1
            payload = -1
            bandwidth = -1
            pkt_src = pkt_dst = -1
            pid = tid = -1
            mpstat_info = 'mpstat_core%d (usr|sys|idl|iow|irq): |%3d|%3d|%3d|%3d|%3d|' % (core, d_mp_usr, d_mp_sys, d_mp_idl, d_mp_iow, d_mp_irq)

            trace_usr = [
                t_begin,
                event,
                metric,
                deviceId,
                copyKind,
                payload,
                bandwidth,
                pkt_src,
                pkt_dst,
                pid,
                tid,
                mpstat_info,
                0]
            
            mpstat_list.append(trace_usr)
            
        mpstat_traces = list_to_csv_and_traces(logdir, mpstat_list, 'mpstat.csv', 'w')

    with open('%s/diskstat.txt' % logdir) as f:
        diskstats = f.readlines()
        diskstat_list = []
        diskstat_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
        tmp_list = []
        for diskstat in diskstats:
            m = diskstat[:-1]
            m = m.split(',')
            tmp_list.append(m)
        devs = list(map(lambda x: x[1], tmp_list))
        n_dev = len(set(devs)) 

        for i in range(len(diskstats)):
            if i < n_dev:
                continue
            m = diskstats[i][:-1]
            m = m.split(',')
            dev = m[1]
            m_last = diskstats[i-n_dev][:-1]
            m_last = m_last.split(',')

            # get sector size
            try:
                f = open('/sys/block/'+dev+'/queue/hw_sector_size')
                s = f.readline()
                s = re.match("\d+", s)
                secsize = int(s.group())
            except:
                pass

            d_read = int(m[2]) - int(m_last[2])
            d_read *= secsize
            d_write = int(m[3]) - int(m_last[3])
            d_write *= secsize
            d_disk_total = d_read + d_write
            d_disk_total *= secsize
            if not d_disk_total:
                continue
            t_begin = float(m[0])

            if not cfg.absolute_timestamp:
                t_begin = t_begin - cfg.time_base     
            
            event = -1
            rw = d_disk_total
            deviceId = -1
            copyKind = -1
            payload = -1
            bandwidth = -1
            pkt_src = -1
            pkt_dst = -1
            pid = -1
            tid = -1
            diskstat_info = 'diskstat_dev:%s (read|write): |%3d|%3d| bytes' % (m[1], d_read, d_write)
            trace = [
                t_begin,
                event,
                rw,
                deviceId,
                copyKind,
                payload,
                bandwidth,
                pkt_src,
                pkt_dst,
                pid,
                tid,
                diskstat_info,
                0]

            diskstat_list.append(trace)
        diskstat_traces = list_to_csv_and_traces(logdir, diskstat_list, 'diskstat.csv', 'w')

    
    #     dev   cpu   sequence  timestamp   pid  event operation start_block+number_of_blocks   process
    # <mjr,mnr>        number
    #     8,0    6        1     0.000000000 31479  A   W 691248304 + 1024 <- (8,5) 188175536
    #     8,0    6        2     0.000001254 31479  Q   W 691248304 + 1024 [dd]
    #     8,0    6        3     0.000003353 31479  G   W 691248304 + 1024 [dd]
    #     8,0    6        4     0.000005004 31479  I   W 691248304 + 1024 [dd]
    #     8,0    6        5     0.000006175 31479  D   W 691248304 + 1024 [dd]
    #     8,0    2        1     0.001041752     0  C   W 691248304 + 1024 [0]
    if cfg.blktrace_device is not None:
        with open('%s/blktrace.txt' % logdir) as f:
            lines = f.readlines()
            print_info(cfg,"Length of blktrace = %d" % len(lines))
            if len(lines) > 0:
                blktrace_d_list = []
                blktrace_list = []
                blktrace_d_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                blktrace_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                record_error_flag = 0

                t = 0
                for i in range(len(lines)):
                    # filter some total calculate information in the below of blktrace.txt file
                    if len(lines[i]) > 50 and "Read" not in lines[i] and "CPU" not in lines[i] and "IO unplugs" not in lines[i]:
                        fields = lines[i].split()
                        blktrace_dev = fields[0]
                        blktrace_cpu = fields[1]
                        blktrace_sequence_number = fields[2]
                        blktrace_timestamp = float(fields[3])
                        blktrace_pid = fields[4]
                        blktrace_event = fields[5]
                        blktrace_operation = fields[6]
                        try: 
                            blktrace_start_block = int(fields[7])
                        except:
                            blktrace_start_block = 0
                            record_error_flag = 1
                            pass
                        # the two column blktrace_block_size and blktrace_process is for future used
                        if len(fields) > 10:
                            blktrace_block_size = fields[9]
                            blktrace_process = fields[10]

                        t_begin = blktrace_timestamp
                        deviceId = cpuid = blktrace_cpu
                        event = blktrace_event
                        copyKind = -1
                        payload = -1
                        bandwidth = -1
                        pkt_src = pkt_dst = -1
                        pid = tid = blktrace_pid
                        name_info = 'starting_block='+str(blktrace_start_block)
                        trace = [
                            t_begin,
                            event,
                            blktrace_start_block,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            name_info,
                            cpuid]

                        if 'D' is event:
                            blktrace_d_list.append(trace)

                        if 'C' is event:
                            for i in range(len(blktrace_d_list)):
                                if i==0:
                                    continue
                                if int(blktrace_d_list[i][2])==int(blktrace_start_block):
                                    time_consume = float(blktrace_timestamp)-float(blktrace_d_list[i][0])
                                    # print('blktrace_d_list[i]:%s'%blktrace_d_list[i])
                                    # print('int(blktrace_timestamp):%f, int(blktrace_d_list[i][0]:%f, time_consume:%f' % (float(blktrace_timestamp), float(blktrace_d_list[i][0]), time_consume))
                                    trace = [
                                        blktrace_d_list[i][0],
                                        event,
                                        float(time_consume),
                                        deviceId,
                                        copyKind,
                                        payload,
                                        bandwidth,
                                        pkt_src,
                                        pkt_dst,
                                        pid,
                                        tid,
                                        name_info,
                                        cpuid]
                                    blktrace_list.append(trace)
                                    blktrace_d_list[i][11] = 'latency=%0.6f' % float(time_consume)

                blk_d_traces = list_to_csv_and_traces(
                    logdir, blktrace_d_list, 'blktrace.csv', 'w')
                blk_traces = list_to_csv_and_traces(
                    logdir, blktrace_list, 'blktrace.csv', 'a')

                if record_error_flag == 1 :
                    print_warning('blktrace maybe record failed!')


    # procs -----------------------memory---------------------- ---swap-- -
    #  r  b         swpd         free         buff        cache   si   so    bi    bo   in   cs  us  sy  id  wa  st
    #  2  0            0    400091552       936896    386150912    0    0     3    18    0    1   5   0  95   0   0
    # ============ Preprocessing VMSTAT Trace ==========================
    with open('%s/vmstat.txt' % logdir) as f:
        lines = f.readlines()
        print_info(cfg,"Length of vmstat_traces = %d" % len(lines))
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

                    if cfg.absolute_timestamp:
                        t_begin = t + cfg.time_base
                    else:
                        t_begin = t
                    
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
                logdir, vm_bi_list, 'vmstat.csv', 'w')
            vm_bo_traces = list_to_csv_and_traces(
                logdir, vm_bo_list, 'vmstat.csv', 'a')
            vm_in_traces = list_to_csv_and_traces(
                logdir, vm_in_list, 'vmstat.csv', 'a')
            vm_cs_traces = list_to_csv_and_traces(
                logdir, vm_cs_list, 'vmstat.csv', 'a')
            vm_wa_traces = list_to_csv_and_traces(
                logdir, vm_wa_list, 'vmstat.csv', 'a')
            vm_st_traces = list_to_csv_and_traces(
                logdir, vm_st_list, 'vmstat.csv', 'a')
            vm_usr_traces = list_to_csv_and_traces(
                logdir, vm_usr_list, 'vmstat.csv', 'a')
            vm_sys_traces = list_to_csv_and_traces(
                logdir, vm_sys_list, 'vmstat.csv', 'a')

    
    # timestamp, name, index, utilization.gpu [%], utilization.memory [%]
    # 2019/05/16 16:49:04.650, GeForce 940MX, 0, 0 %, 0 %
    if os.path.isfile('%s/nvsmi_query.txt' % logdir):
        with open('%s/nvsmi_query.txt' % logdir) as f:
            next(f)
            lines = f.readlines()
            nvsmi_query_has_data = True
            for line in lines:
                if line.find('failed') != -1 or line.find('Failed') != -1:
                    nvsmi_query_has_data = False
                    print_warning('No nvsmi query data.')
                    break
            if nvsmi_query_has_data:
                print_info(cfg,"Length of nvsmi_query_traces = %d" % len(lines))
                nvsmi_sm_list = []
                nvsmi_mem_list = []
                nvsmi_sm_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                nvsmi_mem_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())

                for i in range(len(lines)):                 
                    fields = lines[i].split(',')
                    nv_time = fields[0]
                    nv_time = datetime.datetime.strptime(nv_time, '%Y/%m/%d %H:%M:%S.%f').timestamp()
                    nvsmi_id = int(fields[2])
                    nvsmi_sm = int(fields[3][:-2])
                    nvsmi_mem = int(fields[4][:-2])
                    
                    # nvtime 
                    t_begin = nv_time
                    if not cfg.absolute_timestamp:
                        t_begin = t_begin - cfg.time_base
     
                    deviceId = cpuid = nvsmi_id
                    event = -1
                    copyKind = -1
                    payload = -1
                    bandwidth = -1
                    pkt_src = pkt_dst = -1
                    pid = tid = -1
                    sm_info = "GPUID_sm=%d_%d" % (nvsmi_id, nvsmi_sm)
                    mem_info = "GPUID_mem=%d_%d" % (nvsmi_id, nvsmi_mem)

                    trace = [
                        t_begin,
                        0,
                        nvsmi_sm,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        sm_info,
                        cpuid]
                
                    nvsmi_sm_list.append(trace)

                    trace = [
                        t_begin,
                        1,
                        nvsmi_mem,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        mem_info,
                        cpuid]
                    
                    nvsmi_mem_list.append(trace)
                    
                if len(nvsmi_sm_list)>1:
                    nvsmi_sm_traces = list_to_csv_and_traces(logdir, nvsmi_sm_list, 'nvsmi_trace.csv', 'w')
                    nvsmi_mem_traces = list_to_csv_and_traces(logdir, nvsmi_mem_list, 'nvsmi_trace.csv', 'a')


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
                print_info(cfg,"Length of nvsmi_traces = %d" % len(lines))
                nvsmi_enc_list = []
                nvsmi_dec_list = []
                nvsmi_enc_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                nvsmi_dec_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                t = 0
                for i in range(len(lines)):
                    if lines[i].find('gpu') == -1 and lines[i].find('Idx') == -1:
                        fields = lines[i].split()
                        if len(fields) < 5:
                            continue
                        nvsmi_id = int(fields[0])
                        if fields[3] == '-':
                            nvsmi_enc = int(0) 
                        else:
                            nvsmi_enc = int(fields[3])
                        if fields[4] == '-':
                            nvsmi_dec = int(0) 
                        else:
                            nvsmi_dec = int(fields[4]) 

                        if cfg.absolute_timestamp: 
                            t_begin = t + cfg.time_base
                        else:
                            t_begin = t
                        deviceId = cpuid = nvsmi_id
                        event = -1
                        copyKind = -1
                        payload = -1
                        bandwidth = -1
                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        enc_info = "GPUID_enc=%d_%d" % (nvsmi_id, nvsmi_enc)
                        dec_info = "GPUID_dec=%d_%d" % (nvsmi_id, nvsmi_dec)

                        trace = [
                            t_begin,
                            2,
                            nvsmi_enc,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            enc_info,
                            cpuid]
                        if t > 3 :
                            nvsmi_enc_list.append(trace)

                        trace = [
                            t_begin,
                            3,
                            nvsmi_dec,
                            deviceId,
                            copyKind,
                            payload,
                            bandwidth,
                            pkt_src,
                            pkt_dst,
                            pid,
                            tid,
                            dec_info,
                            cpuid]
                        if t > 3 :
                            nvsmi_dec_list.append(trace)
                        
                        if nvsmi_id == 0:
                            t = t + 1
                if len(nvsmi_enc_list)>1:
                    nvsmi_enc_traces = list_to_csv_and_traces(logdir, nvsmi_enc_list, 'nvsmi_trace.csv', 'a')
                    nvsmi_dec_traces = list_to_csv_and_traces(logdir, nvsmi_dec_list, 'nvsmi_trace.csv', 'a')
                else:
                    print_warning("Program exectution time is fewer than 3 seconds, so nvsmi trace analysis will not be displayed.")

    # ============ Preprocessing Network Trace ==========================
    
    if os.path.isfile('%s/sofa.pcap' % logdir):
        with open(logdir + 'net.tmp', 'w') as f:
            subprocess.check_call(
                ["tcpdump", "-q", "-n", "-tt", "-r",
                "%s/sofa.pcap"%logdir ], stdout=f, stderr=subprocess.DEVNULL)
        with open(logdir + 'net.tmp') as f:
            packets = lines = f.readlines()
            print_info(cfg,"Length of net_traces = %d" % len(packets))
            if packets:
                with mp.Pool(processes=cpu_count) as pool:
                    res = pool.map(
                        partial(
                            net_trace_read,
                            cfg=cfg,
                            t_offset=0),
                        packets)
                res_viz = list_downsample(res, cfg.plot_ratio)
                net_traces = pd.DataFrame(res_viz)
                net_traces.columns = sofa_fieldnames
                net_traces.to_csv(
                    logdir + 'nettrace.csv',
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.6f')

                # ============ Apply for Network filter =====================
                if cfg.net_filters:
                    filtered_net_groups = []
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
    # ============ Preprocessing Network Bandwidth Trace ============
    with open('%s/netstat.txt' % logdir) as f:
        lines = f.readlines()
        tmp_time = float(lines[0].split(',')[0])
        tmp_tx = int(lines[0].split(',')[1])
        tmp_rx = int(lines[0].split(',')[2])
        all_time = []
        all_tx = []
        all_rx = []
        tx_list = []
        rx_list = []
        bandwidth_result = pd.DataFrame([], columns=['time', 'tx_bandwidth', 'rx_bandwidth'])

        for line in lines[1:]:
            time = float(line.split(',')[0])
            tx = int(line.split(',')[1])
            rx = int(line.split(',')[2])
            tx_bandwidth = (tx - tmp_tx) / (time - tmp_time) 
            rx_bandwidth = (rx - tmp_rx) / (time - tmp_time)
            
            #sofa_fieldnames = [
            #    "timestamp",  # 0
            #    "event",  # 1
            #    "duration",  # 2
            #    "deviceId",  # 3
            #    "copyKind",  # 4
            #    "payload",  # 5
            #    "bandwidth",  # 6
            #    "pkt_src",  # 7
            #    "pkt_dst",  # 8
            #    "pid",  # 9
            #    "tid",  # 10
            #    "name",  # 11
            #    "category"] # 12
            
            t_begin = time
            if not cfg.absolute_timestamp:
                t_begin = t_begin - cfg.time_base

            trace = [ 
                t_begin, # timestamp
                0, # event
                -1,
                -1,
                -1,
                -1,
                tx_bandwidth, # tx bandwidth
                -1,
                -1,
                -1,
                -1,
                "network_bandwidth_tx(bytes):%d" % tx_bandwidth,
                0
                ]
            tx_list.append(trace)

            trace = [
                t_begin, # timestamp
                1, # event
                -1,
                -1,
                -1,
                -1,
                rx_bandwidth, # rx bandwidth
                -1,
                -1,
                -1,
                -1,
                "network_bandwidth_rx(bytes):%d" % rx_bandwidth,
                0
                ]
            rx_list.append(trace)

            # for visualize
            all_time.append(time)
            all_tx.append(tx_bandwidth)
            all_rx.append(rx_bandwidth)
            
            # for pandas
            result = [time, tx_bandwidth, rx_bandwidth]
            tmp_bandwidth_result = pd.DataFrame([result], columns=['time', 'tx_bandwidth', 'rx_bandwidth'])
            bandwidth_result = pd.concat([bandwidth_result, tmp_bandwidth_result], ignore_index=True)
            
            # prepare for next round loop        
            tmp_time = time
            tmp_tx = tx
            tmp_rx = rx    
        tx_traces = pd.DataFrame(tx_list, columns = sofa_fieldnames)
        tx_traces.to_csv(
                    logdir + 'netstat.csv',
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.6f')
        rx_traces = pd.DataFrame(rx_list, columns = sofa_fieldnames)
        rx_traces.to_csv(
                    logdir + 'netstat.csv',
                    mode='a',
                    header=False,
                    index=False,
                    float_format='%.6f')
        
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
            print_info(cfg,'NO MEMSET')
        try:
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMCPY',engine)).iloc[0]['start'])
        except BaseException:
            print_info(cfg,'NO MEMCPY')
        try:
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL',engine)).iloc[0]['start'])
        except BaseException:
            print_info(cfg,'NO CONCURRENT KERNEL')
        try:
            t_glb_gpu_bases.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_KERNEL',engine)).iloc[0]['start'])
        except BaseException:
            print_info(cfg,'NO KERNEL')
        
        if len(t_glb_gpu_bases) > 0:
            t_glb_gpu_base = sorted(t_glb_gpu_bases)[0]*1.0/1e+9
        else:
           print_warning("There is no data in tables of NVVP file.")

        print_info(cfg,"Timestamp of the first GPU trace = " + str(t_glb_gpu_base))

        print_progress("Read " + nvvp_filename + " by nvprof -- end")
        num_cudaproc = num_cudaproc + 1
        with open(logdir + 'gputrace.tmp') as f:
            records = f.readlines()
            # print(records[1])

            if len(records) > 0 and records[1].split(',')[0] == '"Start"':
                indices = records[1].replace(
                    '"', '').replace(
                    '\n', '').split(',')
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
                    print_info(cfg,"The payload unit in gputrace.tmp was not recognized!")
                    sys.exit(1)

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
                print_info(cfg,"Length of gpu_traces = %d" % len(records))
                t_base = float(records[0].split(',')[0])
                with mp.Pool(processes=cpu_count) as pool:
                    res = pool.map(
                        partial(
                            gpu_trace_read,
                            cfg=cfg,
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
                print_info(cfg,'NO RUNTIME')

            if len(t_glb_gpu_bases) > 0:
                t_glb_gpu_base = sorted(t_glb_gpu_bases)[0]*1.0/1e+9
            else:
               print_warning("There is no data in tables of NVVP file.")

            print_info(cfg,"Timestamp of the first CUDA API trace = " + str(t_glb_gpu_base))

            print_progress("Read " + nvvp_filename + " by nvprof -- end")
            num_cudaproc = num_cudaproc + 1
            with open(logdir + 'cuda_api_trace.tmp') as f:
                records = f.readlines()
                # print(records[1])
                if len(records) > 0 and records[1].split(',')[0] == '"Start"':
                    indices = records[1].replace(
                        '"', '').replace(
                        '\n', '').split(',')

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
                    print_info(cfg,"Length of cuda_api_traces = %d" % len(records))

                    #TODO: Apply parallel search to speed up
                    t_base = float(records[0].split(',')[0])
                    if len(records[0].split(',')) == 4:
                        for record in records:
                            if int(record.split(',')[3]) == first_corid:
                                t_base = float(record.split(',')[0])
                                print_info(cfg,'First Correlation_ID ' + str(first_corid) + ' is found in cuda_api_trace.tmp')
                                print_info(cfg,'First API trace timestamp is ' + str(t_base))
                                break

                    with mp.Pool(processes=cpu_count) as pool:
                        res = pool.map(
                            partial(
                                cuda_api_trace_read,
                                cfg=cfg,
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
        print_info(cfg,'perf_events_used: %s' % (cfg.perf_events))
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
            print_info(cfg,'NO MEMSET')

        try:
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_MEMCPY',engine)).iloc[-1]['start'])
        except BaseException:
            print_info(cfg,'NO MEMCPY')

        try:
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL',engine)).iloc[-1]['start'])
        except BaseException:
            print_info(cfg,'NO CONCURRENT KERNEL')

        try:
            last_nvvp_tss.append( (pd.read_sql_table('CUPTI_ACTIVITY_KIND_KERNEL',engine)).iloc[-1]['start'])
        except BaseException:
            print_info(cfg,'NO KERNEL')
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
            print_info(cfg,"Length of cpu_traces = %d" % len(samples))
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

    # STRACE Preprocessing
    #CASE1: strace: Process 8361 attached
    #CASE2: 1550311783.488821 mmap(NULL, 262144, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0) = 0x7f501f910000 <0.000012>
    #CASE3: [pid  8372] 1550311820.640979 +++ exited with 0 +++
    total_strace_duration = 0
    filter_keys = []
    #filter_keys.append('futex')
    filter_keys.append('resume')
    filter_keys.append('nanosleep')
    filter_keys.append('clock_gettime')
    filter_keys.append('brk')
    filter_keys.append('stat')
    filter_keys.append('close')
    filter_keys.append('exited')
    filter_keys.append('access')
    filter_keys.append('lseek')
    filter_keys.append('getrusage')
    filter_keys.append('exited')
    if os.path.isfile('%s/strace.txt' % logdir):
        with open('%s/strace.txt' % logdir) as f:
            lines = f.readlines()
            print_info(cfg,"Length of straces = %d" % len(lines))
            if len(lines) > 1:
                strace_list = []
                strace_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())               
                for i in range(len(lines)):
                    if i % cfg.plot_ratio > 0:
                        continue
                    pid = cfg.pid
                    tid = 0
                    
                    b_skip = False
                    for key in filter_keys:
                        if lines[i].find(key) != -1:
                            b_skip = True
                    if b_skip:
                        continue

                    fields = lines[i].split()
                    if fields[0].find('pid') != -1 :
                        tid = int(fields[1].split(']')[0])
                        t_begin = float(fields[2])
                        strace_info = ''.join(fields[3:-3])
                    else:
                        tid = pid
                        t_begin = float(fields[1])
                        strace_info = ''.join(fields[1:-3])
                    
                    if not cfg.absolute_timestamp:
                        t_begin = t_begin - cfg.time_base
                    
                    #strace_info = strace_info.split('(')[0] 
                    try:
                        duration = float(fields[-1].split('<')[1].split('>')[0]) 
                    except:
                        duration = 0 
                    total_strace_duration = total_strace_duration + duration

                    if duration < cfg.strace_min_time:
                        continue

                    deviceId = -1
                    event = -1
                    copyKind = -1
                    payload = -1
                    bandwidth = -1
                    pkt_src = pkt_dst = -1
                    trace = [
                        t_begin,
                        event,
                        duration,
                        deviceId,
                        copyKind,
                        payload,
                        bandwidth,
                        pkt_src,
                        pkt_dst,
                        pid,
                        tid,
                        strace_info,
                        cpuid]
                    strace_list.append(trace)
                
                print_info(cfg, 'strace.txt reading is done.')
                if len(strace_list)>1:
                    strace_traces = list_to_csv_and_traces(logdir, strace_list, 'strace.csv', 'w')
    print_info(cfg,'Total strace duration: %.3lf' % total_strace_duration)


    # Pystacks Preprocessing

    def parse_pystacks(filepath, ignore_idle=False):
        ret = {}
        with open(filepath, 'r') as f:
            for ts, fs in itertools.zip_longest(*[f] * 2):
                fs = fs.replace('\n', '').replace(';', '<br>')
                if ignore_idle:
                    if fs.find('idle') != -1:
                        continue
                    ret[int(ts) / 10 ** 6] = fs
        duration = {}
        prev = None
        for k, val in ret.items():
            if prev is None:
                prev = k
                continue
            duration[prev] = k - prev
            prev = k
        del ret[max(ret.keys())]

        return ret, duration

    if os.path.isfile('{}/pystacks.txt'.format(logdir)):
        fstack, dur = parse_pystacks('{}/pystacks.txt'.format(logdir), ignore_idle=True)
        pystacks_list = []

        if fstack:
            for key, info in fstack.items():
                deviceId = -1
                event = -1
                copyKind = -1
                payload = -1
                bandwidth = -1
                pkt_src = pkt_dst = -1
                pid = tid = -1
                t_begin = key if cfg.absolute_timestamp else key - cfg.time_base
                trace = [
                    t_begin,
                    event,
                    float(dur[key]),
                    deviceId,
                    copyKind,
                    payload,
                    bandwidth,
                    pkt_src,
                    pkt_dst,
                    pid,
                    tid,
                    info,
                    cpuid
                ]
                pystacks_list.append(trace)
        if pystacks_list:
            pystacks_traces = list_to_csv_and_traces(logdir, pystacks_list, 'pystacks.csv', 'w')    

    
    
    # Time synchronization among BIOS Time (e.g. used by perf)  and NTP Time (e.g. NVPROF, tcpdump, etc.)
    if perf_timebase_unix == 0:
        with open(logdir + 'perf_timebase.txt') as f:
            lines = f.readlines()
            if len(lines) > 3:
                perf_timebase_uptime = float(lines[-2].split()[2].split(':')[0])
                perf_timebase_unix = float(lines[-1].split()[0])
            else:
                print_warning('Recorded progrom is too short.')
                sys.exit(1)
    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        print_info(cfg,"Length of cpu_traces = %d" % len(samples))
        if len(samples) > 0:
            with mp.Pool(processes=cpu_count) as pool:
                res = pool.map(
                    partial(
                        cpu_trace_read,
                        cfg = cfg,
                        t_offset = perf_timebase_unix - perf_timebase_uptime,
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
    # ### hierarchical swarm generation
    try:
        swarm_groups = []
        swarm_stats = []
        swarm_groups, swarm_stats = sofa_hsg(cfg, swarm_groups, swarm_stats, perf_timebase_unix - perf_timebase_uptime, cpu_mhz_xp, cpu_mhz_fp)
    except TypeError:
        print_warning('HSG returned a None object to swarm_groups, check if sofalog/perf.data can be accessed.')
        pass 
    #=== Intel PCM Trace =======#
    ### Skt,PCIeRdCur,RFO,CRd,DRd,ItoM,PRd,WiL,PCIe Rd (B),PCIe Wr (B)
    ### 0,0,852,0,0,48,0,0,54528,57600
    ### 1,0,600,0,0,0,0,0,38400,38400
    if cfg.enable_pcm and os.path.isfile('%s/pcm_pcie.csv' % logdir):
        with open( logdir + '/pcm_pcie.csv' ) as f:
            lines = f.readlines()
            print_info(cfg,"Length of pcm_pcie_traces = %d" % len(lines))
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

                        if not cfg.absolute_timestamp:
                            t_begin = t_begin - cfg.time_base

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
            print_info(cfg,"Length of pcm_memory_traces = %d" % len(lines))
            if len(lines) > 0:
                pcm_memory_list = []
                pcm_memory_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                for line in lines:
                    if line.find('Skt') == -1:
                        fields = line.split(',')

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

                        if not cfg.absolute_timestamp:
                            t_begin = t_begin - cfg.time_base

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
    

    if len(swarm_groups) > 0 :
        traces = sofa_hsg_to_sofatrace(cfg, swarm_groups, traces) # append data of hsg function

    sofatrace = SOFATrace()
    sofatrace.name = 'blktrace_starting_block'
    sofatrace.title = 'BLKTRACE_STARTING_BLOCK'
    sofatrace.color = 'Green'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = blk_d_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'blktrace_time'
    sofatrace.title = 'BLKTRACE_TIME'
    sofatrace.color = 'DodgerBlue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = blk_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'vmstat_cs'
    sofatrace.title = 'VMSTAT_CS'
    sofatrace.color = 'Pink'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = vm_cs_traces
    traces.append(sofatrace)

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

    if cfg.enable_mpstat:
        sofatrace = SOFATrace()
        sofatrace.name = 'mpstat_usr'
        sofatrace.title = 'MPSTAT_USR'
        sofatrace.color = 'Cyan'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = mpstat_traces
        traces.append(sofatrace)

    if cfg.enable_diskstat:
        sofatrace = SOFATrace()
        sofatrace.name = 'diskstat'
        sofatrace.title = 'DISK_USAGE'
        sofatrace.color = 'GreenYellow'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = diskstat_traces
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
    sofatrace.name = 'strace'
    sofatrace.title = 'STRACE.'
    sofatrace.color = 'DarkSlateGray'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = strace_traces
    traces.append(sofatrace)
    
    sofatrace = SOFATrace()
    sofatrace.name = 'pystacks'
    sofatrace.title = 'Python-stacks.'
    sofatrace.color = 'Tomato'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = pystacks_traces
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

    if cfg.enable_encode_decode:
        sofatrace = SOFATrace()
        sofatrace.name = 'nvsmi_enc'
        sofatrace.title = 'GPU_ENC_Util.'
        sofatrace.color = 'rgba(255, 215, 0, 0.8)' #Gold
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = nvsmi_enc_traces
        traces.append(sofatrace)

        sofatrace = SOFATrace()
        sofatrace.name = 'nvsmi_dec'
        sofatrace.title = 'GPU_DEC_Util.'
        sofatrace.color = 'rgba(218, 165, 32, 0.8)' #GoldenRod
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = nvsmi_dec_traces
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
    sofatrace.name = 'tx_bandwidth'
    sofatrace.title = 'tx Bandwidth'
    sofatrace.color = 'rgba(135,206,250,0.8)' # LightSkyBlue
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'bandwidth'
    sofatrace.data = tx_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'rx_bandwidth'
    sofatrace.title = 'rx Bandwidth'
    sofatrace.color = 'rgba(25,25,112,0.8)' # MidnightBlue
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'bandwidth'
    sofatrace.data = rx_traces
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
