#!/usr/bin/python
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


def list_downsample(list_in, plot_ratio):
    new_list = []
    for i in xrange(len(list_in)):
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


def cpu_trace_read(sample, t_offset):
    fields = sample.split()

    if re.match('\[\d+\]', fields[1]) is not None:
        time = float(fields[2].split(':')[0])
        func_name = fields[4].replace('-', '_') + fields[6]
        cycles = float(fields[3])
        event = np.log(1.0 * int("0x01" + fields[5], 16))
    else:
        time = float(fields[1].split(':')[0])
        func_name = fields[3].replace('-', '_') + fields[5]
        cycles = float(fields[2])
        event = np.log(1.0 * int("0x01" + fields[4], 16))

    t_begin = time + t_offset
    t_end = time + t_offset
    trace = [t_begin,
             event,  # % 1000000
             cycles / 1e9,
             -1,
             -1,
             0,
             0,
             -1,
             -1,
             int(fields[0].split('/')[0]),
             int(fields[0].split('/')[1]),
             func_name,
             0]
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
    for i in xrange(4):
        pkt_src = pkt_src + \
            int(packet.split()[2].split('.')[i]) * np.power(1000, 3 - i)
        pkt_dst = pkt_dst + \
            int(packet.split()[4].split('.')[i]) * np.power(1000, 3 - i)
    trace = [	t_begin,
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


def gpu_trace_read(
        record,
        indices,
        n_cudaproc,
        ts_rescale,
        dt_rescale,
        t_offset):
    values = record.replace('"', '').split(',')
    kernel_name = values[indices.index('Name')]
    # print("kernel name = %s" % kernel_name)
    time = float(values[indices.index('Start')]) / ts_rescale + t_offset
    duration = float(values[indices.index('Duration')]) / dt_rescale
    t_begin = time
    t_end = time + duration
    try:
        payload = int(float(values[indices.index('Size')]) * 1024 * 1024)
    except BaseException:
        payload = 0

    try:
        bandwidth = float(values[indices.index('Throughput')])
    except BaseException:
        bandwidth = 1e-6

    pid = n_cudaproc

    try:
        deviceId = int(float(values[indices.index('Context')]))
    except BaseException:
        deviceId = -1

    try:
        tid = streamId = int(float(values[indices.index('Stream')]))
    except BaseException:
        tid = streamId = -1

    pkt_src = pkt_dst = copyKind = 0
    if kernel_name.find('HtoD') != -1:
        copyKind = 1
        pkt_src = 0
        pkt_dst = deviceId
        kernel_name = "gpu%d_copyKind_%d_%dB" % (deviceId, copyKind, payload)
    elif kernel_name.find('DtoH') != -1:
        copyKind = 2
        pkt_src = deviceId
        pkt_dst = 0
        kernel_name = "gpu%d_copyKind_%d_%dB" % (deviceId, copyKind, payload)
    elif kernel_name.find('DtoD') != -1:
        copyKind = 8
        pkt_src = deviceId
        pkt_dst = deviceId
        kernel_name = "gpu%d_copyKind_%d_%dB" % (deviceId, copyKind, payload)
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

        kernel_name = "gpu%d_copyKind_%d_%dB" % (deviceId, copyKind, payload)
    else:
        copyKind = 0

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
    print(bcolors.WARNING + "[WARNING] " + content + bcolors.ENDC)


def print_info(content):
    print(bcolors.OKGREEN + "[INFO] " + content + bcolors.ENDC)


def print_progress(content):
    print(bcolors.OKBLUE + "[PROGRESS] " + content + bcolors.ENDC)


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


sofa_fieldnames = [
    'timestamp',  # 0
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
    "category"]  # 12


def sofa_preprocess(logdir, cfg):
    t_glb_base = 0
    t_glb_net_base = 0
    t_glb_gpu_base = 0

    with open('%s/perf.script' % logdir, 'w') as logfile:
        subprocess.call(['perf',
                         'script',
                         '--kallsym',
                         '%s/kallsym' % logdir,
                         '-i',
                         '%s/perf.data' % logdir,
                         '-F',
                         'time,pid,tid,ip,sym,period,event'],
                        stdout=logfile)

    # sys.stdout.flush()
    with open(logdir + 'sofa_time.txt') as f:
        t_glb_base = float(f.readlines()[0])
        t_glb_net_base = t_glb_base
        t_glb_gpu_base = t_glb_base
        print t_glb_base
        print t_glb_net_base

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
    gpu_traces = []
    gpu_traces_viz = []
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

    # ============ Preprocessing CPU Trace ==========================
    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        print_info("Length of cpu_traces = %d" % len(samples))
        if len(samples) > 0:
            pool = mp.Pool(processes=cpu_count)
            t_base = float((samples[0].split())[1].split(':')[0])
            res = pool.map(
                partial(
                    cpu_trace_read,
                    t_offset=t_glb_base -
                    t_base),
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
            t_base = t = 0
            for i in xrange(len(lines)):
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

                    t_begin = t - t_base + t_glb_base
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
            print_info("Length of nvsmi_traces = %d" % len(lines))
            if len(lines) > 0:
                nvsmi_sm_list = []
                nvsmi_mem_list = []
                nvsmi_sm_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                nvsmi_mem_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
                t_base = t = 0
                for i in xrange(len(lines)):
                    if lines[i].find('gpu') == -1 \
                            and lines[i].find('Idx') == -1:
                        fields = lines[i].split()
                        if len(fields) < 5:
                            continue
                        nvsmi_id = int(fields[0])
                        nvsmi_sm = float(fields[1]) + 1e-5
                        nvsmi_mem = float(fields[2]) + 1e-5

                        t_begin = t - t_base + t_glb_base
                        deviceId = cpuid = nvsmi_id
                        event = -1
                        copyKind = -1
                        payload = -1
                        bandwidth = -1
                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        nvsmi_info = "GPUID.sm.mem=%d_%lf_%lf" % (
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
                        nvsmi_mem_list.append(trace)
                        if nvsmi_id == 0:
                            t = t + 1
                nvsmi_sm_traces = list_to_csv_and_traces(logdir, nvsmi_sm_list, 'nvsmi_trace.csv', 'w')
                nvsmi_mem_traces = list_to_csv_and_traces(logdir, nvsmi_mem_list, 'nvsmi_trace.csv', 'a')

    t_first_nv = sys.float_info.max
    for i in xrange(len(cpu_traces)):
        if re.search('_nv\d+rm', cpu_traces.iat[i,11]) is not None and float(cpu_traces.iat[i,0]) < t_first_nv:
            t_first_nv = float(cpu_traces.iat[i, 0])

    if t_first_nv == sys.float_info.max:
        print_warning("'_nv*rm' was not found.")
        t_first_nv = t_glb_base
    print("t_first_nv: %lf" % (t_first_nv))
    # Apply filters for cpu traces
    
    filtered_groups = []
    color_of_filtered_group = []
    if len(cpu_traces) > 0:
        df_grouped = cpu_traces.groupby('name')
        for filter in cfg.cpu_filters:
            group = cpu_traces[cpu_traces['name'].str.contains(
                filter.keyword)]
            filtered_groups.append({'group': group,
                                    'color': filter.color,
                                    'keyword': filter.keyword})

    # ============ Preprocessing Network Trace ==========================
    os.system(
        "tcpdump -q -n -tt -r " +
        logdir +
        "sofa.pcap" +
        " > " +
        logdir +
        "net.tmp")
    with open(logdir + 'net.tmp') as f:
        packets = lines = f.readlines()
        print_info("Length of net_traces = %d" % len(packets))
        if len(packets) > 0:
            t_base = float(lines[0].split()[0])
            pool = mp.Pool(processes=cpu_count)
            res = pool.map(
                partial(
                    net_trace_read,
                    t_offset=t_glb_net_base -
                    t_base),
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

    # ============ Preprocessing GPU Trace ==========================
    num_cudaproc = 0
    filtered_gpu_groups = []
    indices = []
    for nvvp_filename in glob.glob(logdir + "gputrace*[0-9].nvvp"):
        print_progress("Read " + nvvp_filename + " by nvprof -- begin")
        os.system(
            "nvprof --csv --print-gpu-trace -i " +
            nvvp_filename +
            " 2> " +
            logdir +
            "gputrace.tmp")

        # TODO: align cpu time and gpu time
        os.system(
            "nvprof --print-api-trace -i " +
            nvvp_filename +
            " 2> " +
            logdir +
            "nvapi.txt")
        with open(logdir + 'nvapi.txt') as f:
            lines = f.readlines()
            t_api_offset = 0.0
            for line in lines:
                if line.find('cudaMemcpy') != -1:
                    ts = line.split()[0]
                    if ts.find('ms') != -1:
                        t_api_offset = int(
                            re.search(r'\d+', ts).group()) * 1e-3
                    elif ts.find('us') != -1:
                        t_api_offset = int(
                            re.search(r'\d+', ts).group()) * 1e-6
                    else:
                        t_api_offset = int(re.search(r'\d+', ts).group()) * 1.0
                    break
            print('t_api_offset = %lf' % t_api_offset)
            print('cfg.gpu_time_offset = %lf' % (cfg.gpu_time_offset * 1e-3))
            t_glb_gpu_base = t_api_offset + t_first_nv + cfg.gpu_time_offset * 1e-3
        print t_glb_gpu_base

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
                t_offset = t_glb_gpu_base - t_base
                pool = mp.Pool(processes=cpu_count)
                res = pool.map(
                    partial(
                        gpu_trace_read,
                        indices=indices,
                        ts_rescale=ts_rescale,
                        dt_rescale=dt_rescale,
                        n_cudaproc=num_cudaproc,
                        t_offset=t_glb_gpu_base -
                        t_base),
                    records)
                gpu_traces = pd.DataFrame(res)
                gpu_traces.columns = sofa_fieldnames
                gpu_traces.to_csv(
                    logdir + 'gputrace.csv',
                    mode='w',
                    header=True,
                    index=False,
                    float_format='%.6f')
                res_viz = list_downsample(res, cfg.plot_ratio)
                gpu_traces_viz = pd.DataFrame(res_viz)
                gpu_traces_viz.columns = sofa_fieldnames

                # Apply filters for GPU traces
                df_grouped = gpu_traces.groupby('name')
                color_of_filtered_group = []
                for filter in cfg.gpu_filters:
                    group = gpu_traces[gpu_traces['name'].str.contains(
                        filter.keyword)]
                    filtered_gpu_groups.append({'group': group,
                                                'color': filter.color,
                                                'keyword': filter.keyword})
            else:
                print_warning(
                    "gputrace existed, but no kernel traces were recorded.")

                os.system('cat %s/gputrace.tmp' % logdir)
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
        sofatrace.title = 'keyword_' + sofatrace.name
        sofatrace.color = filtered_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_group['group'].copy()
        traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'vmstat_usr'
    sofatrace.title = 'VMSTAT_USR'
    sofatrace.color = 'Magenta'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = vm_usr_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'vmstat_sys'
    sofatrace.title = 'VMSTAT_SYS'
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
    sofatrace.name = 'net_trace'
    sofatrace.title = 'NET'
    sofatrace.color = 'blue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = net_traces
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
        sofatrace.title = 'keyword_' + sofatrace.name
        sofatrace.color = filtered_gpu_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_gpu_group['group'].copy()
        traces.append(sofatrace)

    traces_to_json(traces, logdir + 'report.js', cfg)
    print_progress(
        "Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- end")
