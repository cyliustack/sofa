#!/usr/bin/env python3
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
from sofa_aisi import *
import networkx as nx
import re 


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


def overlap(pa, pb, pc, pd):
    if pb - pc >= 0 and pd - pa >= 0:
        return min(pb, pd) - max(pa, pc)
    else:
        return 0

def partial_sum(df):
    psum = 0


# print_format_table()
cktable = {-1: "NON", 0: "KER", 1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
ckindex = [1, 2, 8, 10]


def comm_profile(cfg, df_gpu):
    total_traffic = 0.0
    total_h2d_traffic = 0.0
    total_d2h_traffic = 0.0
    total_p2p_traffic = 0.0
    total_memcopy_time = 0.0

    # sofa_fieldnames = [
    #    'timestamp',
    #    "event",
    #    "duration",
    #    "deviceId",
    #    "copyKind",
    #    "payload",
    #    "bandwidth",
    #    "pkt_src",
    #    "pkt_dst",
    #    "pid",
    #    "tid",
    #    "name",
    #    "category"]
    n_gpus = 0
    for i in range(len(df_gpu)):
        if df_gpu.iat[i, 3] > n_gpus:
            n_gpus = int(df_gpu.iat[i, 3])

    if n_gpus == 0:
        print_warning("No GPU communication traces are collected.")
        return

    print_title("Data Traffic for each CopyKind (MB)")
    data_copyKind = grouped_df = df_gpu.groupby("copyKind")["payload"]
    for key, item in grouped_df:
        print((
            "[%s]: %lf" %
            (cktable[key], grouped_df.get_group(key).sum() / 1000000.0)))
        if int(key) == 1:
            total_h2d_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 2:
            total_d2h_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 10:
            total_p2p_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) != 8:
            total_traffic = total_traffic + \
                grouped_df.get_group(key).sum() / 1000000.0
    print(("Total traffic: %.2lf" % total_traffic))

    print_title("Data Communication Time for each CopyKind (s)")
    durations_copyKind = grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        print(("[%s]: %lf" % (cktable[key], grouped_df.get_group(key).sum())))
        if key == 0:
            total_kernel_time = grouped_df.get_group(key).sum()
        else:
            total_memcopy_time = total_memcopy_time + \
                grouped_df.get_group(key).sum()

    bw = (data_copyKind.sum() / 1000000) / durations_copyKind.sum() / 1000
    bw_h2d = bw_d2h = bw_p2p = avg_bw = 1e-10

    for i in range(len(bw)):
        key = list(bw.keys())[i]
        if cktable[key] == 'H2D' or cktable[key] == 'D2H' or cktable[key] == 'D2D' or cktable[key] == 'P2P': 
            print(("Averaged Achieved %s Unidirectional Bandwidth: %.1f (GB/s)" % (cktable[key], bw.iloc[i])))
        else:
            continue

    print_title("Summary of Comm.")
    print(("MeasuredTotalTraffic : %lf (MB)" % total_traffic))
    print(("MeasuredTotalH2DTraffic : %lf (MB)" % total_h2d_traffic))
    print(("MeasuredTotalD2HTraffic : %lf (MB)" % total_d2h_traffic))
    print(("MeasuredTotalP2PTraffic : %lf (MB)" % total_p2p_traffic))

    accum = np.zeros((1 + n_gpus, 1 + n_gpus))
    accum_count = np.zeros((1 + n_gpus, 1 + n_gpus))

    # TODO: Parallelize payload accumulatoin
    #print("df length: %d" % len(df_gpu))
    #cpu_count = mp.cpu_count()
    #pool = mp.Pool(processes=cpu_count)
    #res_accum = pool.map( partial(payload_sum), df_gpu)

    for i in range(len(df_gpu)):
        if df_gpu.iat[i, 4] == 0 or df_gpu.iat[i, 4] == 8:
            continue
        src = df_gpu.iat[i, 7]
        dst = df_gpu.iat[i, 8]
        payload = df_gpu.iat[i, 5]
        accum[src][dst] = float(accum[src][dst] + payload)
        accum_count[src][dst] = int(accum_count[src][dst] + 1)

    print("Traffic Matrix (log10(B)):")
    row_str = "\tHOST\t"
    for i in range(1, accum.shape[1]):
        row_str = row_str + "GPU%d" % i + "\t"
    print(row_str)
    for i in range(accum.shape[0]):
        if i == 0:
            row_str = "HOST\t"
        else:
            row_str = "GPU%d\t" % i
        
        for j in range(accum.shape[1]):
            row_str = row_str + "%d" % (int(np.log10(1 + accum[i][j]))) + "\t"
        print(row_str)

    print("Traffic Matrix (MB):")
    row_str = "\tHOST\t"
    for i in range(1, accum.shape[1]):
        row_str = row_str + "GPU%d" % i + "\t"
    print(row_str)
    for i in range(accum.shape[0]):
        if i == 0:
            row_str = "HOST\t"
        else:
            row_str = "GPU%d\t" % i
        
        for j in range(accum.shape[1]):
            row_str = row_str + "%d" % (accum[i][j] / (1024 * 1024)) + "\t"
        print(row_str)

def gpu_profile(cfg, df_gpu):
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

    comm_profile(cfg, df_gpu)
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))

    print_title("Summary of Kernels")
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))
    print(("MeasuredAllReduceTime : %lf (s)" % all_reduce_time))


def net_profile(cfg, df):
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


def cpu_profile(cfg, df):
    print_title("CPU Profiling: Task Time (IO included) for each Cores (s)")
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

# TODO: Analyze vmstat instead.
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
