#!/usr/bin/python
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


class Event:

    def __init__(self, name, ttype, timestamp, duration):
        self.name = name
        self.ttype = ttype  # 0 for begin, 1 for end
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return repr((self.name, self.ttype, self.timestamp, self.duration))

# Assume pa<pb, pc<pd:


def overlap(pa, pb, pc, pd):
    if pb - pc >= 0 and pd - pa >= 0:
        return min(pb, pd) - max(pa, pc)


# print_format_table()
cktable = {-1: "KER", 1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
ckindex = [1, 2, 8, 10]

def comm_profile(cfg, df_gpu):
    n_gpus = 0
    for i in range(len(df_gpu)):
        if df_gpu.loc[i,'pkt_src'] > n_gpus:
            n_gpus = df_gpu.loc[i,'pkt_src']
     
    accum = np.zeros((1+n_gpus, 1+n_gpus))
    accum_count = np.zeros((1+n_gpus, 1+n_gpus))
    
    for i in range(len(df_gpu)):
        if df_gpu.loc[i,'copyKind'] == 0:
            continue
        src = df_gpu.loc[i,'pkt_src']
        dst = df_gpu.loc[i,'pkt_dst']
        payload = df_gpu.loc[i,'payload']
        accum[src][dst] = int(accum[src][dst] + payload)
        accum_count[src][dst] = int(accum_count[src][dst] + 1)
        
        if cfg['enable_verbose'] == "true":
            if df_gpu.loc[i,'copyKind'] == 1:
                print("[H2D] HOST%d to GPU%d, count:%d\tpayload:%d\taccum_payload:%d" % ( df_gpu.loc[i,'pkt_src'],df_gpu.loc[i,'pkt_dst'], accum_count[src][dst], payload, accum[src][dst]))
            if df_gpu.loc[i,'copyKind'] == 2:
                print("[D2H] GPU%d to HOST%d, count:%d\tpayload:%d\taccum_payload:%d" % ( df_gpu.loc[i,'pkt_src'],df_gpu.loc[i,'pkt_dst'], accum_count[src][dst], payload, accum[src][dst]))
            if df_gpu.loc[i,'copyKind'] == 10:
                print("[P2P] GPU%d to GPU%d: count:%d\tpayload:%d\taccum_payload:%d" % ( df_gpu.loc[i,'pkt_src'],df_gpu.loc[i,'pkt_dst'], accum_count[src][dst], payload, accum[src][dst]))

    row_str = "\tHOST\t"
    print_title("Summary of Comm. (MB)")
    for i in range(1,accum.shape[1]):
            row_str = row_str + "GPU%d"%i + "\t"
    print(row_str)

    for i in range(accum.shape[0]):
        if i == 0:
            row_str = "HOST\t"
        else:
            row_str = "GPU%d\t"%i
        for j in range(accum.shape[1]):
            row_str = row_str + "%d"%(accum[i][j]/(1024*1024)) + "\t"
        print(row_str)
        #for i in range(len(df_gpu)):

def gpu_profile(cfg, df_gpu):

    total_kernel_time = 0.0
    total_gpu_time = 0.0
    total_memcopy_time = 0.0
    total_traffic = 0.0
    total_h2d_traffic = 0.0
    total_d2h_traffic = 0.0
    total_p2p_traffic = 0.0
    top_k = int(cfg['top_k'])
    print_title("Task Time (MEMCPY included) for each Device (s)")
    grouped_df = df_gpu.groupby("deviceId")["duration"]
    total_tasktime = 0
    for key, item in grouped_df:
        print("[%d]: %lf" % (int(float(key)), grouped_df.get_group(key).sum()))
        total_tasktime = total_tasktime + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    per_gpu_time = total_tasktime / n_devices
    print("Averaged GPU time of devices: %.2lf" % per_gpu_time)

    print_title("Data Traffic (bidirection) for each Device (MB)")
    grouped_df = df_gpu.groupby("deviceId")["payload"]
    for key, item in grouped_df:
        print("[%d]: %lf" % (key, grouped_df.get_group(key).sum() / 1000000.0))

    print_title("Data Traffic for each CopyKind (MB)")
    data_copyKind = grouped_df = df_gpu.groupby("copyKind")["payload"]
    for key, item in grouped_df:
        print(
            "[%s]: %lf" %
            (cktable[key], grouped_df.get_group(key).sum() / 1000000.0))
        if int(key) == 1:
            total_h2d_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 2:
            total_d2h_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 10:
            total_p2p_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) != 8:
            total_traffic = total_traffic + \
                grouped_df.get_group(key).sum() / 1000000.0
    print("Total traffic: %.2lf" % total_traffic)

       
    print_title("Data Communication Time for each CopyKind (s)")
    durations_copyKind = grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        print("[%s]: %lf" % (cktable[key], grouped_df.get_group(key).sum()))
        if key == -1:
            total_kernel_time = grouped_df.get_group(key).sum()
        else:
            total_memcopy_time = total_memcopy_time + \
                grouped_df.get_group(key).sum()

    print_title("All-reduce Time (s)")
    all_reduce_time=0
    grouped_df = df_gpu.groupby("name")["duration"]
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("AllReduce") != -1:
            all_reduce_time = all_reduce_time +  grouped_df.get_group(key).sum()
                

    if cfg['enable_verbose'] == "true":
        print_title("Data Traffic for Each Pair of deviceId and CopyKind (MB)")
        devcopy = grouped_df = df_gpu.groupby(["deviceId", "copyKind"])[
            "payload"].sum() / 1000000
        print(devcopy)
        print_title(
            "Data Communication Time for Each Pair of deviceId and CopyKind (s)")
        devcopytime = grouped_df = df_gpu.groupby(
            ["deviceId", "copyKind"])["duration"].sum()
        print(devcopytime)

    print_title("Task Time spent on Each Stream (s)")
    grouped_df = df_gpu.groupby("pid")["duration"]
    

    s = []
    for key, item in grouped_df:
        s.append( [key, grouped_df.get_group(key).sum()] )
    topk_streams = sorted(s,key=lambda l:l[1], reverse=True)[:-top_k]
    for s in topk_streams:   
        print("[%d]: %.3lf" % (s[0],s[1]))
    print("Mean of Top-%d Stream Times = %.2lf" %
          (len(topk_streams), np.mean(topk_streams)))

    print_title("Averaged Achieved Bandwidth for each CopyKind: (GB/s)")
    bw = (data_copyKind.sum() / 1000000) / durations_copyKind.sum() / 1000
    for i in range(len(bw)):
        print("[%s]: %.3lf" % (cktable[bw.keys()[i]], bw.iloc[i]))

    print_title("Model Performance (Meas.)")


    print("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time)
    print("MeasuredTotalMemCopyTime : %lf (s)" % total_memcopy_time)
    print("MeasuredTotalTraffic : %lf (MB)" % total_traffic)
    print("MeasuredTotalH2DTraffic : %lf (MB)" % total_h2d_traffic)
    print("MeasuredTotalD2HTraffic : %lf (MB)" % total_d2h_traffic)
    print("MeasuredTotalP2PTraffic : %lf (MB)" % total_p2p_traffic)
    print("MeasuredAllReduceTime : %lf (s)" % all_reduce_time)
    print_title("Model Performance (Calc.)")

    if enable_overlapness:
        print_title("Overlapness for All Events (s)")
        events = []
        for i in range(len(df_gpu)):
            t_begin = df_gpu.iloc[i]['timestamp']
            d = df_gpu.iloc[i]['duration']
            t_end = t_begin + d
            e = Event(i, 0, t_begin, d)
            events.append(e)
            e = Event(i, 1, t_end, d)
            events.append(e)
        # for i in range(3):
        # print("df_gpu[%d]=%lf" % (i,df_gpu.iloc[i]['timestamp']))
        #    t_begin =   i
        #    d = 0.5 * random.randint(1, 10)
        #    t_end = t_begin + d
        #    e = Event(i, 0, t_begin, d)
        #    events.append(e)
        #    e = Event(i, 1, t_end, d)
        #    events.append(e)
        events.sort(key=attrgetter('timestamp'))

        event_stack = []
        overlaptime = 0
        for e in events:
            # print(event_stack)
            if e.ttype == 0:
                event_stack.append(e)
            if e.ttype == 1:
                # print("reach end of time for event-%d" % (e.name))
                # find all the previous event with
                for es in event_stack:
                    if es.name != e.name:
                        # print("n:%d t:%lf d:%lf overlaptime:%lf" % (es.name,
                        # es.timestamp, es.duration, overlaptime))
                        overlaptime = overlaptime + overlap(
                            es.timestamp,
                            es.timestamp + es.duration,
                            e.timestamp - e.duration,
                            e.timestamp)
                # print("pop out %d" % e.name)
                event_stack = [es for es in event_stack if es.name != e.name]
        print("Measured Overlapped time of Events: %lf" % (overlaptime))

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
    print("total network time = %.3lf" % total_net_time)
    print("total amount of network packets  = %d" % n_packets)

def cpu_profile(cfg, df):
    print_title("CPU Profiling: Task Time (IO included) for each Cores (s)")
    grouped_df = df.groupby("deviceId")["duration"]
    total_exec_time = 0
    for key, item in grouped_df:
        if cfg['enable_verbose'] == "true":
            print("[%d]: %lf" % (key, grouped_df.get_group(key).sum()))
        total_exec_time = total_exec_time + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    avg_exec_time = total_exec_time / n_devices
    print("total execution time = %.3lf" % total_exec_time)
    print("average execution time across devices = %.3lf" % avg_exec_time)


if __name__ == "__main__":
    print('Number of arguments: %d arguments' % len(sys.argv))
    print('Argument List: %s' % str(sys.argv))
    logdir = []
    filein = []
    enable_verbose = False
    enable_overlapness = False
    df_gpu = []
    df_cpu = []

    parser = argparse.ArgumentParser(description='SOFA Analyze')
    parser.add_argument(
        "--logdir",
        metavar="/path/to/logdir/",
        type=str,
        required=True,
        help='path to the directory of SOFA logged files')
    parser.add_argument(
        '--config',
        metavar="/path/to/config.cfg",
        type=str,
        required=True,
        help='path to the directory of SOFA configuration file')

    args = parser.parse_args()
    logdir = args.logdir + "/"
    filein_gpu = logdir + "gputrace.csv"
    filein_cpu = logdir + "cputrace.csv"


    cfg = read_config(args.config)

    try:
        df_gpu = pd.read_csv(filein_gpu)
        gpu_profile(cfg, df_gpu)
        comm_profile(cfg, df_gpu)
    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

    try:
        df_cpu = pd.read_csv(filein_cpu)
        cpu_profile(cfg, df_cpu)
        net_profile(cfg, df_cpu)
    except IOError:
        print_warning(
            "cputrace.csv is not found")
        quit()
