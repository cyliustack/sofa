#!/usr/bin/python
from scapy.all import *
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


def gpu_profile(df_gpu):

    total_kernel_time = 0.0
    total_gpu_time = 0.0
    total_memcopy_time = 0.0
    total_traffic = 0.0
    step_kernel_time = 0.0
    batch_size = 64
    throughput = 1414.0
    n_steps = 20

    print_title("Task Time (IO included) for each Device (s)")
    grouped_df = df_gpu.groupby("deviceId")["duration"]
    total_tasktime = 0
    for key, item in grouped_df:
        print("[%d]: %lf" % (int(float(key)), grouped_df.get_group(key).sum()))
        total_tasktime = total_tasktime + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    per_gpu_time = total_tasktime / n_devices
    step_gpu_time = per_gpu_time / n_steps
    print("Averaged GPU time of devices: %.2lf" % per_gpu_time)
    theory_overlaptime = step_gpu_time * (n_devices * (n_devices - 1) / 2)

    print_title("Data Traffic (bidirection) for each Device (MB)")
    grouped_df = df_gpu.groupby("deviceId")["payload"]
    for key, item in grouped_df:
        print("[%d]: %lf" % (key, grouped_df.get_group(key).sum() / 1000000.0))
        total_traffic = total_traffic + \
            grouped_df.get_group(key).sum() / 1000000.0
    print("Total traffic: %.2lf" % total_traffic)

    print_title("Data Traffic for each CopyKind (MB)")
    data_copyKind = grouped_df = df_gpu.groupby("copyKind")["payload"]
    for key, item in grouped_df:
        print(
            "[%s]: %lf" %
            (cktable[key], grouped_df.get_group(key).sum() / 1000000.0))

    print_title("Data Communication Time for each CopyKind (s)")
    durations_copyKind = grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        print("[%s]: %lf" % (cktable[key], grouped_df.get_group(key).sum()))
        if key == -1:
            total_kernel_time = grouped_df.get_group(key).sum()
        else:
            total_memcopy_time = total_memcopy_time + \
                grouped_df.get_group(key).sum()

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
    stream_durations = []
    for key, item in grouped_df:
        if cfg['enable_verbose'] == "true":
            print("[%d]: %lf" % (key, grouped_df.get_group(key).sum()))
        stream_durations = np.append(
            stream_durations,
            grouped_df.get_group(key).sum())
    topk_streams = np.sort(stream_durations)[-8:]
    print(topk_streams)
    print("Mean of Top-%d Stream Times = %.2lf" %
          (len(topk_streams), np.mean(topk_streams)))

    print_title("Averaged Achieved Bandwidth for each CopyKind: (GB/s)")
    bw = (data_copyKind.sum() / 1000000) / durations_copyKind.sum() / 1000
    for i in range(len(bw)):
        print("[%s]: %.3lf" % (cktable[bw.keys()[i]], bw.iloc[i]))

    print_title("Model Performance (Meas.)")
    meas = pd.DataFrame(
        [],
        columns=[
            'step_time',
            'step_gpu_time',
            'step_kernel_time'])

    step_time = (n_devices * batch_size / throughput)
    step_kernel_time = total_kernel_time / float(n_devices) / n_steps

    print("Measured Total MemCopy Time = %lf (s)" % total_memcopy_time)
    print("Measured Total Traffic = %lf (MB)" % total_traffic)
    print("Detected Number of Steps = %d" % n_steps)
    print("Measured Step Time = %lf (s)" % step_time)
    print("Measured Step Kernel/MemCopy Time = %lf (s)" % step_gpu_time)
    print("Measured Step Kernel Time = %lf (s)" % step_kernel_time)
    meas['step_time'] = step_time
    meas['step_gpu_time'] = step_gpu_time
    meas['step_kernel_time'] = step_kernel_time
    print_title("Model Performance (Calc.)")
    step_memcopy_time = step_gpu_time - step_kernel_time
    print("Calculated Step MemCopy Time = %lf (s)" % step_memcopy_time)
#    print("Calculated Kernel Time = %lf" % () )

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
        print(
            "Theoritical overlapped time of Events: %lf" %
            (theory_overlaptime))


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
        gpu_profile(df_gpu)
    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

    try:
        df_cpu = pd.read_csv(filein_cpu)
        cpu_profile(cfg, df_cpu)
    except IOError:
        print_warning(
            "cputrace.csv is not found")
        quit()
