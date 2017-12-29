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
ckindex = [1,2,8,10]
       
def gpu_profile(df_gpu): 
    #df_gpu = df_gpu.convert_objects(convert_numeric=True)
    #with open(logdir + 'overhead.js', 'w') as jsonfile:
    #    x = y = data = []
    #    A = df_gpu.groupby("copyKind")
    #    for ck in range(len(A)):
    #        print(ck)
    #        y = A.get_group(ckindex[ck])["duration"]
    #        x = A.get_group(ckindex[ck])["time"]
    #        for i in range(0,len(x)):
    #            if i%1 == 0 :
    #                data.append([x.iloc[i],y.iloc[i]])
    #        jsonfile.write("overhead_"+cktable[ckindex[ck]]+" = ")
    #        json.dump(data, jsonfile)
    #        jsonfile.write("\n")
          
    print_title("Task Time (IO included) for each Device (s)")
    grouped_df = df_gpu.groupby("deviceId")["duration"]
    total_tasktime = 0
    for key, item in grouped_df:
        print("[%d]: %lf" % (int(float(key)), grouped_df.get_group(key).sum() ))
        total_tasktime = total_tasktime + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    avg_tasktime = total_tasktime / n_devices
    print("Averaged task time of devices: %.2lf"%avg_tasktime)
    theory_overlaptime = avg_tasktime * (n_devices * (n_devices - 1) / 2)

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

    print_title("Data Communication Time for each CopyKind (s)")
    durations_copyKind = grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        print("[%s]: %lf" % (cktable[key], grouped_df.get_group(key).sum()))

    if cfg['enable_verbose'] == "true":
        print_title("Data Traffic for Each Pair of deviceId and CopyKind (MB)")
        devcopy = grouped_df = df_gpu.groupby(["deviceId","copyKind"])["payload"].sum()/1000000
        print(devcopy)
        print_title("Data Communication Time for Each Pair of deviceId and CopyKind (s)")
        devcopytime = grouped_df = df_gpu.groupby(["deviceId","copyKind"])["duration"].sum()
        print(devcopytime)


    print_title("Task Time spent on Each Stream (s)")
    grouped_df = df_gpu.groupby("pid")["duration"]
    stream_durations = []
    for key, item in grouped_df:
        if cfg['enable_verbose'] == "true":
            print("[%d]: %lf" % (key, grouped_df.get_group(key).sum()))
        stream_durations = np.append( stream_durations, grouped_df.get_group(key).sum() )
    topk_streams = np.sort(stream_durations)[-8:]
    print(topk_streams)
    print("Mean of Top-%d Stream Times = %.2lf" % (len(topk_streams),np.mean(topk_streams)))
    

    print_title("Averaged Achieved Bandwidth for each CopyKind: (GB/s)")
    bw = (data_copyKind.sum() / 1000000) / durations_copyKind.sum() / 1000
    for i in range(len(bw)):
        print("[%s]: %.3lf" % (cktable[bw.keys()[i]], bw.iloc[i]))


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
    print("total execution time = %.3lf" % total_exec_time )
    print("average execution time across devices = %.3lf" % avg_exec_time )


def cpu_overhead_report(df,cfg): 
    print_progress("cpu overhead report -- begin")
        #with open(logdir + 'overhead.js', 'w') as jsonfile:
    #    x = y = data = []
    #    A = df_gpu.groupby("copyKind")
    #    for ck in range(len(A)):
    #        print(ck)
    #        y = A.get_group(ckindex[ck])["duration"]
    #        x = A.get_group(ckindex[ck])["time"]
    #        for i in range(0,len(x)):
    #            if i%1 == 0 :
    #                data.append([x.iloc[i],y.iloc[i]])
    #        jsonfile.write("overhead_"+cktable[ckindex[ck]]+" = ")
    #        json.dump(data, jsonfile)
    #        jsonfile.write("\n")
          

    print_progress("cpu overhead report -- end")
 

if __name__ == "__main__":
    print('Number of arguments: %d arguments' % len(sys.argv))
    print('Argument List: %s' % str(sys.argv))
    logdir = []
    filein = []
    enable_verbose=False
    enable_overlapness = False
    df_gpu=[]
    df_cpu=[]

    parser = argparse.ArgumentParser(description='SOFA Analyze')
    parser.add_argument("--logdir", metavar="/path/to/logdir/", type=str, required=True, 
                    help='path to the directory of SOFA logged files')
    parser.add_argument('--config', metavar="/path/to/config.cfg", type=str, required=True,
                    help='path to the directory of SOFA configuration file')
    
    args =parser.parse_args()
    logdir = args.logdir + "/"
    filein_gpu = logdir + "gputrace.csv"
    filein_cpu = logdir + "cputrace.csv"

    cfg = read_config(args.config) 
    #try:
    #    with open(args.config) as f:
    #        cfg = json.load(f)
    #except:
    #    with open( 'sofa.cfg', "w") as f:
    #        json.dump(cfg,f)
    #        f.write("\n")
    #print_info("SOFA Configuration: ")    
    #print(cfg)

    try:
        df_gpu = pd.read_csv(filein_gpu)
        #gpu_overhead_report(df_gpu,cfg)        
        gpu_profile(df_gpu)
    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

    try:
        df_cpu = pd.read_csv(filein_cpu)
        cpu_overhead_report(df_cpu,cfg)        
        cpu_profile(cfg, df_cpu)
    except IOError:
        print_warning(
            "cputrace.csv is not found")
        quit()

