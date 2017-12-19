#!/usr/bin/python
from scapy.all import *
import sqlite3
import pandas as pd
import numpy as np
import csv
import cxxfilt
import json


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
    print(bcolors.OKBLUE + "[INFO] " + content + bcolors.ENDC)



print('Argument List: %s', str(sys.argv))
logdir = []
filein = []

if len(sys.argv) < 2:
    print_info("Usage: sofa-preproc.py /path/to/logdir")
    quit()
else:
    logdir = sys.argv[1] + "/"
    filein = logdir + "gputrace.nvp"


class CPUTrace:
    fieldnames = [
        'time',
        "event",
        "duration",
        "deviceId",
        "pid",
        "tid",
        "data",
        "pkt_src",
        "pkt_dst"]
    time = 0
    duration = 0
    event = -1
    name = "none"
    vaddr = -1
    deviceId = -1
    pid = -1
    tid = -1
    data = 0
    pkt_src = -1
    pkt_dst = -1

    def info(self):
        return 'hello world'

with open(logdir + 'sofa_time.txt') as f:
    t_glb_base = float(f.readlines()[0])
    print t_glb_base

iptable = []


series = []
net_traces = []
cpu_traces = []

cputrace = CPUTrace()
with open(logdir + 'cputrace.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=cputrace.fieldnames)
    writer.writeheader()
    packets = rdpcap(logdir + 'sofa.pcap')
    t_base = 0
    net_traces = []
    sid = 0
    symbols = []
    for i in range(0, len(packets)):
        try:
            time = packets[i][IP].time
            if i == 0:
                t_base = time
            if (i % 1) == 0:
                t_begin = (time - t_base) + t_glb_base
                t_end = (time - t_base) + t_glb_base
                duration = t_end - t_begin
                cputrace.time = t_begin
                cputrace.pkt_src = packets[i][IP].src.split('.')[3]
                cputrace.pkt_dst = packets[i][IP].dst.split('.')[3]
                cputrace.data = packets[i].len
                cputrace.name = "%s->%s:%d" % (
                    cputrace.pkt_src,
                    cputrace.pkt_dst,
                    cputrace.data)
                cputrace.event = cputrace.data * 100 + 17
                net_traces.append(
                    {"x": cputrace.time,
                     "y": cputrace.event,
                     "name": cputrace.name})
                writer.writerow(
                    {'time': cputrace.time,
                     'event': cputrace.event,
                     'pid': cputrace.pid,
                     'tid': cputrace.tid,
                     'deviceId': cputrace.deviceId,
                     'duration': cputrace.duration,
                     'data': cputrace.data,
                     'pkt_src': cputrace.pkt_src,
                     'pkt_dst': cputrace.pkt_dst})
        except Exception as e:
            print(e)
    series.append(
        {"name": 'Network',
                 "color": 'rgba(3, 183, 183, .5)',
                 "data": net_traces})
    print("Number of network traces: %d" % (len(net_traces)))

with open(logdir + 'cputrace.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=cputrace.fieldnames)
    # 0/0     [001] 15880.413677:   10101010  ffffffff816ab576
    # native_safe_halt1
    with open(logdir + 'perf.script') as f:
        lines = f.readlines()
        t_base = 0
        for i in range(0, len(lines)):
            fields = lines[i].split()
            time = float(fields[2].split(':')[0])
            if i == 0:
                t_base = time
            func_name = fields[5]
            if (i % 1) == 0:
                t_begin = time - t_base + t_glb_base
                t_end = time - t_base + t_glb_base
                duration = t_end - t_begin
                cputrace.time = t_begin
                cputrace.pid = int(fields[0].split('/')[0])
                cputrace.tid = int(fields[0].split('/')[1])
                cputrace.vaddr = int("0x" + fields[4], 16) % 1000000
                cputrace.deviceId = int(fields[1].split('[')[1].split(']')[0])
                cputrace.duration = float(fields[3]) / 1.5e9
                cputrace.data = 0
                cputrace.name = fields[5].replace("[", "_").replace("]", "_")
                # print(cputrace.name)
                cputrace.event = cputrace.vaddr
                # cpu_traces.append([cputrace.time,  cputrace.event,
                # cputrace.name])
                cpu_traces.append(
                    {"x": cputrace.time,
                     "y": cputrace.event,
                     "name": cputrace.name})
                writer.writerow(
                    {'time': cputrace.time,
                     'event': cputrace.event,
                     'pid': cputrace.pid,
                     'tid': cputrace.tid,
                     'deviceId': cputrace.deviceId,
                     'duration': cputrace.duration,
                     'data': cputrace.data,
                     'pkt_src': cputrace.pkt_src,
                     'pkt_dst': cputrace.pkt_dst})

    series.append(
        {"name": 'CPU',
                 "color": 'rgba(223, 83, 83, .5)',
                 "data": cpu_traces})
    print("Number of CPU traces: %d" % (len(cpu_traces)))


print(" length of series: %d" % (len(series)))
with open(logdir + 'report.js', 'w') as jsonfile:
    jsonfile.write("trace_data = ")
    json.dump(series, jsonfile)
    jsonfile.write("\n")

print_progress("read nvprof traces -- begin")
sqlite_file = filein
db = sqlite3.connect(sqlite_file)
cursor = db.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
i = 0
ftable = []
for table_name in tables:
    i = i + 1
    tname = table_name[0]
    table = pd.read_sql_query("SELECT * from %s" % tname, db)
    # print("table-%d = %s, count=%d" % (i,tname,len(table.index)) )
    if len(table.index) > 0:
        table.to_csv(logdir + tname + '.csv', index_label='index')
    if tname == "StringTable":
        ftable = table
print_progress("read nvprof traces -- end")

class GPUTrace:
    fieldnames = [
        'time',
        "event",
        "duration",
        "copyKind",
        "deviceId",
        "data_B",
        "streamId"]
    time = 0
    event = 0
    copyKind = 0
    deviceId = 0
    streamId = 0
    duration = 0
    size = 0

    def info(self):
        return 'hello world'

gputrace = GPUTrace()
gpu_kernel_traces = []
gpu_memcpy_traces = []
gpu_memcpy2_traces = []
print_progress("query CUDA kernels -- begin")
try:
    cursor.execute(
        "SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
except sqlite3.OperationalError:
    #print_warning("Cannot find CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
    try:
        cursor.execute("SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    except sqlite3.OperationalError:
        #print_warning("Cannot find CUPTI_ACTIVITY_KIND_KERNEL")
        quit()
print_progress("query CUDA kernels -- end")



print_progress("export-csv CUDA kernels -- begin")
records = cursor.fetchall()
i = 0
begin = []
end = []
event = []
t_base = 0
gputraces = []
gpu_series = []
with open(logdir + 'gputrace.csv', 'w') as f_gputrace:
    writerA = csv.DictWriter(f_gputrace, fieldnames=gputrace.fieldnames)
    writerA.writeheader()
    i=0 
    for record in records:
        i = i + 1
        if i == 1:
            t_base = record[0]
        if True:
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end = (record[1] - t_base) / 1e9 + t_glb_base
            duration = t_end - t_begin
            func_name = cxxfilt.demangle( ("%s" % ftable.loc[ftable._id_ == record[2], 'value']))
            event_id = record[2]
            gputrace.time = t_begin
            gputrace.event = record[2]
            gputrace.copyKind = -1
            gputrace.deviceId = record[4]
            gputrace.streamId = record[3]
            gputrace.duration = duration
            gputrace.data = 0
            # print("event id and its name = %d %s" % (event_id,func_name))
            # print("record-%d: %s at %lf, duration = %lf" % (i,record,
            # t_begin, t_end-t_begin) )
            writerA.writerow(
                {'time': gputrace.time,
                 'event': gputrace.event,
                 'copyKind': gputrace.copyKind,
                 'deviceId': gputrace.deviceId,
                 'streamId': gputrace.streamId,
                 'duration': gputrace.duration,
                 'data_B': gputrace.data})
        if i%100 == 0 : 
            gpu_kernel_traces.append(
                    {"x": gputrace.time,
                     "y": gputrace.duration,
                     "name": "%d"%gputrace.event})
    
    gpu_series.append(
            {   "name": 'Compute',
                "color": 'rgba(255, 80, 80, .5)',
                "data": gpu_kernel_traces})

print_progress("export-csv CUDA kernels -- end")


print_progress("query CUDA memcpy (h2d,d2h,d2d) -- begin")
# index,_id_,copyKind,srcKind,dstKind,flags,bytes,start,end,deviceId,contextId,streamId,correlationId,runtimeCorrelationId
cursor.execute(
    "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY")
records = cursor.fetchall()
print_progress("query CUDA memcpy (h2d,d2h,d2d) -- end")


print_progress("csv-export CUDA memcpy (h2d,d2h,d2d) -- begin")
i = 0
begin = []
end = []
event = []
t_base = 0
with open(logdir + 'gputrace.csv', 'a') as f_gputrace, open(logdir + 'gpu-overhead.csv','a') as f_gpu_overhead:
    writerA = csv.DictWriter(f_gputrace, fieldnames=gputrace.fieldnames)
    writerB = csv.DictWriter(f_gpu_overhead, fieldnames = [
        'timestamp',
        "compute",
        "communicate"])
    i = 0 
    for record in records:
        i = i + 1
        if i == 1:
            t_base = record[0]
        if True:
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end = (record[1] - t_base) / 1e9 + t_glb_base
            duration = t_end - t_begin
            gputrace.time = t_begin
            gputrace.event = -1
            gputrace.copyKind = record[3]
            gputrace.deviceId = record[4]
            gputrace.streamId = record[7]
            gputrace.duration = duration
            gputrace.data = record[2]
            
            writerA.writerow(
                {'time': gputrace.time,
                 'event': gputrace.event,
                 'copyKind': gputrace.copyKind,
                 'deviceId': gputrace.deviceId,
                 'streamId': gputrace.streamId,
                 'duration': gputrace.duration,
                 'data_B': gputrace.data})

        if i%100 == 0 : 
            gpu_memcpy_traces.append(
                    {"x": gputrace.time,
                     "y": gputrace.duration,
                     "name": ("%dB"%gputrace.data)})
    
    gpu_series.append(
            {   "name": 'memcpy',
                "color": 'rgba(80, 255, 80, .5)',
                "data": gpu_memcpy_traces})

print_progress("csv-export CUDA memcpy (h2d,d2h,d2d) -- end")


print_progress("query CUDA memcpy2 (p2p) -- begin")
cursor.execute(
    "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY2")
records = cursor.fetchall()
print_progress("query CUDA memcpy2 (p2p) -- end")

print_progress("csv-export CUDA memcpy2 (p2p) -- begin")
begin = []
end = []
event = []
with open(logdir + 'gputrace.csv', 'a') as f_gputrace, open(logdir + 'gpu-overhead.csv','a') as f_gpu_overhead:
    writerA = csv.DictWriter(f_gputrace, fieldnames=gputrace.fieldnames)
    writerB = csv.DictWriter(f_gpu_overhead, fieldnames = [
        'timestamp',
        "compute",
        "communicate"])

    t_base = 0
    i = 0
    for record in records:
        i = i + 1
        if i == 1:
            t_base = record[0]
        if True:
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end = (record[1] - t_base) / 1e9 + t_glb_base
            duration = t_end - t_begin
            gputrace.time = t_begin
            gputrace.event = -1
            gputrace.copyKind = record[3]
            gputrace.deviceId = record[4]
            gputrace.streamId = record[7]
            gputrace.duration = duration
            gputrace.data = record[2]
            writerA.writerow(
                {'time': gputrace.time,
                 'event': gputrace.event,
                 'copyKind': gputrace.copyKind,
                 'deviceId': gputrace.deviceId,
                 'streamId': gputrace.streamId,
                 'duration': gputrace.duration,
                 'data_B': gputrace.data})
            
        if i%100 == 0:
            gpu_memcpy2_traces.append(
                    {"x": gputrace.time,
                     "y": gputrace.duration,
                     "name": ("%dB"%gputrace.data)})

    gpu_series.append(
    {   "name": 'memcpy2',
        "color": 'rgba(80, 80, 255, .5)',
        "data": gpu_memcpy2_traces})
print_progress("csv-export CUDA memcpy2 (p2p) -- end")

print_progress("export overhead.js -- begin")
print("Length of GPU series: %d" % (len(gpu_series)))
with open(logdir + 'gpu-overhead.js', 'w') as jsonfile:
    jsonfile.write("trace_data = ")
    json.dump(gpu_series, jsonfile)
    jsonfile.write("\n")
print_progress("export overhead.js -- end")
