#!/usr/bin/python
from scapy.all import *
import pandas as pd
import numpy as np
import csv
import random
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv) 
logdir = []
filein = []

if len(sys.argv) < 2:
    print("Usage: sofa-report.py /path/to/logdir")
    quit();
else:
    logdir = sys.argv[1] + "/"
    filein = logdir+"gputrace.csv"

df = pd.read_csv(filein)
print("Data Traffic for each Device (MB)");
print(df.groupby("deviceId")["data_B"].sum()/1000000)

print("Execution Time for each Device (MB)");
print(df.groupby("deviceId")["duration"].sum())


print("Data Traffic for each CopyKind (MB)");
data_copyKind =  df.groupby("copyKind")["data_B"].sum()/1000000
print(data_copyKind)

print("Data Traffic Overhead for each CopyKind");
durations_copyKind = df.groupby("copyKind")["duration"].sum()
print(durations_copyKind)

print("Data Traffic Overhead for each CopyKind");
stream_durations = df.groupby("streamId")["duration"].sum()
print(stream_durations)


print("Averaged Achieved Bandwidth for each CopyKind: (GB/s)") 
bw = data_copyKind/durations_copyKind/1000
print(bw)	
