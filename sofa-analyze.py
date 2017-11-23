#!/usr/bin/python
from scapy.all import *
import pandas as pd
import numpy as np
import csv

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
print df.groupby("deviceId")["data_B"].sum()/1000000

print("Data Traffic for each CopyKind (MB)");
data =  df.groupby("copyKind")["data_B"].sum()/1000000
print data 

print("Data Traffic Overhead for each CopyKind");
durations = df.groupby("copyKind")["duration"].sum()
print durations 


pcie3_bw_h2d = (data.loc[1]/durations.loc[1])/1000
print("Averaged Achieved PCIe-Gen3 H2D Bandwidth: %.1lf (GB/s)" % (pcie3_bw_h2d) ) 
pcie3_bw_d2h = (data.loc[2]/durations.loc[2])/1000
print("Averaged Achieved PCIe-Gen3 D2H Bandwidth: %.1lf (GB/s)" % (pcie3_bw_d2h) ) 
gpumem_bw_d2d = (data.loc[8]/durations.loc[8])/1000
print("Averaged Achieved GPU Memory D2D Bandwidth: %.1lf (GB/s)" % (gpumem_bw_d2d) ) 
nvlink_bw = (data.loc[10]/durations.loc[10])/1000
print("Averaged Achieved NVLink Bandwidth: %.1lf (GB/s)" % (nvlink_bw) ) 




