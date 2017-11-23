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
print "Data Traffic for each Device (MB)";
print df.groupby("deviceId")["data_B"].sum()/1000000
print "Data Traffic for each CopyKind (MB)";
print df.groupby("copyKind")["data_B"].sum()/1000000
