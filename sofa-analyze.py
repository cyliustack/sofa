#!/usr/bin/python
from scapy.all import *
import pandas as pd
import numpy as np
import csv
import random
from operator import itemgetter, attrgetter
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
logdir = []
filein = []

if len(sys.argv) < 2:
    print("Ustimestamp: sofa-report.py /path/to/logdir")
    quit()
else:
    logdir = sys.argv[1] + "/"
    filein = logdir + "gputrace.csv"

df = pd.read_csv(filein)
print("Data Traffic for each Device (MB)")
print(df.groupby("deviceId")["data_B"].sum() / 1000000)

print("Execution Time for each Device (MB)")
print(df.groupby("deviceId")["duration"].sum())

print("Data Traffic for each CopyKind (MB)")
data_copyKind = df.groupby("copyKind")["data_B"].sum() / 1000000
print(data_copyKind)

print("Data Traffic Overhead for each CopyKind")
durations_copyKind = df.groupby("copyKind")["duration"].sum()
print(durations_copyKind)

print("Overhead brought from Each Stream")
stream_durations = df.groupby("streamId")["duration"].sum()
print(stream_durations)

print("Avertimestampd Achieved Bandwidth for each CopyKind: (GB/s)")
bw = data_copyKind / durations_copyKind / 1000
print(bw)

print("Overlapness for All Events (s)")

# Assume pa<pb, pc<pd:
def overlap(pa, pb, pc, pd):
    if pb - pc >= 0 and pd - pa >=0:   
        return  min(pb, pd) - max(pa, pc)

class Event:
    def __init__(self, name, ttype, timestamp, duration):
        self.name = name
        self.ttype = ttype  # 0 for begin, 1 for end
        self.timestamp = timestamp
        self.duration = duration

    def __repr__(self):
        return repr((self.name, self.ttype, self.timestamp, self.duration))

events = []
for i in range(len(df)):
    t_begin = df.iloc[i]['time'] 
    d = df.iloc[i]['duration'] 
    t_end = t_begin + d 
    e = Event(i, 0, t_begin, d)
    events.append(e)
    e = Event(i, 1, t_end, d)
    events.append(e)
#for i in range(3):
#    #print("df[%d]=%lf" % (i,df.iloc[i]['time']))
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
    #print(event_stack)
    if e.ttype == 0:
        event_stack.append(e)
    if e.ttype == 1:
        #print("reach end of time for event-%d" % (e.name))
        # find all the previous event with
        for es in event_stack:
            if es.name != e.name:
                #print("n:%d t:%lf d:%lf overlaptime:%lf" % (es.name, es.timestamp, es.duration, overlaptime))
                overlaptime = overlaptime + overlap(es.timestamp,es.timestamp+es.duration, e.timestamp-e.duration,e.timestamp)
        #print("pop out %d" % e.name)
        event_stack = [es for es in event_stack if es.name != e.name]
print("overlapped time of Events: %lf" % (overlaptime))
