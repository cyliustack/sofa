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

print("Overhead brought from Each Stream");
stream_durations = df.groupby("streamId")["duration"].sum()
print(stream_durations)

print("Avertimestampd Achieved Bandwidth for each CopyKind: (GB/s)") 
bw = data_copyKind/durations_copyKind/1000
print(bw)

print("Overlapness for All Events (s)");
class Event:
        def __init__(self, name, ttype, timestamp):
                self.name = name
                self.ttype = ttype # 0 for begin, 1 for end
                self.timestamp = timestamp
        def __repr__(self):
                return repr((self.name, self.ttype, self.timestamp))


events = []
for i in range(len(df)):
	#print("df[%d]=%lf" % (i,df.iloc[i]['time']))
	e = Event(i,0,i)
	events.append(e)
	e = Event(i,1,i+0.5*random.randint(1,10))
	events.append(e)
events.sort(key=attrgetter('timestamp'))
#sorted(events, key=lambda events: events.ttype)
#events.sort()
event_stack=[]
overlaptime=0
for e in events:
	if e.ttype == 0:
		event_stack.append(e)
	if e.ttype == 1:
		print("reach end of time for event-%d"%(e.name))
		# find all the previous event with 
		for es in event_stack:
			print("top of event_stack: name:%d"%(event_stack[len(event_stack)-1].name))
			overlaptime = overlaptime + (e.timestamp - es.timestamp)
				
	print("e.t = %lf, e.ttype = %d" % (e.timestamp,e.ttype))
	
print("overlapped time: %lf"%(overlaptime))



