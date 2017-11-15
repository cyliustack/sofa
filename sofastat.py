#!/usr/bin/python
from scapy.all import *
import sqlite3
import pandas as pd
import numpy as np
import csv
import cxxfilt
print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv) 
filein = []
if len(sys.argv) < 2:
    print("Usage: sofastat.py test.nvp")
    quit();
else:
    filein = sys.argv[1]




# rdpcap comes from scapy and loads in our pcap file
packets = rdpcap('sofa.pcap')
for i in range(0,len(packets)):
	src = packets[i][IP].src
	dst = packets[i][IP].dst
	payload = packets[i].len	
	print("[%d] src:%s dst:%s len:%d " % ( i, src, dst, payload))

with open('perf.script') as f:
	lines = f.readlines()

count = 0
x = []
for i in range(0,len(lines)):
	input = '1 3.0 false hello'
	count = count + 1
	fields = lines[i].split()
	timestamp = fields[3].split(':')[0]
	func_name = fields[7]
	x.append([count, timestamp, func_name]) 
	#print("x = (%d, %s, %s)" % (count, timestamp, func_name))
print(x)

print("Read nvprof traces ...")
sqlite_file = filein
#sqlite_file = 'mm_2262.nvp'
#sqlite_file = 'vgg16_bs64_gpux8_119832.nvprof'
db = sqlite3.connect(sqlite_file)
cursor = db.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
i=0;
ftable=[];
for table_name in tables: 
    i=i+1
    tname = table_name[0]
    table = pd.read_sql_query("SELECT * from %s" % tname, db)
    print("table-%d = %s, count=%d" % (i,tname,len(table.index)) )
    if len(table.index) > 0:
        table.to_csv(tname + '.csv', index_label='index')
    if tname == "StringTable":
        ftable=table
#        print("StringTable Content:")
#        for record in table:
#            ftable['Name']=record[1]
#            print("record = %s" % (record)) 


class GPUTrace:
    fieldnames = ['time(ms)', "event", "duration(ms)","copyKind", "data(bytes)", "streamId"]
    time=0
    event=0
    copyKind=0
    streamId=0
    duration=0
    size=0
    def info(self):
	    return 'hello world'

gputrace = GPUTrace()


cursor.execute("SELECT start,end,name,streamId FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
records = cursor.fetchall()
i=0
begin = []
end = []
event = []
t_base = 0

with open('gputrace.csv', 'w') as csvfile: 
    writer = csv.DictWriter(csvfile, fieldnames=gputrace.fieldnames)
    writer.writeheader()
    for record in records:
        i = i + 1
        if i == 1:
            t_base = record[0]
        if ( i % 10 ) == 0 :
            print(record)
            t_begin = record[0] - t_base
            t_end = record[1]- t_base
            duration = t_end - t_begin
            begin = np.append(begin, t_begin )
            end = np.append(end, t_end )
            func_name = cxxfilt.demangle( ("%s" % ftable.loc[ftable._id_==record[2],'value'])) 
            event_id = record[2]
            gputrace.time=t_begin/1000000 #from ns to ms
            gputrace.event=record[2]
            gputrace.copyKind=-1
            gputrace.streamId=record[3]
            gputrace.duration=duration/1000000 # from ns to ms
            gputrace.data=0
            print("event id and its name = %d %s" % (event_id,func_name)) 
            event = np.append(event, event_id)
            print("record-%d: %s at %d, duration = %d" % (i,record, t_begin, t_end-t_begin) )
            print("ID-%d = %s" % ( record[2], func_name ))
            writer.writerow({'time(ms)': gputrace.time, 'event': gputrace.event, 'copyKind': gputrace.copyKind, 'streamId':gputrace.streamId, 'duration(ms)':gputrace.duration, 'data(bytes)': gputrace.data })

#index,_id_,copyKind,srcKind,dstKind,flags,bytes,start,end,deviceId,contextId,streamId,correlationId,runtimeCorrelationId
cursor.execute("SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY")
records = cursor.fetchall()
i=0
begin = []
end = []
event = []
t_base = 0
with open('gputrace.csv', 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=gputrace.fieldnames)
    writer.writeheader()
    for record in records:
        i = i + 1
        if i == 1:
            t_base = record[0]
        if ( i % 10 ) == 0 :
            t_begin = record[0] - t_base
            t_end = record[1]- t_base
            duration = t_end - t_begin
            begin = np.append(begin, t_begin )
            end = np.append(end, t_end )
            event_id = record[2]
            gputrace.time=t_begin/1000000 #from ns to ms
            gputrace.event=record[4] # deviceId
            gputrace.copyKind=record[3]
            gputrace.streamId=record[7]
            gputrace.duration=duration/1000000 # from ns to ms
            gputrace.data=record[2]
            writer.writerow({'time(ms)': gputrace.time, 'event': gputrace.event, 'copyKind': gputrace.copyKind, 'streamId':gputrace.streamId, 'duration(ms)':gputrace.duration, 'data(bytes)': gputrace.data })
            print(record)
            #writer.writerow({'begin': t_begin, 'duration': duration, 'bytes': record[2], 'copyKind':record[3], 'deviceId':record[4], 'srcKind':record[5], 'dstKind':record[6], 'streamId':record[7] })
 

#cursor.execute("SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY2")
#records = cursor.fetchall()
#i=0
#begin = []
#end = []
#event = []
#t_base = 0
#with open('gputrace_memcpy2.csv', 'w') as csvfile:
#    fieldnames = ['begin', "duration", "bytes", "copyKind", "deviceId", "srcKind","dstKind","streamId"]
#    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#    writer.writeheader()
#    for record in records:
#        i = i + 1
#        if i == 1:
#            t_base = record[0]
#        if ( i % 1 ) == 0 :
#            print(record)
#            t_begin = record[0] - t_base
#            t_end = record[1]- t_base
#            duration = t_end - t_begin
#            writer.writerow({'begin': t_begin, 'duration': duration, 'bytes': record[2], 'copyKind':record[3], 'deviceId':record[4], 'srcKind':record[5], 'dstKind':record[6], 'streamId':record[7] })
# 



    
#for record in records:
#    i = i + 1
#    if i == 1:
#        t_base = record[0]/1000
#    if ( i % 10 ) == 0 :
#        print(record)
#        end = np.append(end, t_end )
#        func_name = ftable.loc[ftable._id_==record[2],'value']
#        event = np.append(event, func_name)
#        print("record-%d: %s at %d, duration = %d" % (i,record, t_begin, t_end-t_begin) )
#        print("ID-%d = %s" % ( record[2], func_name ))
#        print("t_base = %d" % t_base) 
#plt.barh(range(len(begin)),  end-begin, left=begin)
#plt.yticks(range(len(begin)), event)
#plt.show()
