#!/usr/bin/python
from scapy.all import *
import sqlite3
import pandas as pd

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
print x

print("Read data.nvprof...")
sqlite_file = 'mm_2262.nvprof'
#sqlite_file = 'vgg16_bs64_gpux8_119832.nvprof'
db = sqlite3.connect(sqlite_file)
cursor = db.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
i=0;
for table_name in tables:
    i=i+1
    table_name = table_name[0]
    table = pd.read_sql_query("SELECT * from %s" % table_name, db)
    table.to_csv(table_name + '.csv', index_label='index')
    print("table-%d name: %s" % (i,table_name) )

cursor.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL")
records = cursor.fetchall()

i=0
for record in records:
    print("record-%d: %s, duration = %d" % (i,record, record[1]-record[0]) )



