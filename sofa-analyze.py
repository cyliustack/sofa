from scapy.all import *

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
