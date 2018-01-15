#!/usr/bin/python
from scapy.all import *
import sqlite3
import pandas as pd
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp 
import glob, os 
from functools import partial
from sofa_config import *
from sofa_print import *

def sqlite3_read_tables(filename):

    return tables


def nvprof_db_read(logdir):
    db = [] 
    for sqlite_filename in glob.glob(logdir+"gputrace*[0-9].nvp"):
        db = sqlite3.connect(sqlite_filename)
	print("Merging %s" % sqlite_filename)
    return 0

def cpu_trace_read(sample, t_offset):
    fields = sample.split()
    time = float(fields[2].split(':')[0])
    func_name = fields[5]
    t_begin = time + t_offset
    t_end = time  + t_offset

    trace = [t_begin,
    	np.log(int("0x" + fields[4], 16)), #% 1000000
    	float(fields[3]) / 1.5e9,  
    	int(fields[1].split('[')[1].split(']')[0]),
    	-1,
    	0,
    	0,
    	-1,
    	-1,
    	int(fields[0].split('/')[0]),    
    	int(fields[0].split('/')[1]),
    	fields[5], 
    	0]
     
    return trace

def gpu_kernel_trace_read(record, pid, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end   = (record[1] - t_base) / 1e9 + t_glb_base
    kernel_name = "%s"%(gpu_symbol_table.loc[gpu_symbol_table._id_ == record[2], 'value'])
    kernel_name = kernel_name[:-30]
    trace = [	t_begin,
		record[2],
		float(t_end - t_begin),
		record[4],
		-1,
		0,
		0.0,
		-1,
		-1,
                pid,
		record[3],
		kernel_name,
		0]
    return trace

def gpu_memcpy_trace_read(record, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end   = (record[1] - t_base) / 1e9 + t_glb_base
    trace = [	t_begin,
    		record[2],
    		float(t_end - t_begin),
    		record[4],
    		record[3],
    		record[2],
    		float(record[2])/(t_end-t_begin)/1.0e6,
    		-1,
    		-1,
    		record[7], #streamId
    		-1,
    		"gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]), 
    		0]
    return trace

def gpu_memcpy2_trace_read(record, t_base, t_glb_base):
    t_begin = (record[0] - t_base) / 1e9 + t_glb_base
    t_end   = (record[1] - t_base) / 1e9 + t_glb_base
    trace = [	t_begin,
    		record[2],
    		float(t_end - t_begin),
    		record[4],
    		record[3],
    		record[2],
    		float(record[2])/(t_end-t_begin)/1.0e6,
    		-1,
    		-1,
    		record[7], #streamId
    		-1,
    		"gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]), 
    		0]
    return trace

def net_trace_read(packet, t_offset):
    trace = []
    try:
        time = packet[IP].time
        t_begin = time + t_offset
        t_end = time  + t_offset    
        payload = packet.len
        pkt_src = packet[IP].src.split('.')[3]
        pkt_dst = packet[IP].dst.split('.')[3]                
        trace = [   t_begin,
       	            payload*100+17,
       	            payload/125.0e6,   
       	            -1,
       	            -1,
       	            payload,
       	            125.0e6,
       	            pkt_src,
       	            pkt_dst,
       	            -1, 
       	            -1,
                    "network:tcp:%s_to_%s_with_%d" % (pkt_src, pkt_dst, payload),
       	            0]
	return trace
    except Exception as e:
        print(e)
	return trace

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
    print(bcolors.OKBLUE + "[PROGRESS] " + content + bcolors.ENDC)

class SOFATrace:
    data = []
    name = []
    title = []
    color = []
    x_field = []
    y_field = []

### traces_to_json() 
#    Please set "turboThreshold:0" in PlotOptions of ScatterChart to unlimit number of points 
#    series: [{
#        name: 'Female',
#        color: 'rgba(223, 83, 83, .5)',
#        data: [{x:161.2, y:51.6, name:'b'}, {x:167.5, y:59.0, name:"a"}]
#    },
#    {
#        name: 'Male',
#        color: 'rgba(23, 83, 183, .5)',
#        data: [{x:16.2, y:151.6, name:'bd'}, {x:67.5, y:59.0, name:"ad"}]
#    }
#    ]

def traces_to_json(traces, path):
    if len(traces) == 0:
        print_warning("Empty traces!")
        return 
    
    with open(path, 'w') as f:
        for trace in traces:
            print_info("Dump %s to JSON file"%trace.name)    
            if len(trace.data) > 0:
                f.write(trace.name+" = ")
                trace.data.rename(columns={trace.x_field:'x', trace.y_field:'y'}, inplace=True)
                sofa_series = { "name": trace.title,
                                    "color": trace.color,
                                    "data": json.loads(trace.data.to_json(orient='records'))
                                    }
                json.dump(sofa_series, f)
                trace.data.rename(columns={'x':trace.x_field, 'y':trace.y_field}, inplace=True)
            f.write("\n\n")  
        
        f.write("sofa_traces = [ ")
        for trace in traces :
            if len(trace.data) > 0:
                f.write(trace.name+",")
        f.write(" ]")        
            
sofa_fieldnames = [
        'timestamp',
        "event",
        "duration",
        "deviceId",
        "copyKind",
        "payload",
        "bandwidth",
        "pkt_src",
        "pkt_dst",
        "pid",
        "tid",
        "name",
        "category"]

if __name__ == "__main__":
    
 
    parser = argparse.ArgumentParser(description='SOFA Preprocessing')
    parser.add_argument("--logdir", metavar="/path/to/logdir/", type=str, required=True, 
                    help='path to the directory of SOFA logged files')
    parser.add_argument('--config', metavar="/path/to/config.cfg", type=str, required=True,
                    help='path to the directory of SOFA configuration file')

    args = parser.parse_args()
    logdir = args.logdir + "/"

    cfg = read_config(args.config)

    with open(logdir + 'sofa_time.txt') as f:
        t_glb_base = float(f.readlines()[0])
        t_glb_net_base = t_glb_base + 0.5
        t_glb_gpu_base = t_glb_base 
        print t_glb_base
        print t_glb_net_base
        print t_glb_gpu_base
         
    net_traces = []
    cpu_traces = []
    gpu_kernel_traces = []
    gpu_memcpy_traces = []
    gpu_memcpy2_traces = []
    gpu_memcpy_h2d_traces = []
    gpu_memcpy_d2h_traces = []
    gpu_memcpy_d2d_traces = []
    gpu_glb_kernel_traces = []
    gpu_glb_memcpy_traces = []
    gpu_glb_memcpy2_traces = []
    gpu_glb_memcpy_h2d_traces = []
    gpu_glb_memcpy_d2h_traces = []
    gpu_glb_memcpy_d2d_traces = []

    gpulog_mode = 'w'
    gpulog_header = 'True'
    cpu_count = mp.cpu_count()
    
    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        t_base = 0
        	
        pool = mp.Pool(processes=cpu_count)
	t_base = float((samples[0].split())[2].split(':')[0]) 
	res = pool.map( partial(cpu_trace_read, t_offset=t_glb_base - t_base), samples)
	cpu_traces = pd.DataFrame(res)
        cpu_traces.columns = sofa_fieldnames 
        print_info("Length of cpu_traces = %d"%len(cpu_traces))
        #print res
	#for i in range(0, len(cpu_traces)):
        #    fields = samples[i].split()
        #    time = float(fields[2].split(':')[0])
        #    if i == 0:
        #        t_base = time
        #    func_name = fields[5]
        #    t_begin = time - t_base + t_glb_base
        #    t_end = time - t_base + t_glb_base

        #    cpu_traces.at[i] = [t_begin,
        #    			np.log(int("0x" + fields[4], 16)), #% 1000000
        #    			float(fields[3]) / 1.5e9,  
        #    			int(fields[1].split('[')[1].split(']')[0]),
        #    			-1,
        #    			0,
        #    			0,
        #    			-1,
        #    			-1,
        #    			int(fields[0].split('/')[0]),    
        #    			int(fields[0].split('/')[1]),
        #    			fields[5], 
        #    			0]
        #    
        #    if (cpu_traces.loc[i,'name'] == "testHostToDeviceTransfer"):
        #        t_glb_gpu_base = float(cpu_traces.at[i,'timestamp'])

        cpu_traces.to_csv(logdir + 'cputrace.csv', mode='w', header=True, index=False,float_format='%.6f')        


    df_grouped = cpu_traces.groupby('name')
    filtered_groups = []
    color_of_filtered_group = []
    #e.g. cpu_trace_filters = [ {"keyword":"nv_", "color":"red"}, {"keyword":"idle", "color":"green"} ]
    cpu_trace_filters = cfg['filters'] 
    for cpu_trace_filter in cpu_trace_filters:
        group = cpu_traces[ cpu_traces['name'].str.contains(cpu_trace_filter['keyword'], re.IGNORECASE)]
        filtered_groups.append({'group':group,'color':cpu_trace_filter['color'], 'keyword':cpu_trace_filter['keyword']})

    for i in range(0, len(cpu_traces)):
        if (cpu_traces.loc[i,'name'] == "testHostToDeviceTransfer"):
            t_glb_gpu_base = float(cpu_traces.at[i,'timestamp'])


    
    with open(logdir + 'sofa.pcap','r') as f_pcap:
        packets = rdpcap(logdir + 'sofa.pcap')
        net_traces = pd.DataFrame(pd.np.empty((len(packets), len(sofa_fieldnames))) * pd.np.nan)
        net_traces.columns = sofa_fieldnames 
        print_info("Length of net_traces = %d"%len(net_traces))

        #pool = mp.Pool(processes=cpu_count)
	#t_base = packets[0][IP].time 
	#res = pool.map( partial(net_trace_read, t_offset=t_glb_net_base - t_base), packets)
	#net_traces = pd.DataFrame(res)
        #net_traces.columns = sofa_fieldnames
        for i in range(0, len(net_traces)):
            try:
                time = packets[i][IP].time
                if i == 0:
                    t_base = time
                t_begin = (time - t_base) + t_glb_net_base
                t_end = (time - t_base) + t_glb_net_base
                payload = packets[i].len
                pkt_src = packets[i][IP].src.split('.')[3]
                pkt_dst = packets[i][IP].dst.split('.')[3]                
                net_traces.at[i] = [	t_begin,
                			payload*100+17,
                			payload/125.0e6,   
                			-1,
                			-1,
                			payload,
                			125.0e6,
                			pkt_src,
                			pkt_dst,
                			-1, 
                			-1,
                                        "network:tcp:%s_to_%s_with_%d" % (pkt_src, pkt_dst, payload),
                			0]
            except Exception as e:
                print(e)
        net_traces.to_csv(logdir + 'cputrace.csv', mode='a', header=False, index=False, float_format='%.6f')        
    

    ### ============ Preprocessing GPU Trace ==========================
    print_progress("read and csv-transform nvprof traces -- begin")
    num_cudaproc = 0
    for sqlite_filename in glob.glob(logdir+"gputrace*[0-9].nvp"):
	
 	print("Merging %s" % sqlite_filename)
	db = sqlite3.connect(sqlite_filename)
	cursor = db.cursor()
    	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    	tables = cursor.fetchall()
    	gpu_symbol_table = []
    	for table_name in tables:
    	    tname = table_name[0]
    	    table = pd.read_sql_query("SELECT * from %s" % tname, db)
    	    # print("table-%d = %s, count=%d" % (i,tname,len(table.index)) )
    	    if len(table.index) > 0:
    	        table.to_csv(logdir + ("CUDAPROC%d_"%(num_cudaproc)) + tname + '.csv', index_label='index')
    	    if tname == "StringTable":
    	        gpu_symbol_table = table
    	print_progress("read and csv-transform nvprof traces -- end")
    	
    	
    	print_progress("query CUDA kernels -- begin")
    	try:
    	    cursor.execute(
    	        "SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
    	except sqlite3.OperationalError:
    	    try:
    	        cursor.execute("SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    	    except sqlite3.OperationalError:
    	        print_info("No GPU traces were collected.")
    	print_progress("query CUDA kernels -- end")
    	
    	#with open(logdir+"gputrace.csv",mode='w') as f:
    	#    print_info("Create new gputrace.csv")    
    	
    	if  os.path.exists(logdir+"CUDAPROC%d_CUPTI_ACTIVITY_KIND_KERNEL.csv"%(num_cudaproc) ) or os.path.exists(logdir+"CUDAPROC%d_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.csv"%num_cudaproc ) : 
    	    print_progress("export-csv CUDA kernels -- begin")
    	    
    	    gpu_kernel_records = cursor.fetchall()
    	    gpu_kernel_traces = pd.DataFrame(pd.np.empty((len(gpu_kernel_records), len(sofa_fieldnames))) * pd.np.nan) 
    	    gpu_kernel_traces.columns = sofa_fieldnames 
    	    print_info("Length of gpu_kernel_traces = %d"%len(gpu_kernel_traces))
    	   	
    	    pool = mp.Pool(processes=cpu_count)
    	    
    	    record = gpu_kernel_records[0]
    	    t_base = float(record[0])
    	    res = pool.map( partial(gpu_kernel_trace_read, pid=num_cudaproc, t_base=t_base, t_glb_base=t_glb_gpu_base), gpu_kernel_records)
    	    gpu_kernel_traces = pd.DataFrame(res)
    	    gpu_kernel_traces.columns = sofa_fieldnames
    	    gpu_kernel_traces.to_csv(logdir + 'gputrace.csv', mode=gpulog_mode, header=gpulog_header, index=False,float_format='%.6f')
    	    gpulog_mode = 'a'
    	    gpulog_header = False
            gpu_glb_kernel_traces.append(gpu_kernel_traces)
    	    print_progress("export-csv CUDA kernels -- end")

 
    	print_progress("query CUDA concurrent kernels -- begin")
    	try:
    	    cursor.execute(
    	        "SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL")
    	except sqlite3.OperationalError:
    	    try:
    	        cursor.execute("SELECT start,end,name,streamId,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL")
    	    except sqlite3.OperationalError:
    	        print_info("No GPU traces were collected.")
    	print_progress("query CUDA concurrent kernels -- end")
    	
   
    	print_progress("query CUDA memcpy (h2d,d2h,d2d) -- begin")
    	try:
    	    # index,_id_,copyKind,srcKind,dstKind,flags,bytes,start,end,deviceId,contextId,streamId,correlationId,runtimeCorrelationId
    	    cursor.execute(
    	        "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    	    gpu_memcpy_records = cursor.fetchall() 
    	except sqlite3.OperationalError:
    	        print_info("No GPU MEMCPY traces were collected.")
    	print_progress("query CUDA memcpy (h2d,d2h,d2d) -- end")
   
    	
    	print_progress("export-csv CUDA memcpy (h2d,d2h,d2d) -- begin")
    	if  os.path.exists(logdir+"CUDAPROC%d_CUPTI_ACTIVITY_KIND_MEMCPY.csv"%num_cudaproc ) : 
    	    gpu_memcpy_traces = pd.DataFrame(pd.np.empty((len(gpu_memcpy_records), len(sofa_fieldnames))) * pd.np.nan) 
    	    gpu_memcpy_traces.columns = sofa_fieldnames 
    	    t_base = 0
    	    print_info("Length of gpu_memcpy_traces = %d"%len(gpu_memcpy_traces))
    	    
    	    record = gpu_memcpy_records[0]
    	    t_base = float(record[0])
    	    res = pool.map( partial(gpu_memcpy_trace_read, t_base=t_base, t_glb_base=t_glb_gpu_base), gpu_memcpy_records)
    	    gpu_memcpy_traces = pd.DataFrame(res)
    	    gpu_memcpy_traces.columns = sofa_fieldnames
    	    #for i in range(0, len(gpu_memcpy_traces)):
    	    #    record = gpu_memcpy_records[i]
    	    #    if i == 0:
    	    #        t_base = record[0]
   
    	    #    t_begin = (record[0] - t_base) / 1e9 + t_glb_gpu_base
    	    #    t_end   = (record[1] - t_base) / 1e9 + t_glb_gpu_base
    	    #    gpu_memcpy_traces.at[i] = [ t_begin,
    	    #    				record[2],
    	    #    				float(t_end - t_begin),
    	    #    				record[4],
    	    #    				record[3],
    	    #    				record[2],
    	    #    				float(record[2])/(t_end-t_begin)/1.0e6,
    	    #    				-1,
    	    #    				-1,
    	    #    				record[7], #streamId
    	    #    				-1,
    	    #    				"gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]), 
    	    #    				0]
    	    gpu_memcpy_traces.to_csv(logdir + 'gputrace.csv', mode=gpulog_mode, header=gpulog_header, index=False, float_format='%.6f')
    	    gpu_glb_memcpy_traces.append(gpu_memcpy_traces)
            gpulog_mode = 'a'
    	    gpulog_header = False
    	    if len(gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 1)]) > 0:
    	        gpu_memcpy_h2d_traces = gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 1)].copy()
    	    
    	    if len(gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 2)]) > 0:
    	        gpu_memcpy_d2h_traces = gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 2)].copy()
    	    
    	    if len(gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 8)]) > 0:
    	        gpu_memcpy_d2d_traces = gpu_memcpy_traces[(gpu_memcpy_traces.copyKind == 8)].copy()
            
    	print_progress("export-csv CUDA memcpy (h2d,d2h,d2d) -- end")
 
    	
    	print_progress("query CUDA memcpy2 (p2p) -- begin")
    	try:
    	    cursor.execute(
    	        "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY2")
    	    gpu_memcpy2_records = cursor.fetchall()
    	except sqlite3.OperationalError:
    	    print_info("No GPU MEMCPY2 traces were collected.")
    	print_progress("query CUDA memcpy2 (p2p) -- end")
   
    	print_progress("export-csv CUDA memcpy2 (p2p) -- begin")
    	if  os.path.exists(logdir+"CUDAPROC%d_CUPTI_ACTIVITY_KIND_MEMCPY2.csv"%num_cudaproc ) : 
    	    gpu_memcpy2_traces = pd.DataFrame(pd.np.empty((len(gpu_memcpy2_records), len(sofa_fieldnames))) * pd.np.nan) 
    	    gpu_memcpy2_traces.columns = sofa_fieldnames 
    	    t_base = 0
    	    print_info("Length of gpu_memcpy2_traces = %d"%len(gpu_memcpy2_traces))
    	    
    	    record = gpu_memcpy2_records[0]
    	    t_base = float(record[0])
    	    res = pool.map( partial(gpu_memcpy2_trace_read, t_base=t_base, t_glb_base=t_glb_gpu_base), gpu_memcpy2_records)
    	    gpu_memcpy2_traces = pd.DataFrame(res)
    	    gpu_memcpy2_traces.columns = sofa_fieldnames
    	    #for i in range(0, len(gpu_memcpy2_traces)):
    	    #    record = gpu_memcpy2_records[i]
    	    #    if i == 0:
    	    #        t_base = record[0]
   
    	    #    t_begin = (record[0] - t_base) / 1e9 + t_glb_gpu_base
    	    #    t_end   = (record[1] - t_base) / 1e9 + t_glb_gpu_base
    	    #    gpu_memcpy2_traces.at[i] = [t_begin,
    	    #    				record[2], 
    	    #    				float(t_end - t_begin),
    	    #    				record[4],
    	    #    				record[3],
    	    #    				record[2],
    	    #    				record[2]/float((t_end-t_begin))/1.0e6,
    	    #    				-1,
    	    #    				-1,
    	    #    				record[7], #streamId
    	    #    				-1,
    	    #    				"gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]),
    	    #    				0]
    	    gpu_memcpy2_traces.to_csv(logdir + 'gputrace.csv', mode=gpulog_mode, header=gpulog_header, index=False, float_format='%.6f')
    	    gpulog_mode = 'a'
    	    gpulog_header = False
            gpu_glb_memcpy2_traces.append(gpu_memcpy2_traces)
    	print_progress("export-csv CUDA memcpy2 (h2d,d2h,d2d) -- end")
    	
	num_cudaproc = num_cudaproc + 1  # End of reading NVPROF SQLite3 Database
    

    print_progress("Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- begin")

    traces = []
    sofatrace = SOFATrace()
    sofatrace.name = 'cpu_trace'
    sofatrace.title = 'CPU'
    sofatrace.color = 'orange'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = cpu_traces
    traces.append(sofatrace)

    for filtered_group in filtered_groups:
        sofatrace = SOFATrace()
        sofatrace.name = filtered_group['keyword']
        sofatrace.title = 'keyword_'+sofatrace.name
        sofatrace.color = filtered_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_group['group'].copy()
        traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'net_trace'
    sofatrace.title = 'NET'
    sofatrace.color = 'blue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = net_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_kernel_trace'
    sofatrace.title = 'GPU kernel'
    sofatrace.color = 'rgba(0,180,0,0.8)'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_kernel_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_memcpy_h2d_trace'
    sofatrace.title = 'GPU memcpy (H2D)'
    sofatrace.color = 'red'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_memcpy_h2d_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_memcpy_d2h_trace'
    sofatrace.title = 'GPU memcpy (D2H)'
    sofatrace.color = 'greenyellow'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_memcpy_d2h_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_memcpy_d2d_trace'
    sofatrace.title = 'GPU memcpy (D2D)'
    sofatrace.color = 'darkblue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_memcpy_d2d_traces
    traces.append(sofatrace)

    if cfg['enable_plot_bandwidth'] == 'true':
        sofatrace = SOFATrace()
        sofatrace.name = 'gpu_memcpy_h2d_bw_trace'
        sofatrace.title = 'GPU memcpy H2D bandwidth (MB/s)'
        sofatrace.color = 'Crimson'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'bandwidth'
        sofatrace.data = gpu_memcpy_h2d_traces
        traces.append(sofatrace)

        sofatrace = SOFATrace()
        sofatrace.name = 'gpu_memcpy_d2h_bw_trace'
        sofatrace.title = 'GPU memcpy D2H bandwidth (MB/s)'
        sofatrace.color = 'DarkOliveGreen'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'bandwidth'
        sofatrace.data = gpu_memcpy_d2h_traces
        traces.append(sofatrace)

        sofatrace = SOFATrace()
        sofatrace.name = 'gpu_memcpy_d2d_bw_trace'
        sofatrace.title = 'GPU memcpy D2D bandwidth (MB/s)'
        sofatrace.color = 'DarkMagenta'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'bandwidth'
        sofatrace.data = gpu_memcpy_d2d_traces
        traces.append(sofatrace)


    sofatrace = SOFATrace()
    sofatrace.name = 'gpu_memcpy2_trace'
    sofatrace.title = 'GPU memcpy2'
    sofatrace.color = 'brown'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = gpu_memcpy2_traces
    traces.append(sofatrace)

    if cfg['enable_plot_bandwidth'] == 'true':
        sofatrace = SOFATrace()
        sofatrace.name = 'gpu_memcpy2_bw_trace'
        sofatrace.title = 'GPU memcpy2 bandwidth (MB/s)'
        sofatrace.color = 'DarkSeaGreen'
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'bandwidth'
        sofatrace.data = gpu_memcpy2_traces
        traces.append(sofatrace)

    traces_to_json(traces, logdir+'report.js')
    traces_to_json(traces, logdir+'overhead.js')
    print_progress("Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- end")
