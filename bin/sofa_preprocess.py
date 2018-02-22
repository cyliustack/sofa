#!/usr/bin/python
import pandas as pd
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp 
import glob, os 
from functools import partial
import subprocess
from sofa_config import *
from sofa_print import *

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

def net_trace_read(packet, t_offset):
    time = float(packet.split()[0]) 
    t_begin = time + t_offset
    t_end   = time + t_offset
    if packet.split()[1] != 'IP':
        return []
    payload = int(packet.split()[6])
    duration = float(payload/125.0e6)
    bandwidth = 125.0e6
    pkt_src = int(packet.split()[2].split('.')[3]) 
    pkt_dst = int(packet.split()[4].split('.')[3])
    trace   = [	t_begin,
        	    payload*100+17,
        	    duration,   
        	    -1,
        	    -1,
        	    payload,
                bandwidth,
                pkt_src,
        	    pkt_dst,
        	    -1, 
        	    -1,
                "network:tcp:%d_to_%d_with_%d" % (pkt_src, pkt_dst, payload),
        	    0
                ]
    return trace

def gpu_trace_read(record, n_cudaproc, ts_rescale, dt_rescale, t_offset):
    idx_name = len(record.split(',')) - 1
    time =  float(record.split(',')[0])/ts_rescale + t_offset
    duration = float(record.split(',')[1])/dt_rescale
    t_begin = time
    t_end   = time + duration 
    kernel_name = "%s"%(record.split(',')[idx_name].replace('"','')[:-1])
    payload = 0 if record.split(',')[11] == '' else int(float(record.split(',')[11])*1024*1024)
    bandwidth = 1e-6 if record.split(',')[12] == '' else float(record.split(',')[12])
    pid = n_cudaproc
    deviceId = 0 if record.split(',')[14] == '' else int(record.split(',')[14].replace('"','')) 
    tid = streamId = -1 if record.split(',')[15] == '' else int(record.split(',')[15].replace('"','')) 
    pkt_src = pkt_dst = copyKind = 0
    if kernel_name.find('HtoD') != -1:
        copyKind = 1
        pkt_src = 0 
        pkt_dst = deviceId
        kernel_name = "gpu%d_copyKind_%d_%dB" % ( deviceId, copyKind, payload )
    elif kernel_name.find('DtoH') != -1:
        copyKind = 2
        pkt_src = deviceId 
        pkt_dst = 0
        kernel_name = "gpu%d_copyKind_%d_%dB" % ( deviceId, copyKind, payload )
    elif kernel_name.find('DtoD') != -1:
        copyKind = 8
        pkt_src = deviceId 
        pkt_dst = deviceId
        kernel_name = "gpu%d_copyKind_%d_%dB" % ( deviceId, copyKind, payload )
    elif kernel_name.find('PtoP') != -1:
        copyKind = 10
        pkt_src = 0 if record.split(',')[17] == '' else int(record.split(',')[17].replace('"','')) 
        pkt_dst = 0 if record.split(',')[19] == '' else int(record.split(',')[19].replace('"',''))     
        kernel_name = "gpu%d_copyKind_%d_%dB" % ( deviceId, copyKind, payload )
    else:
        copyKind = 0

    #print("%d:%d [%s] ck:%d, %lf,%lf: %d -> %d: payload:%d, bandwidth:%lf, duration:%lf "%(deviceId, streamId, kernel_name, copyKind, t_begin,t_end, pkt_src, pkt_dst, payload, bandwidth, duration))
    trace = [ t_begin,
            payload*100+17,
            duration,   
            deviceId,
            copyKind,
            payload,
            bandwidth,
            pkt_src,
            pkt_dst,
            pid, 
            tid,
            kernel_name,
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
    
    src = -1
    dst = -1
    if record[3] == 1:
        src = 0
        dst = record[4]+1
    elif record[3] == 2:
        src = record[4]+1
        dst = 0
    elif record[3] == 8:
        src = record[4]+1
        dst = record[4]+1


    trace = [	t_begin,
    		record[2],
    		float(t_end - t_begin),
    		record[4],
    		record[3],
    		record[2],
    		float(record[2])/(t_end-t_begin)/1.0e6,
    		src,
    		dst,
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
    		record[8],
    		record[9],
    		record[7], #streamId
    		-1,
    		"gpu%d_copyKind%d_%dB" % (record[4], record[3], record[2]), 
    		0]
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
            #print_info("Dump %s to JSON file"%trace.name)    
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
        'timestamp',    #0
        "event",        #1
        "duration",     #2
        "deviceId",     #3
        "copyKind",     #4
        "payload",      #5
        "bandwidth",    #6
        "pkt_src",      #7
        "pkt_dst",      #8
        "pid",          #9
        "tid",          #10
        "name",         #11
        "category"]     #12

def sofa_preprocess(logdir, cfg):
    t_glb_base = 0 
    t_glb_net_base = 0  
    t_glb_gpu_base = 0


    with open('%s/perf.script'%logdir, 'w') as logfile:
        subprocess.call(['perf', 'script', '-i', '%s/perf.data'%logdir, '-F', 'time,cpu,pid,tid,ip,sym,period'],stdout=logfile)
    #sys.stdout.flush() 
    with open(logdir + 'sofa_time.txt') as f:
        t_glb_base = float(f.readlines()[0])
        t_glb_net_base = t_glb_base 
        t_glb_gpu_base = t_glb_base 
        print t_glb_base
        print t_glb_net_base
        print t_glb_gpu_base
         
    net_traces = []
    cpu_traces = []
    mpstat_usr_traces = []
    mpstat_sys_traces = []
    mpstat_iowait_traces = []
    gpu_traces = []
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
    
    ### ============ Preprocessing CPU Trace ==========================
    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        print_info("Length of cpu_traces = %d"%len(samples))
        if len(samples) > 0:
            pool = mp.Pool(processes=cpu_count)
            t_base = float((samples[0].split())[2].split(':')[0]) 
            res = pool.map( partial(cpu_trace_read, t_offset=t_glb_base - t_base), samples)
            cpu_traces = pd.DataFrame(res)
            cpu_traces.columns = sofa_fieldnames 
            cpu_traces.to_csv(logdir + 'cputrace.csv', mode='w', header=True, index=False,float_format='%.6f')        

    ### ============ Preprocessing MPSTAT Trace ==========================
    with open('%s/mpstat.txt'%logdir) as f:
        lines = f.readlines()[1:]
        print_info("Length of mpstat_traces = %d"%len(lines))
        if len(lines) > 0: 
            usr_list = []
            sys_list = []
            iowait_list = []
            usr_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            sys_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            iowait_list.append(np.empty((len(sofa_fieldnames), 0)).tolist())
            t_base = t = 0
            #mprec = np.zeros((len(lines),11))
            for i in xrange(len(lines)):
                #print(lines[i])
                if len(lines[i].split())>2 :
                    cpuid = lines[i].split()[2]
                    if cpuid != 'CPU':
                        if cpuid == 'all':
                        	event = -1 
			else:
				try:
				    event = int(cpuid)
				except ValueError:
                        		continue
			mpst_usr=float(lines[i].split()[3])
                        mpst_sys=float(lines[i].split()[5])
                        mpst_iowait=float(lines[i].split()[6])
                        t_begin = t - t_base + t_glb_base
                        duration = mpst_usr
                        deviceId = cpuid
                        copyKind = -1
                        payload = mpst_sys
                        bandwidth = mpst_iowait
                        pkt_src = pkt_dst = -1
                        pid = tid = -1
                        trace   = [	t_begin,
            	                    event,
            	                    mpst_usr,   
            	                    deviceId,
            	                    0,
            	                    payload,
                                    bandwidth,
                                    pkt_src,
            	                    pkt_dst,
            	                    pid, 
            	                    tid,
                                    "mpstat_usr[%s]=%.1lf" % (cpuid, mpst_usr),
            	                    cpuid
                                    ]
                        usr_list.append(trace)
                        
                        trace   = [	t_begin,
            	                    event,
            	                    mpst_sys,   
            	                    deviceId,
            	                    1,
            	                    payload,
                                    bandwidth,
                                    pkt_src,
            	                    pkt_dst,
            	                    pid, 
            	                    tid,
                                    "mpstat_sys[%s]=%.1lf" % (cpuid, mpst_sys),
            	                    cpuid
                                    ]
                        sys_list.append(trace)

                        trace   = [	t_begin,
            	                    event,
            	                    mpst_iowait,   
            	                    deviceId,
            	                    2,
            	                    payload,
                                    bandwidth,
                                    pkt_src,
            	                    pkt_dst,
            	                    pid, 
            	                    tid,
                                    "mpstat_iowait[%s]=%.1lf" % (cpuid, mpst_iowait),
            	                    cpuid
                                    ]
                        iowait_list.append(trace)
                else:
                    t = t + 1
            mpstat_usr_traces = pd.DataFrame(usr_list[1:])
            mpstat_usr_traces.columns = sofa_fieldnames 
            mpstat_usr_traces.to_csv(logdir + 'mpstat_trace.csv', mode='w', header=True, index=False, float_format='%.6f')
            
            mpstat_sys_traces = pd.DataFrame(sys_list[1:])
            mpstat_sys_traces.columns = sofa_fieldnames 
            mpstat_sys_traces.to_csv(logdir + 'mpstat_trace.csv', mode='a', header=False, index=False, float_format='%.6f')
            
            mpstat_iowait_traces = pd.DataFrame(iowait_list[1:])
            mpstat_iowait_traces.columns = sofa_fieldnames 
            mpstat_iowait_traces.to_csv(logdir + 'mpstat_trace.csv', mode='a', header=False, index=False, float_format='%.6f')
    


    #Linux 4.4.0-81-generic (ubuntu1404) 	02/06/2018 	_x86_64_	(40 CPU)
    #
    #06:32:01 AM  CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
    #06:32:02 AM  all    0.15    0.00    0.59    0.02    0.00    0.02    0.00    0.00    0.00   99.21
    #06:32:02 AM    0    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00  100.00
    
    
    #TODO: align cpu time and gpu time
    t_nv = sys.float_info.max
    for i in xrange(len(cpu_traces)):
        #print("name:%s"%cpu_traces.loc[i,'name'])
        if cpu_traces.iat[i,11].find('nv_alloc_system_pages')!=-1 and float(cpu_traces.iat[i,0]) < t_nv:
            t_nv = float(cpu_traces.iat[i,0]) 
    if t_nv < sys.maxint:
        t_glb_gpu_base = t_nv + 0.1
        print("t_base: cpu=%lf gpu=%lf" % ( t_glb_base, t_glb_gpu_base))
    
    # Apply filters for cpu traces
    df_grouped = cpu_traces.groupby('name')
    filtered_groups = []
    color_of_filtered_group = []
    #e.g. cpu_trace_filters = [ {"keyword":"nv_", "color":"red"}, {"keyword":"idle", "color":"green"} ]
    cpu_trace_filters = cfg['filters'] 
    if len(cpu_traces) > 0:
        for cpu_trace_filter in cpu_trace_filters:
            group = cpu_traces[ cpu_traces['name'].str.contains(cpu_trace_filter['keyword'])]
            filtered_groups.append({'group':group,'color':cpu_trace_filter['color'], 'keyword':cpu_trace_filter['keyword']})
   

    ### ============ Preprocessing Network Trace ==========================
    os.system("tcpdump -q -n -tt -r " + logdir + "sofa.pcap" + " > " + logdir+ "net.tmp" ) 
    with open(logdir + 'net.tmp') as f:
        packets = lines = f.readlines()
        print_info("Length of net_traces = %d"%len(packets))
        if len(packets) > 0 :
            t_base = float(lines[0].split()[0]) 
    	    pool = mp.Pool(processes=cpu_count)
            res = pool.map( partial(net_trace_read, t_offset=t_glb_net_base - t_base), packets)
            net_traces = pd.DataFrame(res)
            net_traces.columns = sofa_fieldnames
            net_traces.to_csv(logdir + 'cputrace.csv', mode='a', header=False, index=False, float_format='%.6f')

    ### ============ Preprocessing GPU Trace ==========================
    num_cudaproc = 0
    filtered_gpu_groups = []
    for nvvp_filename in glob.glob(logdir+"gputrace*[0-9].nvvp"):
        print_progress("Read "+nvvp_filename+" -- begin")
        os.system("nvprof --csv --print-gpu-trace -i " + nvvp_filename + " 2> " + logdir+ "gputrace.tmp" ) 
        print_progress("Read "+nvvp_filename+" -- end")
        num_cudaproc = num_cudaproc + 1 
        with open(logdir + 'gputrace.tmp') as f:
            records = f.readlines()
            #ms,ms,,,,,,,,B,B,MB,GB/s,,,,
            ts_rescale = 1.0
            if records[2].split(',')[0] == 'ms':
               ts_rescale = 1.0e3 
            elif records[2].split(',')[0] == 'us':
               ts_rescale = 1.0e6 

            dt_rescale = 1.0
            if records[2].split(',')[1] == 'ms':
               dt_rescale = 1.0e3 
            elif records[2].split(',')[1] == 'us':
               dt_rescale = 1.0e6 

            records = records[3:]
            print_info("Length of gpu_traces = %d"%len(records))
            t_base = float(records[0].split(',')[0]) 
            t_offset = t_glb_gpu_base - t_base
            pool = mp.Pool(processes=cpu_count)
            res = pool.map( partial(gpu_trace_read, ts_rescale=ts_rescale, dt_rescale=dt_rescale, n_cudaproc=num_cudaproc, t_offset=t_glb_gpu_base - t_base), records)
            gpu_traces = pd.DataFrame(res)
            gpu_traces.columns = sofa_fieldnames 
            gpu_traces.to_csv(logdir + 'gputrace.csv', mode='w', header=True, index=False, float_format='%.6f')
            
            # Apply filters for cpu traces
            df_grouped = gpu_traces.groupby('name')
            color_of_filtered_group = []
            #e.g. cpu_trace_filters = [ {"keyword":"nv_", "color":"red"}, {"keyword":"idle", "color":"green"} ]
            gpu_trace_filters = cfg['gpu_filters'] 
            for gpu_trace_filter in gpu_trace_filters:
                group = gpu_traces[ gpu_traces['name'].str.contains(gpu_trace_filter['keyword'])]
                filtered_gpu_groups.append({'group':group,'color':gpu_trace_filter['color'], 'keyword':gpu_trace_filter['keyword']})


    print_progress("Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- begin")

    
    #TODO: provide option to use absolute or relative timestamp
    #cpu_traces.loc[:,'timestamp'] -= cpu_traces.loc[0,'timestamp']
    #net_traces.loc[:,'timestamp'] -= net_traces.loc[0,'timestamp']
    #gpu_traces.loc[:,'timestamp'] -= gpu_traces.loc[0,'timestamp']

    traces = []
    sofatrace = SOFATrace()
    sofatrace.name = 'cpu_trace'
    sofatrace.title = 'CPU'
    sofatrace.color = 'DarkGray'
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
    sofatrace.name = 'mpstat_usr'
    sofatrace.title = 'MPSTAT_USR'
    sofatrace.color = 'LightBlue'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = mpstat_usr_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'mpstat_sys'
    sofatrace.title = 'MPSTAT_SYS'
    sofatrace.color = 'LightCoral'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = mpstat_sys_traces
    traces.append(sofatrace)

    sofatrace = SOFATrace()
    sofatrace.name = 'mpstat_iowait'
    sofatrace.title = 'MPSTAT_IOWAIT'
    sofatrace.color = 'LightSeaGreen'
    sofatrace.x_field = 'timestamp'
    sofatrace.y_field = 'duration'
    sofatrace.data = mpstat_iowait_traces
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
    sofatrace.data = gpu_traces
    traces.append(sofatrace)

    for filtered_gpu_group in filtered_gpu_groups:
        sofatrace = SOFATrace()
        sofatrace.name = filtered_gpu_group['keyword']
        sofatrace.title = 'keyword_'+sofatrace.name
        sofatrace.color = filtered_gpu_group['color']
        sofatrace.x_field = 'timestamp'
        sofatrace.y_field = 'duration'
        sofatrace.data = filtered_gpu_group['group'].copy()
        traces.append(sofatrace)

    traces_to_json(traces, logdir+'report.js')
    print_progress("Export Overhead Dynamics JSON File of CPU, Network and GPU traces -- end")
