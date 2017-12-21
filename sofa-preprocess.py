#!/usr/bin/python
from scapy.all import *
import sqlite3
import pandas as pd
import numpy as np
import csv
import cxxfilt
import json
import sys

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
    print(bcolors.OKBLUE + "[INFO] " + content + bcolors.ENDC)

def traces_to_json(traces, x_field, y_field, name, color, path, mode):
    print_info("Dump %s traces to JSON file"%name)    
    

if __name__ == "__main__":

    sys.stdout.flush()
    print('Argument List: %s', str(sys.argv))
    logdir = []
    filein = []
    
    if len(sys.argv) < 2:
        print_info("Usage: sofa-preproc.py /path/to/logdir")
        quit()
    else:
        logdir = sys.argv[1] + "/"
        filein = logdir + "gputrace.nvp"
    
    #class CPUTrace:
    #    fieldnames = [
    #        'time',
    #        "event",
    #        "duration",
    #        "deviceId",
    #        "pid",
    #        "tid",
    #        "data",
    #        "pkt_src",
    #        "pkt_dst"]
    #    time = 0
    #    duration = 0
    #    event = -1
    #    name = "none"
    #    vaddr = -1
    #    deviceId = -1
    #    pid = -1
    #    tid = -1
    #    data = 0
    #    pkt_src = -1
    #    pkt_dst = -1
    
    with open(logdir + 'sofa_time.txt') as f:
        t_glb_base = float(f.readlines()[0])
        print t_glb_base
    
    sofa_fieldnames = [
        'timestamp',
        "event",
        "duration",
        "deviceId",
        "copyKind",
        "payload",
        "pkt_src",
        "pkt_dst",
        "pid",
        "tid",
        "name",
        "category"]

    iptable = []
    series = []
    net_traces = []
    cpu_traces = []
    gpu_kernel_traces = []
    gpu_memcpy_traces = []
    gpu_memcpy2_traces = []
    t_base = 0

    with open(logdir + 'perf.script') as f:
        samples = f.readlines()
        t_base = 0
        cpu_traces = pd.DataFrame(pd.np.empty((len(samples), len(sofa_fieldnames))) * pd.np.nan) 
        cpu_traces.columns = sofa_fieldnames 
        print_info("Length of cpu_traces = %d"%len(cpu_traces))
        for i in range(0, len(cpu_traces)):
            fields = samples[i].split()
            time = float(fields[2].split(':')[0])
            if i == 0:
                t_base = time
            func_name = fields[5]
            t_begin = time - t_base + t_glb_base
            t_end = time - t_base + t_glb_base
            cpu_traces.loc[i,'timestamp']   =   t_begin
            cpu_traces.loc[i,'event']       =   vaddr = int("0x" + fields[4], 16) % 1000000
            cpu_traces.loc[i,'duration']    =   float(fields[3]) / 1.5e9  
            cpu_traces.loc[i,'deviceId']    =    int(fields[1].split('[')[1].split(']')[0])
            cpu_traces.loc[i,'copyKind']    =   -1
            cpu_traces.loc[i,'payload']     =   0
            cpu_traces.loc[i,'pkt_src']     =   -1 
            cpu_traces.loc[i,'pkt_dst']     =   -1
            cpu_traces.loc[i,'pid']         =   int(fields[0].split('/')[0])     
            cpu_traces.loc[i,'tid']         =   int(fields[0].split('/')[1])
            cpu_traces.loc[i,'name']        =   fields[5].replace("[", "_").replace("]", "_")
            cpu_traces.loc[i,'category']    =   0
        series.append(
        {   "name" : 'CPU',
            "color": 'rgba(255, 0, 0, .5)',
            "data" : cpu_traces})
        cpu_traces.to_csv(logdir + 'cputrace.csv', mode='a', header=False)        


    with open(logdir + 'sofa.pcap','r') as f_pcap:
        packets = rdpcap(logdir + 'sofa.pcap')
        net_traces = pd.DataFrame(pd.np.empty((len(packets), len(sofa_fieldnames))) * pd.np.nan)
        net_traces.columns = sofa_fieldnames 
        print_info("Length of net_traces = %d"%len(net_traces))
        for i in range(0, len(net_traces)):
            try:
                time = packets[i][IP].time
                if i == 0:
                    t_base = time
                t_begin = (time - t_base) + t_glb_base
                t_end = (time - t_base) + t_glb_base
                payload = packets[i].len
                pkt_src = packets[i][IP].src.split('.')[3]
                pkt_dst = packets[i][IP].dst.split('.')[3]                
                net_traces.loc[i,'timestamp']   =   t_begin
                net_traces.loc[i,'event']       =   payload*100+17
                net_traces.loc[i,'duration']    =   payload/125.0e6   
                net_traces.loc[i,'deviceId']    =   -1
                net_traces.loc[i,'copyKind']    =   -1
                net_traces.loc[i,'payload']     =   payload
                net_traces.loc[i,'pkt_src']     =   pkt_src 
                net_traces.loc[i,'pkt_dst']     =   pkt_dst
                net_traces.loc[i,'pid']         =   -1  
                net_traces.loc[i,'tid']         =   -1
                net_traces.loc[i,'name']        =   "%s_to_%s_with_%d" % (pkt_src, pkt_dst, payload)
                net_traces.loc[i,'category']    =   0
            except Exception as e:
                print(e)
        series.append(
            {"name": 'Network',
                     "color": 'rgba(0, 255, 0, .5)',
                     "data": net_traces})
        net_traces.to_csv(logdir+'cputrace.csv')
    
    

    ### ============ Preprocessing GPU Trace ==========================
    print_progress("read nvprof traces -- begin")
    sqlite_file = filein
    db = sqlite3.connect(sqlite_file)
    cursor = db.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    gpu_symbol_table = []
    for table_name in tables:
        tname = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % tname, db)
        # print("table-%d = %s, count=%d" % (i,tname,len(table.index)) )
        if len(table.index) > 0:
            table.to_csv(logdir + tname + '.csv', index_label='index')
        if tname == "StringTable":
            gpu_symbol_table = table
    print_progress("read nvprof traces -- end")
    
    
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
    
    
    if  os.path.exists(logdir+"CUPTI_ACTIVITY_KIND_KERNEL.csv") or os.path.exists(logdir+"CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL.csv") : 
        print_progress("export-csv CUDA kernels -- begin")
        
        gpu_kernel_records = cursor.fetchall()
        gpu_kernel_traces = pd.DataFrame(pd.np.empty((len(gpu_kernel_records), len(sofa_fieldnames))) * pd.np.nan) 
        gpu_kernel_traces.columns = sofa_fieldnames 
        t_base = 0
        print_info("Length of gpu_kernel_traces = %d"%len(gpu_kernel_traces))
        for i in range(0, len(gpu_kernel_traces)):
            record = gpu_kernel_records[i]
            if i == 0:
                t_base = record[0]
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end   = (record[1] - t_base) / 1e9 + t_glb_base
            gpu_kernel_traces.loc[i,'timestamp']   =   t_begin
            gpu_kernel_traces.loc[i,'event']       =   record[2] 
            gpu_kernel_traces.loc[i,'duration']    =   t_end - t_begin
            gpu_kernel_traces.loc[i,'deviceId']    =   record[4]
            gpu_kernel_traces.loc[i,'copyKind']    =   -1
            gpu_kernel_traces.loc[i,'payload']     =   0
            gpu_kernel_traces.loc[i,'pkt_src']     =   -1
            gpu_kernel_traces.loc[i,'pkt_dst']     =   -1
            gpu_kernel_traces.loc[i,'pid']         =   record[3] #streamId
            gpu_kernel_traces.loc[i,'tid']         =   -1
            gpu_kernel_traces.loc[i,'name']        =   cxxfilt.demangle( ("%s" % gpu_symbol_table.loc[gpu_symbol_table._id_ == record[2], 'value'])) # "ID%d"%record[2] #
            gpu_kernel_traces.loc[i,'category']    =   0
        series.append(
        {   "name" : 'GPU_KERNEL',
            "color": 'rgba(0, 0, 255, .5)',
            "data" : gpu_kernel_traces})
        gpu_kernel_traces.to_csv(logdir + 'gputrace.csv', mode='w')        
        print_progress("export-csv CUDA kernels -- end")
 
#series: [{
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

   
    print_progress("query CUDA memcpy (h2d,d2h,d2d) -- begin")
    # index,_id_,copyKind,srcKind,dstKind,flags,bytes,start,end,deviceId,contextId,streamId,correlationId,runtimeCorrelationId
    cursor.execute(
        "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    gpu_memcpy_records = cursor.fetchall()
    print_progress("query CUDA memcpy (h2d,d2h,d2d) -- end")
   
    
    print_progress("export-csv CUDA memcpy (h2d,d2h,d2d) -- begin")
    if  os.path.exists(logdir+"CUPTI_ACTIVITY_KIND_MEMCPY.csv") : 
        gpu_memcpy_traces = pd.DataFrame(pd.np.empty((len(gpu_memcpy_records), len(sofa_fieldnames))) * pd.np.nan) 
        gpu_memcpy_traces.columns = sofa_fieldnames 
        t_base = 0
        print_info("Length of gpu_memcpy_traces = %d"%len(gpu_memcpy_traces))
        for i in range(0, len(gpu_memcpy_traces)):
            record = gpu_memcpy_records[i]
            if i == 0:
                t_base = record[0]
   
          #      t_begin = (record[0] - t_base) / 1e9 + t_glb_base
          #      t_end = (record[1] - t_base) / 1e9 + t_glb_base
          #      duration = t_end - t_begin
          #      gputrace.time = t_begin
          #      gputrace.event = -1
          #      gputrace.copyKind = record[3]
          #      gputrace.deviceId = record[4]
          #      gputrace.streamId = record[7]
          #      gputrace.duration = duration
          #      gputrace.data = record[2]
   
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end   = (record[1] - t_base) / 1e9 + t_glb_base
            gpu_memcpy_traces.loc[i,'timestamp']   =   t_begin
            gpu_memcpy_traces.loc[i,'event']       =   record[2] 
            gpu_memcpy_traces.loc[i,'duration']    =   t_end - t_begin
            gpu_memcpy_traces.loc[i,'deviceId']    =   record[4]
            gpu_memcpy_traces.loc[i,'copyKind']    =   record[3]
            gpu_memcpy_traces.loc[i,'payload']     =   record[2]
            gpu_memcpy_traces.loc[i,'pkt_src']     =   -1
            gpu_memcpy_traces.loc[i,'pkt_dst']     =   -1
            gpu_memcpy_traces.loc[i,'pid']         =   record[7] #streamId
            gpu_memcpy_traces.loc[i,'tid']         =   -1
            gpu_memcpy_traces.loc[i,'name']        =   "%dB"% record[2]#
            gpu_memcpy_traces.loc[i,'category']    =   0
        
    print_progress("export-csv CUDA memcpy (h2d,d2h,d2d) -- end")
 
    
    print_progress("query CUDA memcpy2 (p2p) -- begin")
    cursor.execute(
        "SELECT start,end,bytes,copyKind,deviceId,srcKind,dstKind,streamId  FROM CUPTI_ACTIVITY_KIND_MEMCPY2")
    gpu_memcpy2_records = cursor.fetchall()
    print_progress("query CUDA memcpy2 (p2p) -- end")
    
    print_progress("export-csv CUDA memcpy2 (p2p) -- begin")
    if  os.path.exists(logdir+"CUPTI_ACTIVITY_KIND_MEMCPY2.csv") : 
        gpu_memcpy2_traces = pd.DataFrame(pd.np.empty((len(gpu_memcpy2_records), len(sofa_fieldnames))) * pd.np.nan) 
        gpu_memcpy2_traces.columns = sofa_fieldnames 
        t_base = 0
        print_info("Length of gpu_memcpy2_traces = %d"%len(gpu_memcpy2_traces))
        for i in range(0, len(gpu_memcpy2_traces)):
            record = gpu_memcpy2_records[i]
            if i == 0:
                t_base = record[0]
   
            t_begin = (record[0] - t_base) / 1e9 + t_glb_base
            t_end   = (record[1] - t_base) / 1e9 + t_glb_base
            gpu_memcpy2_traces.loc[i,'timestamp']   =   t_begin
            gpu_memcpy2_traces.loc[i,'event']       =   record[2] 
            gpu_memcpy2_traces.loc[i,'duration']    =   t_end - t_begin
            gpu_memcpy2_traces.loc[i,'deviceId']    =   record[4]
            gpu_memcpy2_traces.loc[i,'copyKind']    =   record[3]
            gpu_memcpy2_traces.loc[i,'payload']     =   record[2]
            gpu_memcpy2_traces.loc[i,'pkt_src']     =   -1
            gpu_memcpy2_traces.loc[i,'pkt_dst']     =   -1
            gpu_memcpy2_traces.loc[i,'pid']         =   record[7] #streamId
            gpu_memcpy2_traces.loc[i,'tid']         =   -1
            gpu_memcpy2_traces.loc[i,'name']        =   "%dB"% record[2]#
            gpu_memcpy2_traces.loc[i,'category']    =   0
        
    print_progress("export-csv CUDA memcpy2 (h2d,d2h,d2d) -- end")
    
    
    #==================== Summary ==================#
    print_progress("json-export for cpu, network and gpu traces -- begin")
    print_info("Number of groups of series: %d" % (len(series)))
    with open(logdir + 'report.js', 'w') as f_report:
        f_report.write("cpu_traces = ")
        cpu_traces.rename(columns={'timestamp': 'x', 'event':'y'}, inplace=True)
        sofa_series = { "name": 'CPU',
                        "color": 'rgba(255, 100, 50, .5)',
                        "data": json.loads(cpu_traces.to_json(orient='records'))
                        }
        json.dump(sofa_series, f_report)
        cpu_traces.rename(columns={'x': 'timestamp', 'y':'event'}, inplace=True)
        f_report.write("\n\n")

        if len(net_traces) > 0:
            f_report.write("net_traces = ")
            net_traces.rename(columns={'timestamp': 'x', 'event':'y'}, inplace=True)
            sofa_series = { "name": 'Network',
                            "color": 'rgba(100, 255, 50, .5)',
                            "data": json.loads(net_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            net_traces.rename(columns={'x': 'timestamp', 'y':'event'}, inplace=True)
        else:
            f_report.write("net_traces = {}")
        f_report.write("\n\n")

        if len(gpu_kernel_traces) > 0:
            f_report.write("gpu_kernel_traces = ")
            gpu_kernel_traces.rename(columns={'timestamp': 'x', 'event':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_Kernel',
                            "color": 'rgba(10, 50, 255, .5)',
                            "data": json.loads(gpu_kernel_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_kernel_traces.rename(columns={'x': 'timestamp', 'y':'event'}, inplace=True)
            f_report.write("\n\n")
        else:
            f_report.write("gpu_kernel_traces = {}") 
        f_report.write("\n\n")
       
        if len(gpu_memcpy_traces) > 0:
            f_report.write("gpu_memcpy_traces = ")
            gpu_memcpy_traces.rename(columns={'timestamp': 'x', 'event':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_MEMCPY',
                            "color": 'rgba(10, 110, 200, .5)',
                            "data": json.loads(gpu_memcpy_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_memcpy_traces.rename(columns={'x': 'timestamp', 'y':'event'}, inplace=True)
            f_report.write("\n\n")
        else:
            f_report.write("gpu_memcpy_traces = {}") 
        f_report.write("\n\n")
        
        if len(gpu_memcpy2_traces) > 0:
            f_report.write("gpu_memcpy2_traces = ")
            gpu_memcpy2_traces.rename(columns={'timestamp': 'x', 'event':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_MEMCPY2',
                            "color": 'rgba(10, 80, 50, .5)',
                            "data": json.loads(gpu_memcpy2_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_memcpy2_traces.rename(columns={'x': 'timestamp', 'y':'event'}, inplace=True)
            f_report.write("\n\n")
        else:
            f_report.write("gpu_memcpy2_traces = {}") 
        f_report.write("\n\n")
        
        f_report.write("sofa_traces = [ cpu_traces, net_traces, gpu_kernel_traces, gpu_memcpy_traces, gpu_memcpy2_traces ]")        
                
    print_progress("json-export for cpu, network and gpu traces -- end")

    print_progress("Display overhead dynamic for cpu, network and gpu traces -- begin")
    print_info("Number of groups of series: %d" % (len(series)))
    with open(logdir + 'overhead.js', 'w') as f_report:
        f_report.write("cpu_traces = ")
        cpu_traces.rename(columns={'timestamp':'x', 'duration':'y'}, inplace=True)
        sofa_series = { "name": 'CPU',
                        "color": 'rgba(255, 100, 50, .5)',
                        "data": json.loads(cpu_traces.to_json(orient='records'))
                        }
        json.dump(sofa_series, f_report)
        cpu_traces.rename(columns={'x':'timestamp', 'y':'duration'}, inplace=True)
        f_report.write("\n\n")

        if len(net_traces) > 0:
            f_report.write("net_traces = ")
            net_traces.rename(columns={'timestamp': 'x', 'duration':'y'}, inplace=True)
            sofa_series = { "name": 'Network',
                            "color": 'rgba(100, 255, 50, .5)',
                            "data": json.loads(net_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            net_traces.rename(columns={'x':'timestamp', 'y':'duration'}, inplace=True)
        else:
            f_report.write("net_traces = {}")
        f_report.write("\n\n")

        if len(gpu_kernel_traces) > 0:
            f_report.write("gpu_kernel_traces = ")
            gpu_kernel_traces.rename(columns={'timestamp': 'x', 'duration':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_Kernel',
                            "color": 'rgba(10, 50, 255, .5)',
                            "data": json.loads(gpu_kernel_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_kernel_traces.rename(columns={'x':'timestamp', 'y':'duration'}, inplace=True)
        else:
            f_report.write("gpu_kernel_traces = {}") 
        f_report.write("\n\n")
       
        if len(gpu_memcpy_traces) > 0:
            f_report.write("gpu_memcpy_traces = ")
            gpu_memcpy_traces.rename(columns={'timestamp': 'x', 'duration':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_MEMCPY',
                            "color": 'rgba(10, 110, 200, .5)',
                            "data": json.loads(gpu_memcpy_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_memcpy_traces.rename(columns={'x':'timestamp', 'y':'duration'}, inplace=True)
        else:
            f_report.write("gpu_memcpy_traces = {}") 
        f_report.write("\n\n")
        
        if len(gpu_memcpy2_traces) > 0:
            #traces_to_json(cpu_traces, 'timestamp', 'event', 'CPU', 'rgba(255,100,50,.5)', logdir + 'gpu-overhead.js', 'a')
            f_report.write("gpu_memcpy2_traces = ")
            gpu_memcpy2_traces.rename(columns={'timestamp': 'x', 'duration':'y'}, inplace=True)
            sofa_series = { "name": 'GPU_MEMCPY2',
                            "color": 'rgba(10, 80, 50, .5)',
                            "data": json.loads(gpu_memcpy2_traces.to_json(orient='records'))
                            }
            json.dump(sofa_series, f_report)
            gpu_memcpy2_traces.rename(columns={'x':'timestamp', 'y':'duration'}, inplace=True)
        else:
            f_report.write("gpu_memcpy2_traces = {}") 
        f_report.write("\n\n")
        
        f_report.write("sofa_traces = [ cpu_traces, net_traces, gpu_kernel_traces, gpu_memcpy_traces, gpu_memcpy2_traces ]")        
                
    print_progress("Display overhead dynamic for cpu, network and gpu traces -- end")



    #print_progress("export overhead.js -- begin")
    #print("Length of GPU series: %d" % (len(gpu_series)))
    #with open(logdir + 'gpu-overhead.js', 'w') as jsonfile:
    #    jsonfile.write("trace_data = ")
    #    json.dump(gpu_series, jsonfile)
    #    jsonfile.write("\n")
    #print_progress("export overhead.js -- end")
