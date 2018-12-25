import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import scipy
import multiprocessing as mp
from functools import partial
from sofa_print import *
from sofa_config import *
from sofa_common import *
from STree import *
from sklearn.cluster import KMeans
from scipy import stats
import re 

table_size = 0
iteration_timelines = []
iteration_table = []
blank_count = 0
def iter_profile(cfg, fields, df_gpu):
    elapsed_time = df_gpu['timestamp'].iat[len(df_gpu)-1] - df_gpu['timestamp'].iat[0]
    fw_time = df_gpu[df_gpu['name'].str.contains("_fw")].loc[:,'duration'].sum()
    bw_time = df_gpu[df_gpu['name'].str.contains("_bw")].loc[:,'duration'].sum()
    gemm_time = df_gpu[df_gpu['name'].str.contains("gemm")].loc[:,'duration'].sum()
    copy_time = df_gpu[df_gpu['name'].str.contains("CUDA_COPY")].loc[:,'duration'].sum()
    allreduce_time = df_gpu[df_gpu['name'].str.contains("AllReduceKernel")].loc[:,'duration'].sum()
    streams = len(df_gpu['tid'].value_counts())
    #print("streams: ",streams)
    gpu_time = df_gpu['duration'].sum() 
    copy_time = copy_time + allreduce_time
    kernel_time = gpu_time - copy_time
    payload = df_gpu['payload'].sum()
    return {'elapsed_time': elapsed_time, 'fw_time': fw_time, 'bw_time': bw_time, 'kernel_time': kernel_time, 'payload': payload, 'copy_time':copy_time, 'gpu_time':gpu_time, 'gemm_time':gemm_time, 'streams':streams}

def gpu_profile(logdir, cfg, df_gpu):
    total_kernel_time = 0.0
    total_gpu_time = 0.0
    if len(df_gpu) == 0:
        print_warning("No GPU traces to profile")
        return
    print_title("Task Time (MEMCPY included) for each Device (s)")
    grouped_df = df_gpu.groupby("deviceId")["duration"]
    total_tasktime = 0
    for key, item in grouped_df:
        print(("[%d]: %lf" % (int(float(key)), grouped_df.get_group(key).sum())))
        total_tasktime = total_tasktime + grouped_df.get_group(key).sum()
    n_devices = len(grouped_df)
    per_gpu_time = total_tasktime / n_devices
    print(("Averaged GPU time of devices: %.2lf" % per_gpu_time))

    print_title("Data Traffic (bidirection) for each Device (MB)")
    grouped_df = df_gpu.groupby("deviceId")["payload"]
    for key, item in grouped_df:
        print(("[%d]: %lf" % (key, grouped_df.get_group(key).sum() / 1000000.0)))

    grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        if key == 0:
            total_kernel_time = grouped_df.get_group(key).sum()

    print_title("All-reduce Time (s)")
    all_reduce_time = 0
    grouped_df = df_gpu.groupby("name")["duration"]
    for key, item in grouped_df:
        #print("[%s]: %lf" % (key, grouped_df.get_group(key).sum()))
        if key.find("AllReduce") != -1:
            all_reduce_time = all_reduce_time + grouped_df.get_group(key).sum()

    comm_profile(logdir, cfg, df_gpu)
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))

    print_title("Summary of Kernels")
    print(("MeasuredTotalKernelTime : %lf (s)" % total_kernel_time))
    print(("MeasuredAllReduceTime : %lf (s)" % all_reduce_time))
    get_top_k_events(df_gpu, 10)

def duration_sum(df_gpu):
    duration_us = int(sum(df_gpu[ df_gpu['duration'] > 1e-5 ]['duration'])/1e-6)
    return duration_us
def kernel_count(df_gpu):
    count = df_gpu[ df_gpu['duration'] > 1e-5 ].count()[0]
    return count

    
def main_string_generate_v1(df_gpu):
    wid = 0
    name_pre = ''
    name_table = {'':0}
    id_seq = []
    for i in range(len(df_gpu)):
        trace  = df_gpu.iloc[i]
        name = trace['name'] 
        #TODO: only parsing one GPU with another better method.
        name = re.sub(r'\[.+\]', '', name.rstrip());
        value = name_table.get(name)
        if value == None:
            #print('new name: ', name, 'wid: ', wid)
            wid = wid + 1
            name_table[name] = wid 
            value = wid
        id_seq.append(str(value))
        #id_seq.append(str(int(trace['duration']/1e-5)))
    main_string = ','.join(id_seq)
    return (main_string,name_table)

def main_string_generate_v2(df_gpu):
    tick = 0
    tick_next = 0
    tick_begin = 0
    tick_end = int( round( ( t_df_end - t_df_begin ) / time_interval ) )
    event_names = get_top_k_events(df_gpu,10)
    events = event_names[:]
    pattern_table = [] #np.array(np.zeros((tick_end,len(event_names)+2),dtype=int))	
    
    iteration_pattern_count = 1
    while(tick < tick_end):
        tick_next = tick + 1
        tick_event = (df_gpu.loc[:, "timestamp"] - t_df_begin) / time_interval
        #slice trace to block by time interval
        df_block = df_gpu[( tick_event < tick_next) & ( tick_event >= tick)]
        #Create vector and count 
        vector = np.array(np.zeros(len(event_names)+2))
        #print(df_block.shape)
        if df_block.shape[0] > 0:
            for i in range(len(event_names)):
                count = event_count('name', event_names[i], df_block)
                vector[i] = count  
            
            vector[len(event_names)] = 0;#duration_sum(df_block)
            vector[len(event_names)+1] = 0;#kernel_count(df_block)
            matched_index = pattern_matching(pattern_table, events, vector, threshold)
            if matched_index != -1:
                iteration_timelines.append(str(matched_index))
            else:
                iteration_timelines.append('0')
        else:
            iteration_timelines.append('0')
        tick += 1
    main_string = ",".join(iteration_timelines)

def get_top_k_events(df, topk):
    topk_events=[]
    print("Top %d Events:"%topk)
    gby = df.groupby(['name'])
    df_agg = gby.aggregate(np.sum)
    df_agg_sorted = df_agg.sort_values(by=['duration'],ascending=False)
    #memcpy = ['copyKind_1_','copyKind_2_','copyKind_8_']
    eventName = df_agg_sorted[df_agg_sorted.columns[0:0]].head(topk).index.values.tolist()
    #eventName.extend(memcpy)

    for i in range(len(eventName)):
        print('[%d] %s'%(i,eventName[i]))
    return eventName

def get_memcpyHtoD(df):

    df_sorted = df.sort_values(by=['payload'],ascending=False)
    HtoD = df_sorted[df_sorted['name'].str.contains('copyKind_1_', na=False)]['name'].head(1).values
    
    return HtoD

def pattern_filter(candidate_patterns):
    filtered_candidate_patterns = []
    for pattern in candidate_patterns:
        #print("0>"+pattern)
        if pattern.startswith(','):
            pattern = pattern[1:len(pattern)]
        #print("1>"+pattern)
        if pattern.endswith(','):
            pattern = pattern[0:len(pattern)-1]
        #print("2>"+pattern)
        seq = list(map(int, pattern.split(',')))
        if sum(seq) != seq[0]*len(seq):
            filtered_candidate_patterns.append(pattern)
    return filtered_candidate_patterns

def iter_detect(logdir, cfg, df_gpu, time_interval, threshold, iteration_times):
    global iteration_timelines, iteration_table, blank_count, \
            iteration_table
    df_gpu.to_csv('dfgpu.csv')
    t_df_begin = df_gpu.iloc[0]['timestamp'] 
    t_df_end = df_gpu.iloc[-1]['timestamp'] 
    candidate_patterns=[]
    (main_string,name_table) = main_string_generate_v1(df_gpu)
    #main_string = "0,1,1,1,1,1,0,2,3,2,3,2,3"
    #main_string = "49,49,49,49,49,49,1,1,2,2,1,2,1,2"
    #print('main_string: '+main_string)
    st = STree(main_string)
    st.find_repeat_pattern(candidate_patterns, iteration_times)
    candidate_patterns.sort(key = lambda s: len(s), reverse=True)
    filtered_candidate_patterns = pattern_filter(candidate_patterns)
    print('There are %d candidate patterns for %d-iteration'%(len(candidate_patterns),cfg.num_iterations))
    pattern_pre = ""
    pat_seq = []
    for pattern in filtered_candidate_patterns:
        #print('A: ',pattern_pre)
        #print('B: ',pattern)
        pp_ratio = fuzz.ratio(pattern,pattern_pre)
        if pp_ratio > 80:
            #print("pattern too similar: ",pp_ratio) 
            continue
        else:
            pattern_pre = pattern 
        #print('original string length of main_string = %d' % len(main_string))
        wid_seq = main_string.split(',')
        pat_seq = pattern.split(',')
        total_length = num_wids = len(wid_seq)
        #print('runtime string length = %d' % num_wids )
        #print('pattern length (block_size) = %d' % len(pat_seq)) 
        block_size = len(pat_seq)
        matched_block_end_pre = 0 
        block_begin = 0
        block_end = 0
        block_end = block_size
        step = 1
        iteration_count = 0
        b_overlap = False
        fw_threshold = 85
        ind = []
        while block_begin <= (total_length - block_size):
            blockString = ",".join(wid_seq[block_begin:block_end])
            fuzz_ratio = fuzz.ratio(blockString,pattern)
            if fuzz_ratio >= fw_threshold:
                ind.append(block_begin)
                block_begin = block_end
                block_end = block_begin + block_size
            else:
                block_begin = block_begin + step
                block_end = block_begin + block_size

        #print('Non-overlapped %d-times pattern: %s'%(len(ind),pattern))
        if len(ind) == cfg.num_iterations:    
            for i in ind:
                iteration_table.append((df_gpu.iloc[i]['timestamp'],df_gpu.iloc[i+block_size-1]['timestamp']))
            break  
        else:
            #print_warning("No matched strings by fuzzywuzzy of threshold %d."%fw_threshold)
            continue

    print("Selected pattern:")
    print(pat_seq)
    print("End of AISI")

def event_count(column, eventName, df):
        #get the count of rows that contain eventName
        count = 0
        for i in range(df.shape[0]):
            if df.iloc[i][column].find(eventName) != -1:
                count = count + 1
        return count 
        
def pattern_matching(pattern_table, events, vector, threshold):
    global table_size
    if np.sum(vector) == 0:
        return -1
    for i in range(table_size):
        pvector = pattern_table[i] 
        if similar(pvector, vector, threshold):
            return i
    pattern_table.append(vector) 

    return table_size 

def similar(a, b, threshold):
    result = 1.0 - scipy.spatial.distance.cosine(a, b)
    if result > threshold: 
        langa = np.linalg.norm(a) 
        langb = np.linalg.norm(b) + 1e-9
        ratio = np.abs(langa/langb) 
        if ratio < 1.25 and ratio > 0.75:
            return True
    return False
 
def traces_to_json(path):
    global iteration_table
    sofa_traces = None
    with open(path, 'r') as f:
        sofa_traces = f.readlines()
    with open(path, 'w') as f:
        f.writelines([line for line in sofa_traces[:-1]])
        f.write("\n\n")
        f.write("iteration_detection_begin = ")
        f.write('{"name": "iteration_begin", "color": "rgba(241,156,162,1)", "data": [')    
        for (IT_beg, IT_end) in iteration_table:
            #print("begin:%f end:%f duration:%f"%(IT_beg, IT_end, IT_end-IT_beg))
            for i in range(10):
                dot_y = np.power(10,float(i-5))
                f.write('{"name": "iteration_begin", "x": ' + str(IT_beg) + ', "y": ' + str(dot_y) + '}, ')
        f.write(']}\n')
        f.write("iteration_detection_end = ")
        f.write('{"name": "iteration_end", "color": "rgba(241,15,162,1)", "data": [')    
        for (IT_beg, IT_end) in iteration_table:
            #print("begin:%f end:%f duration:%f"%(IT_beg, IT_end, IT_end-IT_beg))
            for i in range(10):
                dot_y = np.power(10,float(i-5))
                f.write('{"name": "iteration_end", "x": ' + str(IT_end) + ', "y": ' + str(dot_y) + '}, ')
        f.write(']}\n')
            
        for s in sofa_traces[-1].split(','): 
            if s.find(']') == -1:
                f.write(s+',')
        f.write("iteration_detection_begin,iteration_detection_end]")
    

def trace_timeline(path):
    global iteration_timelines, blank_count
    i = 0
    with open(path, 'w') as f:
        for x in range(blank_count):
            f.write(str(i) + "," + str(0))
            i += 1
        for index in iteration_timelines:
            f.write(str(i) + "," + str(index))
            i += 1

def sofa_aisi(logdir, cfg, df_cpu, df_gpu):
    global iteration_table
    df_gpu_x1 = df_gpu[df_gpu.deviceId == 1]
    a = 0
    times = []
    times2 = []

    print_title('AISI: Per-iteration Performance Summary')
    try: 
        iter_detect(logdir, cfg, df_gpu_x1, 0.01, 0.8, cfg.num_iterations) 
        traces_to_json(logdir + 'report.js')
        xlist = []
       
        if len(iteration_table) == 0:
            return

        for t in iteration_table:
            xlist.append((t[0],0))
            times2.append(t[0])
        
        X = np.array(xlist)
        kmeans = KMeans(n_clusters=cfg.num_iterations, random_state=0).fit(X)
         
        for c in kmeans.cluster_centers_:
            times.append(c[0])
        
        times = np.sort(times2)
        trace_timeline(logdir + 'iteration_timeline.txt')
        
        iter_summary_fields = ['elapsed_time', 'kernel_time', 'fw_time', 'bw_time', 'copy_time', 'gpu_time', 'cpu_time', 'bw_h2d', 'bw_d2h', 'bw_p2p', 'bw_d2h_big', 'bw_d2h_big', 'bw_p2p_big', 'payload', 'gemm_time', 'streams']
        iter_list = []
        for i in range(1,len(times)):
            #print_title("Perormance analyze of iteration-%d"%(i))
            overlapness = 0.0
            cond1 = (df_gpu_x1['timestamp'] >= times[i-1])
            cond2 = (df_gpu_x1['timestamp'] <  times[i])
            df_gpu_iteration = df_gpu_x1[ cond1 & cond2 ]
            #df_cpu_iteration = df_cpu[ cond1 & cond2 ]
            if len(df_gpu_iteration) > 0:
                iter_list.append(iter_profile(cfg, iter_summary_fields, df_gpu_iteration))
        iter_summary = pd.DataFrame( iter_list, columns=iter_summary_fields )

        if not iter_summary.empty:
            mean_fw_time = iter_summary['fw_time'].mean()
            mean_bw_time = iter_summary['bw_time'].mean()
            mean_copy_time = iter_summary['copy_time'].mean()
            mean_gpu_time = iter_summary['gpu_time'].mean()
            mean_gemm_time = iter_summary['gemm_time'].mean()
            mean_kernel_time = mean_gpu_time - mean_copy_time 
            mean_payload = iter_summary['payload'].mean()
            mean_streams = iter_summary['streams'].mean()
            print("Elapsed time of initial iteration (s): ", iter_summary.loc[0,'elapsed_time'])
            mean_step_time = scipy.stats.gmean(iter_summary['elapsed_time'])
            if cfg.num_iterations > 1:
                mean_step_time = scipy.stats.gmean(iter_summary['elapsed_time'].iloc[1:])
            print('Averaged per-iteration elapsed time (s)', mean_step_time) 
            print("Averaged CUDA GPU time (s): ", mean_gpu_time)
            print("Averaged CUDA FW time (s): ", mean_fw_time)
            print("Averaged CUDA BW time (s): ", mean_bw_time)
            print("Averaged CUDA gemm time (s): ", mean_gemm_time)
            print("Averaged CUDA COPY time (s): ", mean_copy_time)
            print("Averaged CUDA payload (B): ", mean_payload)
            print("Averaged number of CUDA streams (B): ", mean_streams)
            print_title('Performance Optimization Hints')
            if mean_copy_time / mean_step_time < 0.15: 
                print("The profiled program is a compute-bound workload, try increasing # of GPUs to improve throughput")
            else: 
                print("The profiled program is a communication-bound workload, %d bytes are monitored on PCIe bus"%mean_payload)
                print("Try using RP Mode parameter synchronization method instead of PS Mode.")
            print('\n\n') 
            return iter_summary
        else:
            print_warning('No iteration detected after scanning runtime string!')
    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

