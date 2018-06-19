#!/usr/bin/python
import os
import sys
import pandas as pd
import numpy as np
import csv
import json
import random
from fuzzywuzzy import fuzz
from scipy import spatial
from operator import itemgetter, attrgetter
import argparse
import multiprocessing as mp
from functools import partial
from sofa_print import *
from sofa_config import *
from STree import *

#events = ["HtoD", "DtoH", "DtoD", "PtoP"]
eventName = ['maxwell_scudnn_128x128_stridedB_splitK_interior_nn','cudnn::detail::bn_bw_1C11_kernel_new', 'maxwell_scudnn_128x128_relu_interior_nn', 'cudnn::detail::bn_fw_tr_1C11_kernel_new', 'maxwell_scudnn_128x128_stridedB_interior_nn', 'maxwell_sgemmBatched_64x64_raggedMn_nt', 'TensorConversionOp', 'maxwell_sgemmBatched_64x64_raggedMn_nn', 'TensorCwiseBinaryOp<Eigen::internal::scalar_max_op', 'TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op']
iteration_begin = 0
iteration_end = 0
iteration_timelines = []
iteration_index = 1
iteration_table = []
blank_count = 0

base_time = 0

def iterationDetection(logdir, cfg, df_gpu, time_interval, threshold, iteration_times):
    global iteration_begin, iteration_end, iteration_index, iteration_timelines, iteration_table, base_time, blank_count
    time_begin = 0
    time_end = int(round(df_gpu.iloc[-1]['timestamp']/time_interval))
    events = eventName[:]
    events.append("timestamp")
    patternTable = pd.DataFrame(columns = events, dtype = int)	
    candidate_pattern=[]
    iteration_pattern_count = 1 
    time = 0
    n = 0
    k = 2
    #Create pattern table by extract top-k feature, and label each vector with specified index 
    while(time < time_end):
        clock = time + 1
        #slice trace to block by time interval
        df_block = df_gpu[(df_gpu.loc[:, "timestamp"] / time_interval  < clock) & (df_gpu.loc[:, "timestamp"] / time_interval  >= time)]
        
        #Create vector and count 
        vector = []
        for e in eventName:
            vector.append(eventCount('name', e, df_block))
        
        if sum(vector)==0:
            #if vector is empty
            if not patternTable.empty: 
                iteration_timelines.append('0,')
            else:
                blank_count += 1
            time += 1
            continue
        patternMatch = patternMatching(patternTable, vector, threshold)
        if patternMatch != -1:
            iteration_timelines.append(str(iteration_index)+",")
            
        else:
            iteration_timelines.append(str(iteration_pattern_count)+",")
            iteration_pattern_count += 1
            vector.append(time)
            vectorSerie = pd.Series(vector, index = events)
            patternTable = patternTable.append(vectorSerie, ignore_index = True)

        time += 1

    #building suffix tree to find pattern
    mainString = "".join(iteration_timelines)
    st = STree(mainString)    
    IT_times = iteration_times
#    while not candidate_pattern:
#        if(IT_times == 0):
#            print("can't find pattern")
#            return 0
    st.find_repeat_pattern(candidate_pattern, IT_times)
#        IT_times -= 1
    #pattern finded  
    print("iteration_timelines:", iteration_timelines)
    print("mainString:", mainString) 
    print("candidate_pattern:",candidate_pattern)
    pattern = max(candidate_pattern, key = len)
    print("pattern:", pattern)
    #reverse main string and pattern then do approximate match by cutting main string to n+k block, where n is iteration_times, k is akon's heart mothod(TODO: k can set by evaluate hardware config). the reason why we doing this is because in DL models, the warm-up stage may cost more time then stable state iteration. 
    total_length = len(mainString)
    block_size = len(pattern)
    process = 0
    block_beg = 0
    block_end = block_size
    iteration_count = 0
    fuzzyRatioTable = []
    print("black_count:",blank_count)
    while process < (total_length - block_size):
        blockString = mainString[block_beg:block_end]
        
        #use fuzzywuzzy as approximate match accuracy. TODO: use reasonable threshold.
        fuzz_ratio = fuzz.token_sort_ratio(blockString, pattern) 
        fuzzyRatioTable.append(fuzz_ratio)
        print("fuzz ratio:", fuzz_ratio)
        block_beg += 2
        block_end += 2
        process += 2
    #find largest fuzzy ratio n blocks (n = iteration_times)
    ind = np.argpartition(fuzzyRatioTable, -iteration_times)[-iteration_times:]
    for index in ind: 
        iteration_table.append((float(index + blank_count) * time_interval, float(index + block_size + blank_count) * time_interval))     
def eventCount(column, eventName, df):
    #get rows count that contains eventName
    return df[df[column].str.contains(eventName, na=False)][column].count()
        
def patternMatching(patternTable, vector, threshold):
    
    global iteration_index
    if patternTable.empty:
        return -1 
    for index, row in patternTable.iterrows():
        #pvector is vector from patternTable without timestamp
        pvector = row.values[:-1]
        
        if similar(pvector, vector):
            iteration_index = index+1
            return 1
    return -1

def similar(a, b):
    langa = np.linalg.norm(a)
    langb = np.linalg.norm(b)
    if (langa/langb)<0.4 or (langb/langa)<0.4:
        return False
    result = 1. - spatial.distance.cosine(a, b)
    if result < 0.7:
        return False
    return True

 
def traces_to_json(path):
    global iteration_begin, iteration_end, iteration_table, base_time
    sofa_traces = None
    with open(path, 'r') as f:
        sofa_traces = f.readlines()
    with open(path, 'w') as f:
        #sofa_traces = f.readlines()
        f.writelines([line for line in sofa_traces[:-1]])
        f.write("\n\n")
        f.write("iteration_detection = ")
        f.write('{"color": "rgba(241,156,162,1)", "data": [')    
        for (IT_beg, IT_end) in iteration_table:
            print("begin:%f end:%f duration:%f"%(IT_beg+base_time,IT_end+base_time,IT_end-IT_beg))
            f.write('{"name": "iteration_begin", "x": ' + str(IT_beg + base_time) + ', "y": 1000000}, {"name": "iteration_end", "x": ' + str(IT_end + base_time) +', "y": 1000000}, ')
        f.write(']}\n')
        for s in sofa_traces[-1].split(','): 
            if s.find(']') == -1:
                f.write(s+',')
        f.write("iteration_detection ]")
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

def sofa_deepprof(logdir, cfg, df_cpu, df_gpu):
    global iteration_begin, iteration_end, base_time

    try:
        base_time = df_gpu.loc[0,'timestamp']
        df_gpu.loc[:,'timestamp'] -= df_gpu.loc[0,'timestamp']
        iterationDetection(logdir, cfg, df_gpu, 0.01, 0.7, 11)
        iteration_begin += base_time
        iteration_end += base_time 
        traces_to_json(logdir + 'report.js')
        trace_timeline(logdir + 'iteration_timeline.txt')

    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

