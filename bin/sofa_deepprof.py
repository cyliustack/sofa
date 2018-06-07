#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import csv
import json
import random
from operator import itemgetter, attrgetter
import argparse
import multiprocessing as mp
from functools import partial
from sofa_print import *
from sofa_config import *
from scipy import spatial

#events = ["HtoD", "DtoH", "DtoD", "PtoP"]
#eventName = ['CUDA memset', 'copyKind_1', 'copyKind_2', 'copyKind_8', 'copyKind_10', 'TensorMap', 'gemm', 'fft', 'relu', 'bn_fw_tr_', 'bn_bw_']
eventName = ['maxwell_scudnn_128x128_stridedB_splitK_interior_nn','cudnn::detail::bn_bw_1C11_kernel_new', 'maxwell_scudnn_128x128_relu_interior_nn', 'cudnn::detail::bn_fw_tr_1C11_kernel_new', 'maxwell_scudnn_128x128_stridedB_interior_nn', 'maxwell_sgemmBatched_64x64_raggedMn_nt', 'TensorConversionOp', 'maxwell_sgemmBatched_64x64_raggedMn_nn', 'TensorCwiseBinaryOp<Eigen::internal::scalar_max_op', 'TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op']
iteration_begin = 0
iteration_end = 0
iteration_timelines = []
iteration_index = 1
total_iteration = []
base_time = 0

def iteration_detect(logdir, cfg, df_gpu, time_interval, threshold):
    global iteration_begin, iteration_end, iteration_index, iteration_timelines, total_iteration, base_time
    time_begin = 0
    time_end = int(round(df_gpu.iloc[-1]['timestamp']/time_interval))
    matchCount = 0
    events = eventName[:]
    events.append("timestamp")
    patternTable = pd.DataFrame(columns = events, dtype = int)	
    iteration_pattern_count = 1 
    iters = []
    time = 0
 
    while(time < time_end):
        clock = time + 1
        print("time:" + str(time) + "block:")
        #for index, row in df_gpu.iterrows():
        #if row['timestamp'] <= clock:
        #    if row['name'].split('_')[1] == "copyKind"

        #Pick the block within a interval
        df_block = df_gpu[(df_gpu.loc[:, "timestamp"] / time_interval  < clock) & (df_gpu.loc[:, "timestamp"] / time_interval  >= time)]
        
        #Create vector and count
        vector = []
        for e in eventName:
            vector.append(eventCount('name', e, df_block))
        
        '''
        count_HtoD = eventCount('name', 'copyKind_1', df_block)
        count_DtoH = eventCount('name', 'copyKind_2', df_block)
        count_DtoD = eventCount('name', 'copyKind_8', df_block)
        count_PtoP = eventCount('name', 'copyKind_10', df_block)
        '''
        #print vector 
        if sum(vector)==0:
            print("vector empty")
            iteration_timelines.append(0)
            time += 1
            continue
        #pattern matching, if match then return iteration time, else then add new pattern to table and continue.
        patternMatch = patternMatching(patternTable, vector, threshold)
        if patternMatch != -1:
            iteration_timelines.append(iteration_index)
            #if matchCount > 30, then it should be iteration
            matchCount += 1
            
            if matchCount == 1:
                iteration_begin = patternMatch * time_interval
                iteration_end = time 
            if matchCount >= 30:
                print("iteration detected")
                print("\npattern table:")
                print(patternTable) 
                print("\nvector:") 
                print(vector)
                print("\niteration begin,end:")
                print(iteration_begin, iteration_end * time_interval) 
                patternTable = pd.DataFrame(columns = events, dtype = int)
                matchCount = 0
                time = iteration_end + 1
                total_iteration.append([iteration_begin, iteration_end * time_interval])
                #return iteration_begin, iteration_end
        else:
            iteration_timelines.append(iteration_pattern_count)
            iteration_pattern_count += 1
            matchCount = 0
            vector.append(time)
            vectorSerie = pd.Series(vector, index = events)
            patternTable=patternTable.append(vectorSerie, ignore_index = True)
            print("\nnew pattern: ")
            print(iteration_pattern_count)
            print(patternTable)
            print("\n")

        time += 1
    return iters 

def eventCount(column, eventName, df):
    #get rows count that contains eventName
    return df[df[column].str.contains(eventName, na=False)][column].count()
        
def patternMatching(patternTable, vector, threshold):
    
    global iteration_index
    #print patternTable 
    if patternTable.empty:
        print("patternTable is empty!")
        return -1 
    for index, row in patternTable.iterrows():
        #pvector is vector from patternTable without timestamp
        pvector = row.values[:-1]
        sim = similarity(pvector, vector)
        if sim >= threshold:
            print("similarity: %f" % sim)
            iteration_index = index+1
            print(row["timestamp"])
            return row["timestamp"]
    print("no pattern match!")
    return -1

def similarity(a, b):
    maxv = max(np.linalg.norm(a), np.linalg.norm(b))
    minv = min(np.linalg.norm(a), np.linalg.norm(b))
    if (maxv/minv)<0.5:
        print(maxv/minv)
        return 0
    result = 1. - spatial.distance.cosine(a, b)
    #print a,b
    #print spatial.distance.cosine(a, b)
    return result

def traces_to_json(path):
    global iteration_begin, iteration_end, total_iteration, base_time
    #if len(traces) == 0:
    #    print_warning("Empty traces!")
    #    return

    with open(path, 'a') as f:
        #for trace in traces:
        #    if len(trace.data) > 0:
        f.write("\n\n")
        f.write("iteration_detection = ")
        f.write('{"color": "rgba(241,156,162,1)", "data": [')    
        for IT in total_iteration:
            f.write('{"name": "iteration_begin", "x": ' + str(IT[0] + base_time) + ', "y": 1000000}, {"name": "iteration_end", "x": ' + str(IT[1] + base_time) +', "y": 1000000}, ')
        f.write(']}')

def trace_timeline(path):
    global iteration_timelines
    i = 0
    with open(path, 'w') as f:
        for index in iteration_timelines:
            f.write(str(i)+","+str(index))
            f.write('\n')
            i += 1

def sofa_deepprof(logdir, cfg, df_cpu, df_gpu):
    print_title("Per-Iteration Performance Info.")
    global iteration_begin, iteration_end, base_time

    df_gpu.loc[:,'timestamp'] -= df_gpu.loc[0,'timestamp']
    iters = iteration_detect(logdir, cfg, df_gpu, 0.01, 0.7)
