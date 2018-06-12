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

def iteration_detect(logdir, cfg, df, time_interval, threshold):
    iters = []
    #fields of df = timestamp,event,duration,deviceId,copyKind,payload,bandwidth,pkt_src,pkt_dst,pid,tid,name,category
    #print(features)
    print(df.iloc[0]['timestamp'])
    print(df.iloc[-1]['timestamp'])
    df_agg = df.groupby(['name']).agg({'duration':sum})
    #g = df_agg['duration'].groupby(level=0, group_keys=False)
    print(df_agg)
    df_g = df_agg.sort_values(by=['duration'],ascending=False)
    print(df_g.head(10))
    top_names = list(df_g.head(10).keys())
    print(top_names) 
    fvectors = []
    t_begin = df.iloc[0]['timestamp']
    t_end = df.iloc[-1]['timestamp']
    fvectors = np.zeros( (int((t_end-t_begin)/time_interval),10))
    for i in range(len(df)-1):
        fvid = int((df.iloc[i]['timestamp'] - t_begin) / time_interval) 
        fvectors[fvid] = fvectors[fvid][0] + df.iloc[i]['duration'] 
        
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
    df_gpu.loc[:,'timestamp'] -= df_gpu.loc[0,'timestamp']
    iters = iteration_detect(logdir, cfg, df_cpu, 0.01, 0.7)
