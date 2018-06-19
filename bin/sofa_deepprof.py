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
import re


iteration_begin = 0
iteration_end = 0
iteration_timelines = []
iteration_index = 1
iteration_table = []

base_time = 0

def get_top_k_events(df, topk):
    topk_events=[]
    print("Top %d Events:"%topk)
    gby = df.groupby(['name'])
    df_agg = gby.aggregate(np.sum)
    df_agg_sorted = df_agg.sort_values(by=['duration'],ascending=False)
    #eventName = ['gpu1_copyKind_1_','gpu1_copyKind_2_','gpu1_copyKind_8_']
    eventName = df_agg_sorted[df_agg_sorted.columns[0:0]].head(topk).index.values.tolist()
    for i in range(len(eventName)):
        print('[%d] %s'%(i,eventName[i]))
    return eventName

def iterationDetection(logdir, cfg, df_gpu, time_interval, threshold, iteration_times):
    global iteration_begin, iteration_end, iteration_index, iteration_timelines, iteration_table, base_time
    time_begin = 0
    time_end = int(round(df_gpu.iloc[-1]['timestamp']/time_interval))
    event_names = get_top_k_events(df_gpu,10)
    events = event_names[:]
    events.append('timestamp')
    patternTable = pd.DataFrame(columns = events, dtype = int)	
    candidate_pattern=[]
    iteration_pattern_count = 1 
    time = 0
    n = 0
    k = 1 
    #Create pattern table by extract top-k feature, and label each vector with specified index 
    while(time < time_end):
        clock = time + 1
        #slice trace to block by time interval
        df_block = df_gpu[(df_gpu.loc[:, "timestamp"] / time_interval  < clock) & (df_gpu.loc[:, "timestamp"] / time_interval  >= time)]
        
        #Create vector and count 
        vector = []
        for e in event_names:
            count = eventCount('name', e, df_block)
            vector.append(count)
        if sum(vector)==0:
            #if vector is empty
            iteration_timelines.append('0,')
        else:
            patternMatch = patternMatching(patternTable, vector, threshold)
            if patternMatch != -1:
                iteration_timelines.append(str(iteration_index)+',')
            else:
                iteration_timelines.append(str(iteration_pattern_count)+',')
                iteration_pattern_count += 1
                vector.append(time)
                vectorSerie = pd.Series(vector, index = events)
                patternTable = patternTable.append(vectorSerie, ignore_index = True)
        
        time += 1

    #building suffix tree to find patter0
    #print(iteration_timelines)
    mainString = "".join(iteration_timelines)
    
    ### Worker Intelligence
    #n_matches = len(re.findall('9,10,0,0,', mainString)) + \
    #            len(re.findall('10,14,0,0,', mainString)) + \
    #            len(re.findall('9,15,0,0,', mainString))  
    #print("matched times = %d"%n_matches)
    
    #mainString='11122231111222311112202'
    #mainString = 'aabbcccaabbcccaabbccc' 
    st = STree(mainString)
    print(mainString)
    IT_times = iteration_times
    st.find_repeat_pattern(candidate_pattern, IT_times)
    print("pattern",candidate_pattern)
    #pattern finded
    print(candidate_pattern)
    pattern = max(candidate_pattern, key = len)
    print("pattern is "+pattern)
    #reverse main string and pattern then do approximate match by cutting main string to n+k block, where n is iteration_times, k is akon's heart mothod(TODO: k can set by evaluate hardware config). the reason why we doing this is because in DL models, the warm-up stage may cost more time then stable state iteration. 
    reversePattern = pattern[::-1]
    reverseMainString = mainString[::-1]
    total_length = len(mainString)
    block_size = int(total_length / (n + k))
    process = 0
    block_beg = 0
    block_end = block_size
    shrink_size = 0
    #TODO: set reasonable rate.
    shrink_rate = int(round(1 / time_interval / 10)) 
    iteration_count = 0
    while process < total_length:
        blockString = reverseMainString[block_beg:block_end]
        #use fuzzywuzzy as approximate match accuracy. TODO: use reasonable threshold.
        if fuzz.token_sort_ratio(blockString, reversePattern) > 70:
            iteration_table.append((float(total_length - 1 - block_end) / time_interval, float(total_length - 1 -block_beg) / time_interval))
            block_beg += block_size
            block_end = block_beg + block_size
            shrink_size = 0
            iteration_count += 1
            if iteration_count == iteration_times:
                print("Iteraion detection complete.")
                break
        else :
            shrink_size += shrink_rate
            block_end -= shrink_size
            if shrink_size >= block_size:
                #print "can't find iteration in this block"
                block_beg += block_size
                block_end = block_beg + block_size
                shrink_size = 0

        process += 1


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
        if similar(pvector, vector, threshold):
            iteration_index = index + 1
            return 0
    return -1

def similar(a, b, threshold): 
    result = 1. - spatial.distance.cosine(a, b)
    if result > threshold: 
        maxv = max(np.linalg.norm(a), np.linalg.norm(b))
        minv = min(np.linalg.norm(a), np.linalg.norm(b))
        ratio = np.abs(maxv/minv) 
        if ratio < 1.2 and ratio > 0.8:
            return True
    return False

 
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
    global iteration_timelines
    i = 0
    with open(path, 'w') as f:
        for index in iteration_timelines:
            f.write(str(i)+","+str(index))
            i += 1

def sofa_deepprof(logdir, cfg, df_cpu, df_gpu):
    global iteration_begin, iteration_end, base_time

    try:
        base_time = df_gpu.loc[0,'timestamp']
        df_gpu.loc[:,'timestamp'] -= df_gpu.loc[0,'timestamp']
        iterationDetection(logdir, cfg, df_gpu, 0.01, 0.7, cfg.iterations)
        iteration_begin += base_time
        iteration_end += base_time 
        traces_to_json(logdir + 'report.js')
        trace_timeline(logdir + 'iteration_timeline.txt')

    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

