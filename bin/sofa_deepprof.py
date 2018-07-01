#!/usr/bin/python
import os
import sys
import pandas as pd
import numpy as np
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
blank_count = 0


def get_top_k_events(df, topk):
    topk_events=[]
    print("Top %d Events:"%topk)
    gby = df.groupby(['name'])
    df_agg = gby.aggregate(np.sum)
    df_agg_sorted = df_agg.sort_values(by=['duration'],ascending=False)
    #eventName = ['gpu1_copyKind_1_','gpu1_copyKind_2_','gpu1_copyKind_8_']
    eventName = df_agg_sorted[df_agg_sorted.columns[0:0]].head(topk).index.values.tolist()
    eventName[7] = eventName[7].split('(')[0]
    for i in range(len(eventName)):
        print('[%d] %s'%(i,eventName[i]))
    return eventName

def select_pattern(candidate_pattern):
    print(candidate_pattern)
    candidate_pattern_filtered = []
    for cp in candidate_pattern: 
        if len(cp)>1:
            if cp.count(cp[0])+cp.count(cp[1]) != len(cp): 
                candidate_pattern_filtered.append(cp)
                #print('filtered cp = '+cp)
            
    pattern = max(candidate_pattern_filtered, key = len)
    print("pattern selected:", pattern)
    return pattern  

def iterationDetection(logdir, cfg, df_gpu, time_interval, threshold, iteration_times):
    global iteration_begin, iteration_end, iteration_index, iteration_timelines, iteration_table, blank_count
    t_df_begin = df_gpu.iloc[0]['timestamp'] 
    t_df_end = df_gpu.iloc[-1]['timestamp'] 
    tick_begin = 0
    tick_end = int( round( ( t_df_end - t_df_begin ) / time_interval ) )
    event_names = get_top_k_events(df_gpu,10)
    events = event_names[:]
    events.append('timestamp')
    patternTable = pd.DataFrame(columns = events, dtype = int)	
    candidate_pattern=[]
    iteration_pattern_count = 1 
    tick = 0 
    #Create pattern table by extract top-k feature, and label each vector with specified index 
    while(tick < tick_end):
        tick_next = tick + 1
        tick_event = (df_gpu.loc[:, "timestamp"] - t_df_begin) / time_interval
        #slice trace to block by time interval
        df_block = df_gpu[( tick_event < tick_next) & ( tick_event >= tick)]
        #Create vector and count 
        vector = []
        for e in event_names:
            count = eventCount('name', e, df_block)
            vector.append(count)
        if sum(vector)==0:
            #if vector is empty
            if not patternTable.empty: 
                iteration_timelines.append('0,')
            else:
                blank_count += 1
            tick += 1
            continue
        patternMatch = patternMatching(patternTable, vector, threshold)
        if patternMatch != -1:
            iteration_timelines.append(str(iteration_index)+",")
            
        else:
            iteration_timelines.append(str(iteration_pattern_count)+",")
            iteration_pattern_count += 1
            vector.append(tick)
            vectorSerie = pd.Series(vector, index = events)
            patternTable = patternTable.append(vectorSerie, ignore_index = True)
        tick += 1
    print("totaltick:",tick)
    print('timelinescount:',len(iteration_timelines))
    #building suffix tree to find patter0
    #print(iteration_timelines)
    mainString = "".join(iteration_timelines)
    #mainString='00000101001001001'
    #mainString = 'aabbcccaabbcccaabbccc' 
    st = STree(mainString)
    #print(mainString)
    st.find_repeat_pattern(candidate_pattern, iteration_times)
    #print("iteration_timelines:", iteration_timelines)
    print("mainString:", mainString) 
    #print("candidate_pattern:",candidate_pattern)
    if candidate_pattern:
        pattern = select_pattern(candidate_pattern)
        print('mainStringlen',len(mainString))
        mainString = mainString.split(',')
        total_length = len(mainString) - 1
        print('total_length',total_length)
        block_size = len(pattern) - pattern.count(',')
        block_beg = 0
        block_end = block_size
        step = 1
        #step = int(block_size/8)
        iteration_count = 0
        fuzzyRatioTable = []
        while block_beg < (total_length - block_size):
            blockString = ',' +  ",".join(mainString[block_beg:block_end]) + ','
            #use fuzzywuzzy as approximate match accuracy. TODO: use reasonable threshold.
            fuzz_ratio = fuzz.token_sort_ratio(blockString, pattern) 
            fuzzyRatioTable.append(fuzz_ratio)
            block_beg += step
            block_end += step
        print('fuzzyTable:',fuzzyRatioTable)
        print('fuzzycount:',len(fuzzyRatioTable))
        #find largest fuzzy ratio n blocks (n = iteration_times)
        ind = []
        begTable = []
        endTable = []
        beg = 0
        end = 0
        #print(fuzzyRatioTable)
        for i in range(len(fuzzyRatioTable)):
            if fuzzyRatioTable[i] > 70:
                ind.append(i)
        print('ind',ind)
        for index in ind:
            iteration_table.append((float(index*step + blank_count) * time_interval + t_df_begin, float(index*step + block_size + blank_count) * time_interval + t_df_begin))

        begTable.append(ind[0])
        beg = ind[0]
        end = ind[0] + block_size


#        for idx in range(len(ind)):
#            if (ind[idx] - beg) > block_size * 2.2:
#                begTable.append(ind[idx])
#                endTable.append(end)
#                beg = ind[idx]
#            else:
#                beg = ind[idx]
#                end = ind[idx] + block_size
#        endTable.append(end)
#        print("len of beg/end",len(begTable),len(endTable))
#
#        #ind = np.argpartition(fuzzyRatioTable, -iteration_times)[-iteration_times:]
#        #comma_factor = total_length / comma_count
#        print('blank:',blank_count)
#        for index in range(len(begTable)): 
#            iteration_table.append((float(begTable[index]*step + blank_count) * time_interval + t_df_begin, float(endTable[index]*step + blank_count) * time_interval + t_df_begin))     


    else:
        print('No iteration patterns detected.')

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
        langa = np.linalg.norm(a)
        langb = np.linalg.norm(b)
        ratio = np.abs(langa/langb) 
        if ratio < 2 and ratio > 0.5:
            return True
    return False
 
def traces_to_json(path):
    global iteration_begin, iteration_end, iteration_table
    sofa_traces = None
    with open(path, 'r') as f:
        sofa_traces = f.readlines()
    with open(path, 'w') as f:
        #sofa_traces = f.readlines()
        f.writelines([line for line in sofa_traces[:-1]])
        f.write("\n\n")
        f.write("iteration_detection = ")
        f.write('{"name": "iteration_detection", "color": "rgba(241,156,162,1)", "data": [')    
        for (IT_beg, IT_end) in iteration_table:
            #print("begin:%f end:%f duration:%f"%(IT_beg, IT_end, IT_end-IT_beg))
            for i in range(10):
                dot_y = np.power(10,float(i-5))
                f.write('{"name": "iteration_begin", "x": ' + str(IT_beg) + ', "y": ' + str(dot_y) + '}, {"name": "iteration_end", "x": ' + str(IT_end) +', "y": ' + str(dot_y) + '}, ')
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
    global iteration_begin, iteration_end

    try: 
        iterationDetection(logdir, cfg, df_gpu, 0.01, 0.7, cfg.iterations) 
        traces_to_json(logdir + 'report.js')
        trace_timeline(logdir + 'iteration_timeline.txt')

    except IOError:
        print_warning(
            "gputrace.csv is not found. If there is no need to profile GPU, just ignore it.")

