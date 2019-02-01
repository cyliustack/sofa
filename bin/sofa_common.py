#!/usr/bin/python3.6
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import scipy

from sofa_config import *
from sofa_print import *


def overlap(pa, pb, pc, pd):
    if pb - pc >= 0 and pd - pa >= 0:
        return min(pb, pd) - max(pa, pc)

def partial_sum(df):
    psum = 0

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

# print_format_table()
cktable = {-1: "NON", 0: "KER", 1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
ckindex = [1, 2, 8, 10]

def comm_profile(logdir, cfg, df_gpu, features):
    total_payload = 0.0
    total_h2d_payload = 0.0
    total_d2h_payload = 0.0
    total_d2d_payload = 0.0
    total_p2p_payload = 0.0
    h2d_bandwidth = 16.0
    d2h_bandwidth = 16.0
    d2d_bandwidth = 16.0
    p2p_bandwidth = 25.0
    h2d_time = 0
    d2h_time = 0
    d2d_time = 0
    p2p_time = 0
    total_memcopy_time = 0.0

    # sofa_fieldnames = [
    #    'timestamp',
    #    "event",
    #    "duration",
    #    "deviceId",
    #    "copyKind",
    #    "payload",
    #    "bandwidth",
    #    "pkt_src",
    #    "pkt_dst",
    #    "pid",
    #    "tid",
    #    "name",
    #    "category"]
    n_gpus = 0
    for i in range(len(df_gpu)):
        if df_gpu.iat[i, 3] > n_gpus:
            n_gpus = int(df_gpu.iat[i, 3])

    if n_gpus == 0:
        print_warning("No GPU communication traces are collected.")
        return

    print("\n=========== Payload report for data-copy kinds =================")
    h2d_payload = d2h_payload = d2d_payload = p2p_payload = 0
    groups = df_gpu.groupby(by='copyKind')['payload']
    payload_table = {1:0, 2:0, 8:0, 10:0}
    for key, item in groups:
        if key == 1 or key == 2 or key == 8 or key == 10 :
            payload_table[key] = groups.get_group(key).sum()
            print('%s : %.2lf (MB)' % (cktable[key], payload_table[key]/np.power(1024,2)))

    print("\n=========== Bandwidth report for large data-copy (64kb+) =======")
    h2d_bandwidth = d2h_bandwidth = d2d_bandwidth = p2p_bandwidth = 16
    groups = df_gpu.query('payload > 64000').groupby(by='copyKind')['bandwidth']
    bw_table = {1:0, 2:0, 8:0, 10:0}
    for key, item in groups:
        if key == 1 or key == 2 or key == 8 or key == 10 :
            bw_table[key] = groups.get_group(key).mean()
            print('%s : %.2lf (gb/s)' % (cktable[key], bw_table[key]))

    print("\n=========== Duration report for large data-copy (64kb+) =======")
    h2d_bandwidth = d2h_bandwidth = d2d_bandwidth = p2p_bandwidth = 16
    groups = df_gpu.groupby(by='copyKind')['duration']
    duration_table = {1:0, 2:0, 8:0, 10:0}
    for key, item in groups:
        if key == 1 or key == 2 or key == 8 or key == 10 :
            duration_table[key] = groups.get_group(key).sum()
            print('%s : %.2lf (s)' % (cktable[key], duration_table[key]))

    print("\n=========== Data-copy payload matrix (MB) ======================")
    accum = np.zeros((1 + n_gpus, 1 + n_gpus))
    accum_bw = np.zeros((1 + n_gpus, 1 + n_gpus))
    accum_count = np.zeros((1 + n_gpus, 1 + n_gpus))
    accum_bw_count = np.zeros((1 + n_gpus, 1 + n_gpus))

    # TODO: Parallelize payload accumulatoin
    #print("df length: %d" % len(df_gpu))
    #cpu_count = mp.cpu_count()
    #pool = mp.Pool(processes=cpu_count)
    #res_accum = pool.map( partial(payload_sum), df_gpu)

    for i in range(len(df_gpu)):
        if df_gpu.iat[i, 4] == 0 or df_gpu.iat[i, 4] == 8:
            continue
        src = df_gpu.iat[i, 7]
        dst = df_gpu.iat[i, 8]
        payload = df_gpu.iat[i, 5]
        bandwidth = df_gpu.iat[i, 6]
        accum[src][dst] = float(accum[src][dst] + payload)
        accum_count[src][dst] = int(accum_count[src][dst] + 1)
        if payload > 64000:
            accum_bw[src][dst] = float(accum_bw[src][dst] + bandwidth)
            accum_bw_count[src][dst] = int(accum_bw_count[src][dst] + 1)

    row_str = "\tHOST\t"
    for i in range(1, accum.shape[1]):
        row_str = row_str + "GPU%d" % i + "\t"
    print(row_str)
    for i in range(accum.shape[0]):
        if i == 0:
            row_str = "HOST\t"
        else:
            row_str = "GPU%d\t" % i

        for j in range(accum.shape[1]):
            row_str = row_str + "%d" % (accum[i][j] / (1024 * 1024)) + "\t"
        print(row_str)


    print("\n=========== Data-Copy Bandwidth Matrix (GB/s) =================")
    row_str = "\tHOST\t"
    for i in range(1, accum_bw.shape[1]):
        row_str = row_str + "GPU%d" % i + "\t"
    print(row_str)
    for i in range(accum_bw.shape[0]):
        if i == 0:
            row_str = "HOST\t"
        else:
            row_str = "GPU%d\t" % i

        for j in range(accum_bw.shape[1]):
            if  accum_bw_count[i][j] > 0:
                row_str = row_str + "%.2lf" % (accum_bw[i][j]/accum_bw_count[i][j]) + "\t"
            else:
                row_str = row_str + "0" + "\t"
        print(row_str)


    df_gpu.to_csv(
        logdir + '/' + 'comm.csv',
        columns=[
            "timestamp",
            "pkt_src",
            "pkt_dst",
            "payload",
            "bandwidth"])
    
    df = pd.DataFrame({ 'name':['total_h2d_payload', 'total_d2h_payload', 'total_d2d_payload', 'total_p2p_payload', 
                                'h2d_bandwidth', 'd2h_bandwidth', 'd2d_bandwidth', 'p2p_bandwidth', 
                                'h2d_time', 'd2h_time', 'd2d_time', 'p2p_time'], 
                        'value':[ payload_table[1], payload_table[2], payload_table[8], payload_table[10], 
                                  bw_table[1], bw_table[2], bw_table[8], bw_table[10],  
                                  duration_table[1], duration_table[2], duration_table[8], duration_table[10] ]}, 
                        columns=['name','value'])
    features = pd.concat([features, df])
    return features
