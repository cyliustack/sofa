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

# print_format_table()
cktable = {-1: "NON", 0: "KER", 1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P"}
ckindex = [1, 2, 8, 10]

def comm_profile(logdir, cfg, df_gpu, features):
    total_traffic = 0.0
    total_h2d_traffic = 0.0
    total_d2h_traffic = 0.0
    total_p2p_traffic = 0.0
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

    print("Data Traffic for each CopyKind (MB)")
    data_copyKind = grouped_df = df_gpu.groupby("copyKind")["payload"]
    for key, item in grouped_df:
        if int(key) == 1:
            total_h2d_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 2:
            total_d2h_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) == 10:
            total_p2p_traffic = grouped_df.get_group(key).sum() / 1000000.0
        if int(key) != 8:
            total_traffic = total_traffic + \
                grouped_df.get_group(key).sum() / 1000000.0
    print(("Total traffic: %.2lf" % total_traffic))

    print("Data Communication Time for each CopyKind (s)")
    durations_copyKind = grouped_df = df_gpu.groupby("copyKind")["duration"]
    for key, item in grouped_df:
        if key == 0:
            total_kernel_time = grouped_df.get_group(key).sum()
        else:
            total_memcopy_time = total_memcopy_time + \
                grouped_df.get_group(key).sum()

    bw = (data_copyKind.sum() / 1000000) / durations_copyKind.sum() / 1000
    bw_h2d = bw_d2h = bw_p2p = avg_bw = 1e-10

    for i in range(len(bw)):
        key = list(bw.keys())[i]
        if cktable[key] == 'H2D' or cktable[key] == 'D2H' or cktable[key] == 'D2D' or cktable[key] == 'P2P':
            print(("Averaged Achieved %s Unidirectional Bandwidth: %.1f (GB/s)" % (cktable[key], bw.iloc[i])))
        else:
            continue

    print("Bandwidth Report for Large Data-copy (64KB+)")
    gp = df_gpu.query('payload > 64000').groupby(by='copyKind')['bandwidth']
    for key, item in gp:
        print(("[%s]: %.3lf (GB/s)" % (cktable[key], gp.get_group(key).mean())))

    print("Summary of Comm.")

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
    
    df = pd.DataFrame({ 'name':['total_h2d_traffic', 'total_d2h_traffic', 'total_p2p_traffic'], 
                        'value':[total_h2d_traffic, total_d2h_traffic, total_p2p_traffic] }, 
                        columns=['name','value'])
    features = pd.concat([features, df])
    return features
