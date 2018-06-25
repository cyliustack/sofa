import pandas as pd
import os
import sys

sofa_fieldnames = [
            'timestamp',  # 0
            "event",  # 1
            "duration",  # 2
            "deviceId",  # 3
            "copyKind",  # 4
            "payload",  # 5
            "bandwidth",  # 6
            "pkt_src",  # 7
            "pkt_dst",  # 8
            "pid",  # 9
            "tid",  # 10
            "name",  # 11
            "category"]  # 12

TF_fieldnames = [
        'NAME',  # 0
        "t_start",
        "timestamp", #t_start_time
        "t_end",
        "t_end_time"]  

def sofa_tf_prepare(_file,logdir):
    print('In sofa_tf_prepare')
    sub_string = 't_start'
    with open(logdir + _file) as f:
        with open(logdir + '/prepare_tmp.csv','w') as df:
            for line in f:
                if sub_string in line:
                    df.write(line)
    df = pd.read_csv(logdir + '/prepare_tmp.csv')
    df.columns = TF_fieldnames
    prepare_traces = pd.DataFrame()
    prepare_traces['timestamp'] = df['timestamp']
    prepare_traces['duration'] = df['t_end_time'] - df['timestamp']
    prepare_traces['name'] = df['NAME']
    for col in ('payload', 'bandwidth', 'category'):
        prepare_traces[col] = 0
    for col in ('event', 'deviceId', 'copyKind', 'pkt_src', 'pkt_dst', 'pid', 'tid'):
        prepare_traces[col] = -1
    prepare_traces[sofa_fieldnames].to_csv(
            logdir + '/cputrace.csv',
             mode='a',
             header=False,
             index=False,
             float_format='%.6f')
    print('sofa_tf_prepare done!')
    return prepare_traces

