import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import warnings
from functools import partial
from operator import itemgetter
from random import randint

import cxxfilt
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.cluster import KMeans

from sofa_common import *
from sofa_config import *
from sofa_models import SOFATrace
from sofa_print import *

sofa_fieldnames = [
    "timestamp",  # 0
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
    "category"] # 12

"""
Move sofa_hsg from sofa_preprocess to sofa_hsg

Goal:

step 1 sofa record "the program" --logdir sofalog1

step 2 sofa record "the program" --logdir sofalog2

step 3 sofa diff --base_logdir=sofalog1  --match_logdir=sofalog2
"""

def list_downsample(list_in, plot_ratio):
    new_list = []
    for i in range(len(list_in)):
        if i % plot_ratio == 0:
            # print("%d"%(i))
            new_list.append(list_in[i])
    return new_list

def cpu_trace_read_hsg(sample, t_offset, cfg, cpu_mhz_xp, cpu_mhz_fp):
    fields = sample.split()
    event = event_raw = 0
    counts = 0

    if re.match(r'\[\d+\]', fields[1]) is not None:
        time = float(fields[2].split(':')[0])
        func_name = '[%s]'%fields[4].replace('-','_') + fields[6] + fields[7]
        counts = float(fields[3])
        event_raw = 1.0 * int("0x01" + fields[5], 16)
        # add new column to cpu_traces
        feature_types = fields[3].split(':')[0]
        mem_addr = fields[5]
    else:
        time = float(fields[1].split(':')[0])
        func_name = '[%s]'%fields[3].replace('-','_')  + fields[5] + fields[6]
        counts = float(fields[2])
        event_raw = 1.0 * int("0x01" + fields[4], 16)
        # add new column to cpu_traces
        feature_types = fields[3].split(':')[0]
        mem_addr = fields[4]

    if not cfg.absolute_timestamp:
        time = time - cfg.time_base

    t_begin = time + t_offset
    t_end = time + t_offset

    if len(cpu_mhz_xp) > 1:
        duration = counts/(np.interp(t_begin, cpu_mhz_xp, cpu_mhz_fp)*1e6)
    else:
        duration = counts/(3000.0*1e6)

    event  = np.log10(event_raw)

    if cfg.perf_events.find('cycles') == -1:
        duration = np.log2(event_raw/1e14)

    trace = [t_begin,                          # 0
             event,  # % 1000000               # 1
             duration,                         # 2
             -1,                               # 3
             -1,                               # 4
             0,                                # 5
             0,                                # 6
             -1,                               # 7
             -1,                               # 8
             int(fields[0].split('/')[0]),     # 9
             int(fields[0].split('/')[1]),     # 10
             func_name,                        # 11
             0,                                # 12
             feature_types,                    # 13
             mem_addr]                         # 14
    return trace

def random_generate_color():
        rand = lambda: randint(0, 255)
        return '#%02X%02X%02X' % (rand(), rand(), rand())

def kmeans_cluster(num_of_cluster, X):
    '''
    num_of_cluster: how many groups of data you prefer
    X: input taining data
    '''
    random_state = 170
    try:
        num_of_cluster = 5
        y_pred = KMeans(n_clusters=num_of_cluster, random_state=random_state).fit_predict(X)
    except :
        num_of_cluster = len(X) # minimum number of data
        y_pred = KMeans(n_clusters=num_of_cluster, random_state=random_state).fit_predict(X)

    return y_pred

def sofa_hsg(cfg, swarm_groups, swarm_stats, t_offset, cpu_mhz_xp, cpu_mhz_fp):
    """
    hierarchical swarm generation
    """
    with open(cfg.logdir + 'perf.script') as f, warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        samples = f.readlines()
        print_info(cfg, "Length of cpu_traces for HSG = %d" % len(samples))
        if len(samples) > 0:
            with mp.Pool() as pool:
                res = pool.map(
                    partial(
                        cpu_trace_read_hsg,
                        t_offset = t_offset,
                        cfg = cfg,
                        cpu_mhz_xp = cpu_mhz_xp,
                        cpu_mhz_fp = cpu_mhz_fp
                        ),
                    samples)
            cpu_traces = pd.DataFrame(res)
            sofa_fieldnames_ext = sofa_fieldnames + ["feature_types", "mem_addr"] # mem_addr for swarm-diff
            cpu_traces.columns = sofa_fieldnames_ext
            cpu_traces.to_csv(
                cfg.logdir + 'hsg_trace.csv',
                mode='w',
                header=True,
                index=False,
                float_format='%.6f')
            res_viz = list_downsample(res, cfg.plot_ratio)
            swarm_cpu_traces_viz = pd.DataFrame(res_viz)
            swarm_cpu_traces_viz.columns = sofa_fieldnames_ext

            char1 = ']'
            char2 = '+'
            # demangle c++ symbol, little dirty work here...
            swarm_cpu_traces_viz['name'] = swarm_cpu_traces_viz['name'].apply(
                lambda x: cxxfilt.demangle(str( x[x.find(char1)+1 : x.find(char2)].split('@')[0] ))
            )

            ### N features ###
            ## In order to merge, give unique id of each data within 10 msec by time quotient
            swarm_cpu_traces_viz['quotient'] = swarm_cpu_traces_viz['timestamp'].apply(lambda x: int( x * 1000 // 10)) # //: quotient

            # count feature_types in each 10 msec groups, and create a dictionary for mapping
            df2s = {}
            for quotient, dataframe in swarm_cpu_traces_viz.groupby(['quotient','event']):
                # api value_counts(): return pandas series
                df2s[quotient] = dataframe.feature_types.value_counts()
            df2 = pd.DataFrame.from_dict(df2s, orient='index').fillna(0).astype(np.int64)

            df = swarm_cpu_traces_viz.copy()
            swarm_cpu_traces_viz = pd.merge(df, df2, left_on=['quotient','event'], right_index=True).copy()

            ### swarm seperation by memory location
            #swarm_groups = []
            feature_list = ['event']
            if cfg.hsg_multifeatures:
                with open(cfg.logdir+'perf_events_used.txt','r') as f:
                    lines = f.readlines()
                    feature_list.extend(lines[0].split(','))
                try:
                    feature_list.remove('cycles')
                    feature_list.remove('event')
                except:
                    pass

            print_info(cfg, 'HSG features: '+','.join(feature_list))

            idx = 0
            showing_idx = 0

            if len(cpu_traces) > 0:
                # get memory index by cheange float to integer
                swarm_cpu_traces_viz['event_int'] = swarm_cpu_traces_viz.event.apply(lambda x: int(x)) # add new column 'event_int'
                # swarm seperate
                event_groups = swarm_cpu_traces_viz.groupby('event_int')
                #swarm_stats = []
                # add different swarm groups
                for mem_index, l1_group in event_groups:
                    # kmeans
                    X = pd.DataFrame(l1_group['event'])
                    num_of_cluster = 2
                    y_pred = kmeans_cluster(num_of_cluster, X)

                    # add new column
                    # TODO: Eliminate warning of SettingWithCopyWarning
                    l1_group['cluster'] = y_pred
                    #for i in range(len(y_pred)):
                    #    group.loc[i, 'cluster'] = y_pred[i]

                    # group by new column
                    clusters = l1_group.groupby('cluster')

                    for l2_group_idx, l2_group in clusters:
                        # group by process id
                        #pid_clusters = cluster.groupby('pid')
                        X = pd.DataFrame(l2_group['event'])
                        num_of_cluster = 4
                        y_pred = kmeans_cluster(num_of_cluster, X)

                        # add new column
                        l2_group['cluster'] = y_pred
                        #for i in range(len(y_pred)):
                        #    l2_group.loc[i, 'cluster'] = y_pred[i]

                        # group by new column
                        l3_groups = l2_group.groupby('cluster')

                        for l3_group_idx, l3_group in l3_groups:
                            # kmeans
                            X = pd.DataFrame(l3_group['event'])
                            num_of_cluster = 4
                            y_pred_pid_cluster = kmeans_cluster(num_of_cluster, X)

                            # add new column
                            l3_group['cluster_in_pid'] = y_pred_pid_cluster
                            # group by new column
                            cluster_in_pid_clusters = l3_group.groupby('cluster_in_pid')

                            for mini_cluster_id, cluster_in_pid_cluster in cluster_in_pid_clusters:
                                # duration time
                                total_duration = cluster_in_pid_cluster.duration.sum()
                                mean_duration = cluster_in_pid_cluster.duration.mean()
                                count = len(cluster_in_pid_cluster)
                                # swarm diff
                                # caption: assign mode of function name
                                mode = str(cluster_in_pid_cluster['name'].mode()[0]) # api pd.Series.mode() returns a pandas series
                                mode = mode.replace('::', '@') # str.replace(old, new[, max])
                                # print('mode of this cluster: {}'.format(str(mode[:35]))) # uncomment this line of code when you need to check the mode of cluster

                                swarm_stats.append({'keyword': 'SWARM_' + '["' + str(mode[:35]) + ']' +  ('_' * showing_idx),
                                                    'duration_sum': total_duration,
                                                    'duration_mean': mean_duration,
                                                    'example':cluster_in_pid_cluster.head(1)['name'].to_string().split('  ')[2],
                                                    'count':count})

                                swarm_groups.append({'group': cluster_in_pid_cluster.drop(columns = ['event_int', 'cluster', 'cluster_in_pid']), # data of each group
                                                     'color':  random_generate_color(),
                                                     'keyword': 'SWARM_' + '[' + str(mode[:35]) + ']' +  ('_' * showing_idx),
                                                     'total_duration': total_duration})
                                idx += 1

                swarm_groups.sort(key=itemgetter('total_duration'), reverse = True) # reverse = True: descending
                swarm_stats.sort(key=itemgetter('duration_sum'), reverse = True)
                print_title('HSG Statistics - Top-%d Swarms'%(cfg.num_swarms))

                print('%45s\t%13s\t%30s'%('SwarmCaption', 'ExecutionTime[sum,mean,count] (s)', 'Example'))
                for i in range(len(swarm_stats)):
                    if i >= cfg.num_swarms:
                        break
                    else:
                        swarm = swarm_stats[i]
                        print('%45s\t%.6lf, %.6lf, %6d\t%45s' % (swarm['keyword'], 
                            swarm['duration_sum']/4.0, 
                            swarm['duration_mean']/4.0, 
                            swarm['count'], swarm['example']))

            return swarm_groups, swarm_stats

def sofa_hsg_to_sofatrace(cfg, swarm_groups, traces): # record_for_auto_caption = True # temperarily: for auto-caption
    dummy_i = 0
    auto_caption_filename_with_path = cfg.logdir + 'auto_caption.csv'
    with open(auto_caption_filename_with_path,'w') as f:
        f.close()
    for swarm in swarm_groups[:cfg.num_swarms]:
        if cfg.display_swarms:
            sofatrace = SOFATrace() # file.class
            sofatrace.name = 'swarm' + str(dummy_i) # avoid errors casued by JavaScript. No special meaning, can be random unique ID.
            sofatrace.title = swarm['keyword'] # add number of swarm
            sofatrace.color = swarm['color']
            sofatrace.x_field = 'timestamp'
            sofatrace.y_field = 'duration'
            sofatrace.data = swarm['group'].copy()
            traces.append(sofatrace)

        # append to csv file every time using pandas funciton
        swarm['group']['cluster_ID'] = dummy_i # add new column cluster ID to dataframe swarm['group']
        copy = swarm['group'].copy()
        #print('*************************')
        
        copy.to_csv(auto_caption_filename_with_path, mode='a', header=False, index=False)
        #print('\nRecord for auto-caption, data preview: \n{}'.format(copy.head(2)))
        #print('*************************')
        # --- for auto-caption --- #
        dummy_i += 1
    csv_input = pd.read_csv(auto_caption_filename_with_path, names=list(copy))
    if 'instructions' not in copy.columns:
        csv_input.insert(17, 'instructions', 0)
    if 'cache-misses' not in copy.columns:
        csv_input.insert(18, 'cache-misses', 0)
    if 'branch-miss' not in copy.columns:
        csv_input.insert(19, 'branch-misses', 0)
    csv_input.to_csv(auto_caption_filename_with_path, header=False)
    return traces

def matching_two_dicts_of_swarm(standard_dict, matching_dict, res_dict):
    """
    String Matching Funciton:
    match two dictoinaries with same amount of key-value pairs
    and return matching result, a dict of dict called res_dict.

    * standard_dict: The standard of dict
    * matching_dict: The dict that i want to match
    * res_dict: the result, a dict of dict
    """
    key = 0 # key: number, no string
    pop_list = [k for k,v in matching_dict.items()]
    #print(pop_list)
    for i in standard_dict.keys(): # control access index of standard_dict. a more pythonic way
        threshold = 0
        for j in pop_list: # control access index of matching_dict
            f_ratio = fuzz.ratio(standard_dict[i], matching_dict[j])
            if f_ratio > threshold: # update matching result only when the fuzz ratio is greater
                #print('New matching fuzz ratio {} is higher than threshold {}'\
                #      .format(f_ratio, threshold))
                key = j # update key
                threshold = f_ratio # update threshold value
                #print('Update new threshold {}'\
                #      .format(threshold))
                res_dict.update({i: {j: matching_dict[i]}}) #
        # pop out matched key-value pair of matching dict
        if pop_list:
            pop_list.remove(key) # remove specific value. remove() fails when no elements remains
        #print(res_dict)

    return res_dict # return result dict

def evaluation_of_matching_result(base_df, matching_df1, final_df, eval_list, tmp_dict):
    """
    calculate intersection rate of two dataframe
    intersection rate = num_t_stdswarm  / total_num_t_mtchswarm
    num_t_stdswarm: traces in standard swarm
    total_num_t_mtchswarm: total traces number in matching swarm
    """
    base_duration_list = []
    match_duration_list = []
    diff_list = []

    # calculate num_t_stdswarm & total_num_t_mtchswarm
    for id_of_cluster in final_df.index:
        base_id = final_df['base_cluster_ID'].loc[id_of_cluster]
        bs_df = base_df.groupby(['cluster_ID','function_name'])\
                        .agg({'function_name':['count']})\
                        .loc[base_id]\
                        .reset_index()
        bs_df.columns = ['base_func_name', 'count']
        # sum up duration time
        base_total_duration = base_df['duration'].loc[base_df['cluster_ID'] == id_of_cluster].sum()
        #print('base_total_duration = {} sec'.format(base_total_duration))
        #print('Function name in cluster: \n{}\n'.format(bs_df.sort_values(by=['count'], ascending=False)))

        # total_num_t_mtchswarm
        match_id = final_df['match_cluster_ID'].loc[id_of_cluster]

        match_df = matching_df1.groupby(['cluster_ID','function_name'])\
                        .agg({'function_name':['count']})\
                        .loc[match_id]\
                        .reset_index()
        match_df.columns = ['match_func_name', 'count']
        # sum up duration time
        match_total_duration = matching_df1['duration'].loc[matching_df1['cluster_ID'] == id_of_cluster].sum()

        total_num_t_mtchswarm = match_df['count'].sum()
        #print('match_total_duration = {} sec'.format(match_total_duration))
        #print('Function name in cluster: \n{}\n'.format(match_df.sort_values(by=['count'], ascending=False)))
        #print('---------------------------------------------------------')
        #print('Total number of function name in cluster: {}'.format(total_num_t_mtchswarm))

        # add total duration of each cluster
        base_duration_list.append(base_total_duration)
        match_duration_list.append(match_total_duration)
        diff_list.append(abs(base_total_duration - match_total_duration))

        # To calculate num_t_stdswarm, get intersection of two cluster first
        intersected_df = bs_df.merge(match_df, left_on='base_func_name', right_on='match_func_name', how='outer')
        intersected_df.dropna(inplace=True) # drop row with NaN value and inplace
        intersected_df['min_value'] = intersected_df.min(axis=1)
        num_t_stdswarm = intersected_df['min_value'].sum()
        intersect_percent = num_t_stdswarm * 100 / float(total_num_t_mtchswarm) # float number

        if(intersect_percent != 0.0):
            eval_list.append(intersect_percent)

        #print('merge frame:\n {}\n'.format(intersected_df))
        #print('num_t_stdswarm = {}'.format(num_t_stdswarm))
        #print('intersection rate = (num_t_stdswarm / total_num_t_mtchswarm) x 100% = {}%'.format(intersect_percent))
        #print('---------------------------------------------------------')
        #break; # test only one cluster

    # How many cluster match correctly
    intersect_percent = len(eval_list) * 100.0 / len(base_df['cluster_ID'].unique())
    #print('Number of intersection rate > 0% percent: {}%'.format(intersect_percent)) #

    # deal with duration time of each cluster among two dataframes
    tmp_dict = {'base_duration(sec)': base_duration_list, 'match_duration(sec)': match_duration_list, 'cluster_diff(sec)': diff_list}
    tmp_df = pd.DataFrame.from_dict(tmp_dict) # dummy dataframe, just for concatenation
    final_df = pd.concat([final_df, tmp_df], axis=1, sort=False)  # axis=1: horizontal direction
    print('Diff Report: \n{}'.format(final_df))

    return final_df # return final_df in case information lost

def sofa_swarm_diff(cfg):
    """
    swarm diff: design for auto-caption. compare two different sofalog
    """

    #print('Python verison: {}'.format(sys.version)) # check python version

    column_list = ["timestamp", "event", "duration",
                "deviceId", "copyKind", "payload",
                "bandwidth", "pkt_src", "pkt_dst",
                "pid", "tid", "function_name", "category",
                "feature_types", "mem_addr", "quotient",
                "cycles", "instructions", "cache-misses", "branch-misses",
                "cluster_ID"]
    base_df = pd.read_csv(cfg.base_logdir + 'auto_caption.csv', names=column_list)
    #print(base_df)
    #print('There are {} clusters in standard_df\n'.format(len(base_df['cluster_ID'].unique())))

    base_df_groupby = base_df.groupby(['cluster_ID','function_name']).agg({'function_name':['count']})

    ## --- Need refactor here --- ##

    ## Access data of multiIndex dataframe
    # get column names
    #TODO: fix bug of 'the label [0] is not in the [index]' 
    print(base_df_groupby)
    df = base_df_groupby.loc[0].reset_index()
    flat_column_names = []
    for level in df.columns:
        # tuple to list
        flat_column_names.extend(list(level)) # extend(): in-place
    if '' in flat_column_names:
        flat_column_names.remove('')
    # remove duplicate and empty
    #flat_column_names = filter(None, flat_column_names) # filter empty
    flat_column_names = list(set(flat_column_names)) # deduplicate
    print('original order: {}'.format(flat_column_names))

    # change member order of list due to set is a random order
    if flat_column_names[0] == 'count':
        myorder = [1,0]
        flat_column_names = [flat_column_names[i] for i in myorder]
        # print('New order: {}'.format(flat_column_names))

    base_df_dict = {}
    # Transform multi-index to single index, and update string to dict standard_df_dict
    for id_of_cluster in base_df['cluster_ID'].unique():
        #print('\nCluster ID : {}'.format(id_of_cluster))
        df = base_df_groupby.loc[id_of_cluster].reset_index()
        df.columns = flat_column_names
        #print(df.sort_values(by=['count'], ascending=False)) # pd.DataFrame.sort_values() return a DataFrame
        base_df_dict.update({id_of_cluster: df.function_name.str.cat(sep='  ', na_rep='?')})

    ## Dataframe that i want to match
    matching_df1 = pd.read_csv(cfg.match_logdir + 'auto_caption.csv', names=column_list)
    matching_df1_groupby = matching_df1.groupby(['cluster_ID','function_name']).agg({'function_name':['count']})

    # get column names
    df = matching_df1_groupby.loc[0].reset_index()
    flat_column_names = []
    for level in df.columns:
        # tuple to list
        flat_column_names.extend(list(level)) # extend(): in-place

    # remove duplicate and empty
    flat_column_names = filter(None, flat_column_names) # filter empty
    flat_column_names = list(set(flat_column_names)) # deduplicate
    # print(flat_column_names)

    # change member order of list due to set is a random order
    if flat_column_names[0] == 'count':
        myorder = [1,0]
        flat_column_names = [flat_column_names[i] for i in myorder]
        # print('New order: {}'.format(flat_column_names))

    matching_df1_dict = {}

    # Transform multi-index to single index, and update string to dict standard_df_dict
    for id_of_cluster in matching_df1['cluster_ID'].unique(): 
        #print('\nCluster ID : {}'.format(id_of_cluster))
        df = matching_df1_groupby.loc[id_of_cluster].reset_index()
        df.columns = flat_column_names
        # print(df.sort_values(by=['count'], ascending=False))

        matching_df1_dict.update({id_of_cluster: df.function_name.str.cat(sep='  ', na_rep='?')})
    ## --- Need refactor here --- ##

    res_dict = {}
    res_dict = matching_two_dicts_of_swarm(base_df_dict, matching_df1_dict, res_dict)
    ## show all stats (Ans) and matching results (algorithm)
    base_dict_to_df = pd.DataFrame.from_dict(base_df_dict, orient='index', columns=['Before: function_name'])
    base_dict_to_df['base_cluster_ID'] = base_dict_to_df.index
    base_dict_to_df = base_dict_to_df[['base_cluster_ID', 'Before: function_name']]

    res_dict_to_df = pd.DataFrame() # create an empty frame
    res_list = [k for k,v in res_dict.items()]
    for key in res_list:
        df = pd.DataFrame.from_dict(res_dict[key], orient='index', columns=['After: funciton name']) # res_dict[key]: a dict
        df['match_cluster_ID'] = df.index
        res_dict_to_df = res_dict_to_df.append(df, ignore_index=True) # df.append(): not in-place

    res_dict_to_df = res_dict_to_df[['match_cluster_ID', 'After: funciton name']]
    final_df = pd.concat([base_dict_to_df, res_dict_to_df], axis=1)

    ## Evaluation: Evaluate matching result of single run, then all different runs
    eval_list = []
    tmp_dict = {}
    final_df = evaluation_of_matching_result(base_df, matching_df1, final_df, eval_list, tmp_dict)

    ## Output result
    log_list = []
    log_list = cfg.base_logdir.split("/")
    log_list.remove(log_list[-1]) # list.remove() removes element in place
    log_str = '/'.join(log_list) # str.join() returns str
    output_logdir =  log_str + '/' + 'sofalog/' # please adjust the output directory path to fit your need
    if not os.path.exists(output_logdir): # df.to_csv does not create directory automatically, create it manually
        os.makedirs(output_logdir)
    final_df.to_csv(os.path.join(output_logdir, 'swarm_diff.csv'))

    # check result
    #print(final_df.head(10))
    print('-------------------------------------')
    print('Output file: {}'.format( os.path.join(output_logdir, 'swarm_diff.csv')))
    print('-------------------------------------')
    return final_df
