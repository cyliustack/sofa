#!/usr/bin/python 
import os 
import subprocess
from scapy.all import *
import sqlite3
import pandas as pd
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp 
import glob, os 
from functools import partial


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Xring Modeling')
    parser.add_argument('--logdir', metavar="/path/to/logdir/", type=str, required=False, 
                    help='path to the directory of profiling log files')
    parser.add_argument('--max_num_gpus', metavar="N", type=int, required=False,
                    help='specify maximum number of GPUs to model')
    parser.add_argument('--metric', type=str, required=False, metavar='metric',
                    help='performance metric, like hotspot, memory pressure')
    parser.add_argument('command', type=str, nargs=1, metavar='command',
            help='specify a command: [record|report]')
 
    logfile = 'xring-report.txt' 
    args = parser.parse_args()
    command = args.command[0]
    max_num_gpus = args.max_num_gpus

    if command == 'record':
        os.system("echo XRING TEST > "+logfile)  
        for i in range(1,max_num_gpus):
            print("Test xring for GPUx%d" % (i+1) )
            os.system("sofa stat  python tf_cnn_benchmarks.py --model=vgg16 --batch_size=64 --variable_update=replicated --num_gpus=%d --local_parameter_device=gpu --num_batches=10 --all_reduce_spec=xring 1>> %s " % ( i+1, logfile ) )
        
    if command == 'report':
        total_traffic = []
        total_h2d_traffic = []
        total_d2h_traffic = []
        total_p2p_traffic = []
        with open(logfile) as f:
            lines = f.readlines()
            for line in lines:
                if line.find("MeasuredTotalTraffic") != -1 :
                    traffic = int(float(line.split()[2])) 
                    total_traffic.append(traffic)
                if line.find("MeasuredTotalH2DTraffic") != -1 :
                    traffic = int(float(line.split()[2])) 
                    total_h2d_traffic.append(traffic)
                if line.find("MeasuredTotalD2HTraffic") != -1 :
                    traffic = int(float(line.split()[2])) 
                    total_d2h_traffic.append(traffic)
                if line.find("MeasuredTotalP2PTraffic") != -1 :
                    traffic = int(float(line.split()[2])) 
                    total_p2p_traffic.append(traffic)

        with open("xring.csv", "w") as outcsv:
            writer = csv.writer(outcsv)
            writer.writerow(['GPUx2', 'GPUx3', 'GPUx4',  'GPUx5', 'GPUx6', 'GPUx7', 'GPUx8',])
            writer.writerow(total_traffic)        
            writer.writerow(total_h2d_traffic)        
            writer.writerow(total_d2h_traffic)        
            writer.writerow(total_p2p_traffic)        
        print('Total Traffic, H2D, D2H, P2P (MB)')
        print(total_traffic)
        print(total_h2d_traffic) 
        print(total_d2h_traffic) 
        print(total_p2p_traffic) 
