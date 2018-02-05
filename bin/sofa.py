#!/usr/bin/python
import numpy as np
import csv
import json
import sys
import argparse
import multiprocessing as mp 
import glob, os 
from functools import partial
from sofa_config import *
from sofa_print import *
from sofa_record import *
from sofa_preprocess import *

if __name__ == "__main__":
    
    logdir   = './sofalog/' 
    cfg_file = './sofa.cfg'
    command = None
    usr_command = None
    sys.stdout.flush() 
 
    parser = argparse.ArgumentParser(description='SOFA')
    parser.add_argument("--logdir", metavar="/path/to/logdir/", type=str, required=False, 
                    help='path to the directory of SOFA logged files')
    parser.add_argument('--config', metavar="/path/to/sofa.cfg", type=str, required=False,
                    help='path to the directory of SOFA configuration file')
    parser.add_argument('command', type=str, nargs=1, metavar='<stat|record|report|preprocess|analyze|visualize>')
    parser.add_argument('usr_command', type=str, nargs='?', metavar='<PROFILED_COMMAND>')

    args = parser.parse_args()
    if args.logdir != None:
        logdir = args.logdir + '/'
    
    if args.config != None:
        cfg_file = args.config
    if args.command != None:
        command = args.command[0]
    if args.usr_command != None:
        usr_command = args.usr_command

    print_info("logdir = %s" % logdir )
    print_info("config = %s" % cfg_file )
    cfg = read_config(cfg_file)
        
    if command == 'stat':
        sofa_record(usr_command, logdir, cfg)
        sofa_preprocess(logdir, cfg)
        #sofa_analyze(logdir, cfg)
    elif command == 'record':
        sofa_record(usr_command, logdir, cfg)
    elif command == 'preprocess':
        sofa_preprocess(logdir, cfg)
    elif command == 'analyze':
        sofa_analyze(logdir, cfg)
    elif command == 'report':
        sofa_preprocess(logdir, cfg)
        #sofa_analyze(logdir, cfg)
        #sofa_visualize(logdir, cfg)
    else:
        print_error("Cannot recognized SOFA-command [%s]" % command )
        quit()
    #sofa_analyze(logdir, cfg)
