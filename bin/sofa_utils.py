import json
import pandas as pd 
import itertools
import os
from collections import defaultdict, OrderedDict

def parse_pyflame(file_path):

    func_dict = {}
    
    with open(file_path, 'r') as f:
        for time_stamp, funct_stack in itertools.zip_longest(*[f] * 2):
            funct_stack = funct_stack.replace('\n', '').replace(';', '<br>')
            func_dict[int(time_stamp)] = funct_stack
        func_dict = OrderedDict(func_dict)

        duration = OrderedDict()
        prev = None
        for key, values in func_dict.items():
            if prev is None:
                prev = key
                continue
            duration[prev] = key - prev
            prev = key
        del func_dict[max(func_dict.keys())]

    return func_dict, duration