import os
import sys
from functools import partial
from sofa_print import *
from sofa_config import *
import subprocess

def sofa_viz(logdir, cfg):
    sofa_home = os.path.dirname(os.path.realpath(__file__))
    subprocess.Popen(['bash', '-c', 'set -x -e; cp %s/../sofaboard/* %s; set +x +e'%(sofa_home,logdir) ]) 
    print_info('Enter Ctrl+C twice to exit~')
    os.system('cd %s && python -m SimpleHTTPServer; cd -'%(logdir))
