import os
import sys
from functools import partial
from sofa_print import *
from sofa_config import *
import subprocess


def sofa_viz(logdir, cfg):
    sofa_home = os.path.dirname(os.path.realpath(__file__))
    subprocess.Popen(
        ['bash', '-c', 'cp %s/../sofaboard/* %s;' % (sofa_home, logdir)])

    subprocess.Popen(['sleep', '2'])
    print_warning(
        'If your rendering timeline is slow, please try \033[4msofa report --plot_ratio=10\033[24m to downsample scatter points.')
    print_info(
        'SOFA Visualization is listening on port \033[4m\033[97mhttp://localhost:8000\033[24m\033[0m\033[24m')
    print_info('Please open your browser to start profiling.')
    print_info('After profiling, please enter Ctrl+C to exit.')
    os.system(
        'cd %s && python -m SimpleHTTPServer 1> /dev/null; cd -' %
        (logdir))
