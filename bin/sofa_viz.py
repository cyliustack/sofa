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
        'If your rendering timeline is slow, please try \033[4msofa preprocess --plot_ratio=10\033[24m to downsample scatter points,')
    print_warning('and then \033[4msofa viz\033[24m to see the downsampled results.')
    print_info(
        'SOFA Visualization is listening on port \033[4m\033[97mhttp://localhost:%d\033[24m\033[0m\033[24m' % (cfg.viz_port) )
    print_info('To change port, please run command: \033[4msofa viz --viz_port=PortNumber\033[24m')
    print_info('Please open your browser to start profiling.')
    print_info('After profiling, please enter Ctrl+C to exit.')
    os.system(
        'cd %s && python -m SimpleHTTPServer %d 2>&1 1> /dev/null; cd -' %
        (logdir,cfg.viz_port))
