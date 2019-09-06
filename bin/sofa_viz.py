import os
import subprocess
import sys
from functools import partial

from sofa_config import *
from sofa_print import *


def sofa_viz(cfg):
    print_warning(cfg,
        'If your rendering timeline is slow, please try \033[4msofa report --plot_ratio=10\033[24m to downsample scatter points,')
    print_warning(cfg, 'and then \033[4msofa viz\033[24m to see the downsampled results.')
    print_hint('SOFA Vlization is listening on port \033[4m\033[97mhttp://localhost:%d\033[24m\033[0m\033[24m' % (cfg.viz_port) )
    print_hint('To change port, please run command: \033[4msofa viz --viz_port=PortNumber\033[24m')
    print_hint('Please open your browser to start profiling.')
    print_hint('After profiling, please enter Ctrl+C to exit.')
    os.system('cd %s && python3.6 -m http.server %d 2>&1 1> /dev/null; cd -' % (cfg.logdir,cfg.viz_port))
