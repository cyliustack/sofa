class Filter:
    keyword = None
    color = None

    def __init__(self, keyword, color):
        self.keyword = keyword
        self.color = color


class SOFA_Config:
    cpu_filters = []
    gpu_filters = []
    net_filters = []
    cluster_ip = []
    perf_events = 'cycles'
    verbose = False
    num_iterations = 20
    num_swarms = 10
    cpu_top_k = 20
    plot_ratio = 1
    viz_port = 8000
    cpu_time_offset = 0
    profile_all_cpus = False
    enable_aisi = False
    aisi_via_strace = False
    strace_min_time = 1e-6
    display_swarms = False
    hsg_multifeatures = False
    use_diff = False # output csv file for sofa swarm-diff
    swarm_diff = False
    logdir = './sofalog/'
    base_logdir = ''
    match_logdir = ''
    enable_vmstat = False
    enable_mpstat = True
    enable_diskstat = True
    enable_pcm = False
    cuda_api_tracing = False
    script_path = ''
    potato_server = None
    elapsed_time = 0
    absolute_timestamp = False
    time_base = 0
    pid = -1
    timeout = 30
    columns = ['timestamp', 'duration']
    enable_strace = False
    enable_py_stacks = False
    blktrace_device = None
    roi_begin=0
    roi_end=0
