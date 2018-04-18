class Filter:
    keyword = None
    color = None

    def __init__(self, keyword, color):
        self.keyword = keyword
        self.color = color


class SOFA_Config:
    cpu_filters = []
    gpu_filters = []
    verbose = False
    cpu_top_k = 20
    plot_ratio = 1
    profile_all_cpus = False
