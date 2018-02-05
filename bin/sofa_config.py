from sofa_print import *
import json
def read_config(path_cfg):
    cfg = json.loads('{"filters":[{"keyword":"idle","color":"cadeblue"}, {"keyword":"flush", "color":"#00BFFF"} ], "enable_verbose":"false", "enable_plot_bandwidth":"false", "top_k":"20", "gpu_filters":[{"keyword":"copyKind1","color":"red"}, {"keyword":"copyKind2", "color":"lightblue"} , {"keyword":"copyKind10", "color":"purple"}] }')
    try:
        with open(path_cfg) as f:
            cfg = json.load(f)
    except:
        with open( 'sofa.cfg', "w") as f:
            json.dump(cfg,f)
            f.write("\n")
    #print_info("SOFA Configuration: ")    
    #print(cfg)
    return cfg
