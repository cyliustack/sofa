from sofa_print import *
import json
def read_config(path_cfg):
    cfg = json.loads('{ "filters":[\
                            {"keyword":"nv_alloc_system_pages","color":"Chartreuse"},\
                            {"keyword":"idle","color":"cadeblue"}],\
                        "enable_verbose":"false",\
                        "enable_plot_bandwidth":"false",\
                        "top_k":"20",\
                        "gpu_filters":[\
                            {"keyword":"copyKind_1_","color":"Red"},\
                            {"keyword":"copyKind_2_", "color":"Peru"} ,\
                            {"keyword":"copyKind_10_", "color":"Purple"}]\
                        }')
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
