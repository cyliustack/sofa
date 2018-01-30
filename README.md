# Introduction
SOFA: Swarm of Functions Analysis
Authors: All the contributors of SOFA

# Prerequisite
1. Run `./tools/prerequisite.sh` to install all the necessary packages and python packages.
2. [OPTIONAL] Run `./tools/empower-tcpdump.sh $(whoami)` to make network related events tracable in SOFA. After running this step, it is required to __re-login__ to __APPLY THE CHANGES__!!!

## Verify Permissions
Run the following simple tests to verify the permission settings of executing tcpdump under your permission.
* `tcpdump -w sofa.pcap`
* `tcpdump -r sofa.pcap`

# Installation

1. Simply run `./install.sh </PATH/TO/INSTALL>` to install SOFA on your system. Note that `sofa` will be appended to the path if the last directory is not sofa.
2. Then, run `source </PATH/TO/INSTALL>/sofa/tools/activate.sh` to activate SOFA running environment. (Need to be executed on each new shell.)
3. [ALTERNATIVE] Add `source </PATH/TO/INSTALL>/sofa/tools/activate.sh` in your `~/.bashrc` to make this environment available on every shells.

# Usages
SOFA supports serveral different usages, like how one can use perf.

## Basic Statistics
* `sofa stat ping google.com.tw -c 3`

## Performance Visualizations
1. `sofa record ping google.com.tw -c 3`
2. `sofa report`
3. Open browser with one of the following links for different visualizations
    * [http://localhost:8000](http://localhost:8000)
    * [http://localhost:8000/cpu-report.html](http://localhost:8000/cpu-report.html)
    * [http://localhost:8000/gpu-report.html](http://localhost:8000/gpu-report.html)

## Run with different __plugins__
1. Run `sofa record ping google.com.tw -c 3` __only once__ to record the events.
2. Run `sofa preprocess` __only once__ to process the events.
3. Run `sofa analyze` with __A__ plugin.
4. Run `sofa analyze` with __B__ plugin.
5. Run `sofa analyze` with __C__ plugin.


# Configurations

## Where to place my configuration file?
When running SOFA, the script will search the current working directory for the configuration file, `sofa.cfg`.
If the config file does not present, SOFA tries to search through the parent directories from the current working directory till reaching the home directory, i.e., `~/`.

User can create `~/sofa.cfg` as a global default setting for all the projects, and create a `sofa.cfg` file in each project for different purposes.

## Add items in filter lists
To be edited. (How to add new items)


# Examples of Visualization Results:
`sofa record ~/cuda_samples/1_Utilities/bandwidthTest/bandwidthTest`
![Alt text](./figures/demo0.png)
`sofa record ./tools/gpu-train.sh resnet50 64 8 --num_warmup_batches=0 --num_batches=1`
![Alt text](./figures/demo1.png)
`sofa record ./tools/gpu-train.sh resnet50 64 8 --num_warmup_batches=0 --num_batches=1`
![Alt text](./figures/demo2.png)
`sofa record mpirun -f hosts.txt -n 4 ./compute_pi`
![Alt text](./figures/demo3.png)
`sofa record ./tools/gpu-train.sh resnet50 32 1 --num_batches=10`
![Alt text](./figures/demo4.png)
